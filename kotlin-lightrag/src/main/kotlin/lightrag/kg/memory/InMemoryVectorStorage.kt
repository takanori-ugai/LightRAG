package lightrag.kg.memory

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull
import lightrag.core.types.BaseVectorStorage
import java.io.File

private val logger = KotlinLogging.logger {}

class InMemoryVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
    private val cosineThreshold: Double? = null,
) : BaseVectorStorage {
    override val cosineBetterThanThreshold: Double = cosineThreshold ?: (globalConfig["cosine_better_than_threshold"] as? Double ?: 0.2)
    override val metaFields: Set<String> = emptySet()

    // Using ConcurrentHashMap for thread safety might be better, but MutableMap is fine for simple impl
    private val vectors = mutableMapOf<String, List<Float>>()
    private val metadata = mutableMapOf<String, Metadata>()
    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "vdb_$namespace.json")
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    private fun debug(msg: () -> String) {
        if (logger.isDebugEnabled()) {
            logger.debug(msg)
        }
    }

    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (!file.exists()) return

        try {
            val content = file.readText()
            if (content.isNotBlank()) {
                val jsonElement = json.parseToJsonElement(content)
                if (jsonElement is JsonObject) {
                    val loadedVectors = mutableMapOf<String, List<Float>>()
                    val loadedMetadata = mutableMapOf<String, Metadata>()
                    jsonElement.entries.forEach { (id, value) ->
                        if (value is JsonObject) {
                            val vec = (value["vector"] as? JsonArray)?.mapNotNull { it.toAny() as? Number }?.map { it.toFloat() }
                            val metaObj = value["metadata"]
                            val rawMap = (metaObj?.toAny() as? Map<String, Any>) ?: emptyMap()
                            if (vec != null) {
                                loadedVectors[id] = vec
                            }
                            loadedMetadata[id] = mapToMetadata(rawMap)
                        }
                    }
                    vectors.putAll(loadedVectors)
                    metadata.putAll(loadedMetadata)
                    logger.info { "Loaded ${vectors.size} vectors for '$namespace' from ${file.absolutePath}" }
                }
            }
        } catch (e: Exception) {
            logger.error(e) { "Error loading vector storage from ${file.absolutePath}" }
        }
    }

    override suspend fun indexDoneCallback() {
        try {
            if (!workingDir.exists()) {
                workingDir.mkdirs()
            }
            val jsonObject =
                JsonObject(
                    vectors.mapValues { (id, vec) ->
                        val meta = metadata[id]?.raw ?: emptyMap()
                        JsonObject(
                            mapOf(
                                "vector" to vec.toJsonElement(),
                                "metadata" to meta.toJsonElement(),
                            ),
                        )
                    },
                )
            val content = json.encodeToString(JsonElement.serializer(), jsonObject)
            file.writeText(content)
            debug { "[$namespace/$workspace] Persisted ${vectors.size} vectors to ${file.absolutePath}" }
        } catch (e: Exception) {
            logger.error(e) { "Error saving vector storage to ${file.absolutePath}" }
        }
    }

    override suspend fun drop(): Map<String, String> {
        vectors.clear()
        metadata.clear()
        return mapOf("status" to "success", "message" to "data dropped (file retained at ${file.absolutePath})")
    }

    override suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>?,
    ): List<Map<String, Any>> {
        val queryVec = queryEmbedding ?: embed(query)
        if (queryVec.isEmpty()) {
            logger.warn { "Query vector is empty for query: '$query'" }
            return emptyList()
        }

        if (vectors.isEmpty()) {
            logger.warn { "Vector storage '$namespace' is empty during query." }
            return emptyList()
        }

        debug {
            "[$namespace/$workspace] Query='$query', topK=$topK, vectors=${vectors.size}, metadata=${metadata.size}"
        }

        // Calculate cosine similarity for all vectors
        val results =
            vectors.map { (id, vec) ->
                val similarity =
                    CosineSimilarity.between(
                        Embedding(queryVec.toFloatArray()),
                        Embedding(vec.toFloatArray()),
                    )
                Triple(id, similarity, metadata[id])
            }
                .filter {
                    logger.error { it }
                    it.third != null && it.second >= cosineBetterThanThreshold
                }
                .sortedByDescending { it.second }
                .take(topK)

        if (results.isEmpty()) {
            logger.warn {
                "No results found for query: '$query' in '$namespace'. Vectors count: ${vectors.size}"
            }
        }

        return results.map { (id, score, meta) ->
            val raw = meta?.raw ?: emptyMap()
            raw + mapOf("id" to id, "score" to score, "distance" to score)
        }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        // data keys are IDs, values are metadata maps
        // We expect metadata to contain "content" field which needs to be embedded if not already vectors?
        // In Python LightRAG, upsert logic in vector storage often handles embedding if content is provided.

        val embeddingModel = embeddingFunc
        debug {
            "[$namespace/$workspace] Upsert ${data.size} items. Has embedding model: ${embeddingModel != null}"
        }

        data.forEach { (id, metaMap) ->
            val meta = mapToMetadata(metaMap)
            metadata[id] = meta
            val content = meta.content

            if (content != null && embeddingModel != null) {
                val vec = embed(content)
                if (vec.isNotEmpty()) {
                    vectors[id] = vec
                } else {
                    logger.error { "Error embedding content for id $id" }
                }
            } else if (meta.vector != null) {
                // If vector is provided directly
                vectors[id] = meta.vector
            } else {
                logger.warn { "No content provided for upsert id $id in '$namespace'" }
            }
        }

        debug {
            "[$namespace/$workspace] Upsert completed. Total vectors=${vectors.size}, metadata=${metadata.size}"
        }
    }

    private fun embed(text: String): List<Float> {
        val embeddingModel = embeddingFunc as? EmbeddingModel ?: return emptyList()
        return try {
            val response = embeddingModel.embed(text)
            val content = response.content()
            when (content) {
                is Embedding -> content.vector().toList()
                else -> emptyList()
            }
        } catch (e: Exception) {
            logger.error(e) { "Error embedding text: '$text'" }
            emptyList()
        }
    }

    override suspend fun deleteEntity(entityName: String) {
        // Remove entities where entity_name matches
        val idsToDelete =
            metadata.filter {
                it.value.entityName == entityName
            }.keys
        delete(idsToDelete.toList())
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        // Remove relations where src_id or tgt_id matches
        val idsToDelete =
            metadata.filter {
                it.value.srcId == entityName || it.value.tgtId == entityName
            }.keys
        delete(idsToDelete.toList())
    }

    override suspend fun getById(id: String): Map<String, Any>? {
        return metadata[id]?.raw
    }

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> {
        return ids.mapNotNull { metadata[it]?.raw }
    }

    override suspend fun delete(ids: List<String>) {
        ids.forEach {
            vectors.remove(it)
            metadata.remove(it)
        }
    }

    override suspend fun getVectorsByIds(ids: List<String>): Map<String, List<Float>> {
        return ids.mapNotNull { id ->
            vectors[id]?.let { id to it }
        }.toMap()
    }

    private fun mapToMetadata(meta: Map<String, Any>): Metadata {
        val content = meta["content"] as? String

        @Suppress("UNCHECKED_CAST")
        val vector = meta["vector"] as? List<Float>
        val entityName = meta["entity_name"] as? String
        val srcId = meta["src_id"] as? String
        val tgtId = meta["tgt_id"] as? String
        return Metadata(
            content = content,
            vector = vector,
            entityName = entityName,
            srcId = srcId,
            tgtId = tgtId,
            raw = meta,
        )
    }

    private fun Any?.toJsonElement(): JsonElement {
        return when (this) {
            null -> JsonNull
            is Boolean -> JsonPrimitive(this)
            is Number -> JsonPrimitive(this)
            is String -> JsonPrimitive(this)
            is List<*> -> JsonArray(this.map { it.toJsonElement() })
            is Map<*, *> -> JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })
            else -> JsonPrimitive(this.toString())
        }
    }

    private fun JsonElement.toAny(): Any? {
        return when (this) {
            is JsonNull -> null
            is JsonPrimitive -> {
                if (isString) {
                    content
                } else {
                    booleanOrNull ?: longOrNull ?: doubleOrNull ?: content
                }
            }
            is JsonArray -> this.map { it.toAny() }
            is JsonObject -> this.mapValues { it.value.toAny() }
        }
    }
}
