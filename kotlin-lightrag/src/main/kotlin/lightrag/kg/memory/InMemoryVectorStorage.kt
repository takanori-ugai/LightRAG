package lightrag.kg.memory

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
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
import lightrag.core.types.EmbeddingFunc
import java.io.File

private val logger = KotlinLogging.logger {}

class InMemoryVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    override val embeddingFunc: EmbeddingFunc = Any(),
) : BaseVectorStorage {
    override val cosineBetterThanThreshold: Double = 0.8
    override val metaFields: Set<String> = emptySet()

    private val vectors = mutableMapOf<String, List<Float>>()
    private val metadata = mutableMapOf<String, Map<String, Any>>()
    private val mutex = Mutex()

    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "vdb_$namespace.json")

    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    override suspend fun initialize() {
        mutex.withLock {
            if (!workingDir.exists()) {
                workingDir.mkdirs()
            }
            if (file.exists()) {
                try {
                    val content = file.readText()
                    if (content.isNotBlank()) {
                        val jsonElement = json.parseToJsonElement(content)
                        if (jsonElement is JsonObject) {
                            val data =
                                jsonElement.entries.associate { (k, v) ->
                                    val obj = v as? JsonObject ?: JsonObject(emptyMap())
                                    val vec =
                                        (obj["vector"] as? JsonArray)?.map {
                                            (it as? JsonPrimitive)?.content?.toFloatOrNull() ?: 0.0f
                                        } ?: emptyList()

                                    @Suppress("UNCHECKED_CAST")
                                    val meta =
                                        (obj["metadata"] as? JsonObject)?.toAny() as? Map<String, Any>
                                            ?: emptyMap()
                                    k to Pair(vec, meta)
                                }

                            data.forEach { (id, pair) ->
                                vectors[id] = pair.first
                                metadata[id] = pair.second
                            }
                            logger.info { "Loaded ${data.size} vectors from ${file.absolutePath}" }
                        }
                    }
                } catch (e: Exception) {
                    logger.error(e) { "Error loading Vector storage from ${file.absolutePath}" }
                }
            }
        }
    }

    override suspend fun indexDoneCallback() {
        mutex.withLock {
            try {
                val exportMap =
                    vectors.keys.associateWith { id ->
                        mapOf(
                            "vector" to (vectors[id] ?: emptyList()),
                            "metadata" to (metadata[id] ?: emptyMap()),
                        ).toJsonElement()
                    }
                val jsonObject = JsonObject(exportMap.mapValues { it.value.toJsonElement() as JsonObject })
                val content = json.encodeToString(JsonElement.serializer(), jsonObject)
                file.writeText(content)
                logger.debug { "Saved ${vectors.size} vectors to ${file.absolutePath}" }
            } catch (e: Exception) {
                logger.error(e) { "Error saving Vector storage to ${file.absolutePath}" }
            }
        }
    }

    override suspend fun drop(): Map<String, String> {
        mutex.withLock {
            vectors.clear()
            metadata.clear()
            if (file.exists()) {
                file.delete()
            }
        }
        return mapOf("status" to "success", "message" to "data dropped")
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
                    // Workaround for potential double-quoting issue
                    if (content.length >= 2 && content.startsWith("\"") && content.endsWith("\"")) {
                        content.substring(1, content.length - 1)
                    } else {
                        content
                    }
                } else {
                    booleanOrNull ?: longOrNull ?: doubleOrNull ?: content
                }
            }
            is JsonArray -> this.map { it.toAny() }
            is JsonObject -> this.mapValues { it.value.toAny() }
        }
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
                .filter { it.third != null }
                .sortedByDescending { it.second }
                .take(topK)

        if (results.isEmpty()) {
            logger.warn {
                "No results found for query: '$query' in '$namespace'. Vectors count: ${vectors.size}"
            }
        }

        return results.map { (id, score, meta) ->
            (meta ?: emptyMap()) + mapOf("id" to id, "score" to score, "distance" to score)
        }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        // data keys are IDs, values are metadata maps
        // We expect metadata to contain "content" field which needs to be embedded if not already vectors?
        // In Python LightRAG, upsert logic in vector storage often handles embedding if content is provided.

        val embeddingModel = embeddingFunc as? EmbeddingModel
        logger.debug { "Upserting ${data.size} items to '$namespace'. Embedding model present: ${embeddingModel != null}" }

        data.forEach { (id, meta) ->
            metadata[id] = meta
            val content = meta["content"] as? String

            if (content != null && embeddingModel != null) {
                // Generate embedding
                try {
                    val embedding = embeddingModel.embed(TextSegment.from(content)).content()
                    vectors[id] = embedding.vector().toList()
                } catch (e: Exception) {
                    logger.error(e) { "Error embedding content for id $id" }
                }
            } else if (meta.containsKey("vector")) {
                // If vector is provided directly
                @Suppress("UNCHECKED_CAST")
                val vec = meta["vector"] as? List<Float>
                if (vec != null) {
                    vectors[id] = vec
                }
            } else {
                if (embeddingModel == null) {
                    logger.warn { "No embedding model provided for upsert in '$namespace'" }
                } else if (content == null) {
                    logger.warn { "No content provided for upsert id $id in '$namespace'" }
                }
            }
        }
    }

    private fun embed(text: String): List<Float> {
        val embeddingModel = embeddingFunc as? EmbeddingModel ?: return emptyList()
        return try {
            embeddingModel.embed(text).content().vector().toList()
        } catch (e: Exception) {
            logger.error(e) { "Error embedding text: '$text'" }
            emptyList()
        }
    }

    override suspend fun deleteEntity(entityName: String) {
        // Remove entities where entity_name matches
        val idsToDelete =
            metadata.filter {
                it.value["entity_name"] == entityName
            }.keys
        delete(idsToDelete.toList())
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        // Remove relations where src_id or tgt_id matches
        val idsToDelete =
            metadata.filter {
                it.value["src_id"] == entityName || it.value["tgt_id"] == entityName
            }.keys
        delete(idsToDelete.toList())
    }

    override suspend fun getById(id: String): Map<String, Any>? {
        return metadata[id]
    }

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> {
        return ids.mapNotNull { metadata[it] }
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

    override suspend fun isEmpty(): Boolean {
        mutex.withLock {
            return vectors.isEmpty()
        }
    }
}
