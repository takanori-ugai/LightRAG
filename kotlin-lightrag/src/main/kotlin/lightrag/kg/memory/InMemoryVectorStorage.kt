package lightrag.kg.memory

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.SerializationException
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import lightrag.core.types.BaseVectorStorage
import java.io.File
import java.io.IOException

private val logger = KotlinLogging.logger {}

/**
 * An in-memory vector storage implementation.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 * @property cosineThreshold The threshold for cosine similarity.
 */
class InMemoryVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
    private val cosineThreshold: Double? = null,
) : BaseVectorStorage {
    /**
     * The threshold for cosine similarity.
     */
    override val cosineBetterThanThreshold: Double = cosineThreshold ?: (globalConfig["cosine_better_than_threshold"] as? Double ?: 0.2)

    /**
     * The set of meta fields.
     */
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

    /**
     * Initializes the storage by loading data from the JSON file.
     */
    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (!file.exists()) return

        try {
            val content = file.readText()
            if (content.isNotBlank()) {
                val loadedData = json.decodeFromString<Map<String, JsonObject>>(content)
                loadedData.forEach { (id, jsonObject) ->
                    val vec = (jsonObject["vector"] as? JsonArray)?.mapNotNull { it.jsonPrimitive.doubleOrNull?.toFloat() }
                    val metaObj = jsonObject["metadata"] as? JsonObject
                    val rawMap =
                        (metaObj?.toAny() as? Map<*, *>)?.entries?.associate { (k, v) ->
                            k.toString() to (v as Any)
                        } ?: emptyMap()
                    if (vec != null) {
                        vectors[id] = vec
                    }
                    metadata[id] = mapToMetadata(rawMap)
                }
                logger.info { "Loaded ${vectors.size} vectors for '$namespace' from ${file.absolutePath}" }
            }
        } catch (e: IOException) {
            logger.error(e) { "I/O error loading vector storage from ${file.absolutePath}" }
        } catch (e: SerializationException) {
            logger.error(e) { "Serialization error loading vector storage from ${file.absolutePath}" }
        }
    }

    /**
     * Saves the current state of the storage to the JSON file.
     */
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
                                "vector" to JsonArray(vec.map { JsonPrimitive(it) }),
                                "metadata" to meta.toJsonElement(),
                            ),
                        )
                    },
                )
            val content = json.encodeToString(jsonObject)
            file.writeText(content)
            if (logger.isDebugEnabled()) {
                logger.debug { "[$namespace/$workspace] Persisted ${vectors.size} vectors to ${file.absolutePath}" }
            }
        } catch (e: IOException) {
            logger.error(e) { "I/O error saving vector storage to ${file.absolutePath}" }
        } catch (e: SerializationException) {
            logger.error(e) { "Serialization error saving vector storage to ${file.absolutePath}" }
        }
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        vectors.clear()
        metadata.clear()
        if (file.exists()) {
            runCatching { file.delete() }.onFailure { logger.warn(it) { "Failed to delete vector store file ${file.absolutePath}" } }
        }
        return mapOf("status" to "success", "message" to "data dropped and file removed at ${file.absolutePath}")
    }

    /**
     * Queries the vector storage.
     * @param query The query string.
     * @param topK The number of top results to return.
     * @param queryEmbedding The query embedding.
     * @return A list of maps representing the results.
     */
    override suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>?,
    ): List<Map<String, Any>> {
        val queryVec = queryEmbedding ?: embedText(embeddingFunc, query, logger)
        if (queryVec.isEmpty()) {
            logger.warn { "Query vector is empty for query: '$query'" }
            return emptyList()
        }

        if (vectors.isEmpty()) {
            logger.warn { "Vector storage '$namespace' is empty during query." }
            return emptyList()
        }

        if (logger.isDebugEnabled()) {
            logger.debug {
                "[$namespace/$workspace] Query='$query', topK=$topK, vectors=${vectors.size}, metadata=${metadata.size}"
            }
        }

        // Calculate cosine similarity for all vectors
        val results =
            vectors
                .mapNotNull { (id, vec) ->
                    val meta = metadata[id]
                    if (meta == null) {
                        logger.warn { "Skipping vector '$id' in '$namespace' due to missing metadata." }
                        return@mapNotNull null
                    }
                    if (vec.isEmpty()) {
                        logger.warn { "Skipping vector '$id' in '$namespace' because it is empty." }
                        return@mapNotNull null
                    }
                    if (vec.size != queryVec.size) {
                        logger.warn {
                            "Skipping vector '$id' in '$namespace' due to dimension mismatch: stored=${vec.size}, query=${queryVec.size}"
                        }
                        return@mapNotNull null
                    }
                    val similarity =
                        CosineSimilarity.between(
                            Embedding(queryVec.toFloatArray()),
                            Embedding(vec.toFloatArray()),
                        )
                    Triple(id, similarity, meta)
                }.filter {
                    it.second >= cosineBetterThanThreshold
                }.sortedByDescending { it.second }
                .take(topK)

        if (results.isEmpty()) {
            logger.warn {
                "No results found for query: '$query' in '$namespace'. Vectors count: ${vectors.size}"
            }
        }

        return results.map { (id, score, meta) ->
            val raw = meta.raw
            raw + mapOf("id" to id, "score" to score, "distance" to score)
        }
    }

    /**
     * Upserts data into the vector storage.
     * @param data The data to upsert.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        // data keys are IDs, values are metadata maps
        // We expect metadata to contain "content" field which needs to be embedded if not already vectors?
        // In Python LightRAG, upsert logic in vector storage often handles embedding if content is provided.

        if (logger.isDebugEnabled()) {
            logger.debug { "[$namespace/$workspace] Upsert ${data.size} items. Has embedding model: true" }
        }

        data.forEach { (id, metaMap) ->
            val meta = mapToMetadata(metaMap)
            metadata[id] = meta
            val content = meta.content

            if (content != null) {
                val vec = embedText(embeddingFunc, content, logger)
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

        if (logger.isDebugEnabled()) {
            logger.debug { "[$namespace/$workspace] Upsert completed. Total vectors=${vectors.size}, metadata=${metadata.size}" }
        }
    }

    /**
     * Deletes an entity from the vector storage.
     * @param entityName The name of the entity to delete.
     */
    override suspend fun deleteEntity(entityName: String) {
        // Remove entities where entity_name matches
        val idsToDelete =
            metadata
                .filter {
                    it.value.entityName == entityName
                }.keys
        delete(idsToDelete.toList())
    }

    /**
     * Deletes an entity relation from the vector storage.
     * @param entityName The name of the entity relation to delete.
     */
    override suspend fun deleteEntityRelation(entityName: String) {
        // Remove relations where src_id or tgt_id matches
        val idsToDelete =
            metadata
                .filter {
                    it.value.srcId == entityName || it.value.tgtId == entityName
                }.keys
        delete(idsToDelete.toList())
    }

    /**
     * Gets an item by its ID.
     * @param id The ID of the item to get.
     * @return A map representing the item.
     */
    override suspend fun getById(id: String): Map<String, Any>? = metadata[id]?.raw

    /**
     * Gets items by their IDs.
     * @param ids The IDs of the items to get.
     * @return A list of maps representing the items.
     */
    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> = ids.mapNotNull { metadata[it]?.raw }

    /**
     * Deletes items by their IDs.
     * @param ids The IDs of the items to delete.
     */
    override suspend fun delete(ids: List<String>) {
        ids.forEach {
            vectors.remove(it)
            metadata.remove(it)
        }
    }

    /**
     * Gets vectors by their IDs.
     * @param ids The IDs of the vectors to get.
     * @return A map of IDs to vectors.
     */
    override suspend fun getVectorsByIds(ids: List<String>): Map<String, List<Float>> =
        ids
            .mapNotNull { id ->
                vectors[id]?.let { id to it }
            }.toMap()
}

private fun embedText(
    embeddingModel: EmbeddingModel,
    text: String,
    logger: KLogger,
): List<Float> =
    try {
        val response = embeddingModel.embed(text)
        response.content().vector().toList()
    } catch (e: IllegalStateException) {
        logger.error(e) { "Illegal state embedding text: '$text'" }
        emptyList()
    } catch (e: IllegalArgumentException) {
        logger.error(e) { "Invalid input embedding text: '$text'" }
        emptyList()
    }

private fun mapToMetadata(meta: Map<String, Any>): Metadata {
    val content = meta["content"] as? String

    @Suppress("UNCHECKED_CAST")
    val vector = (meta["vector"] as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() }
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

private fun Any?.toJsonElement(): JsonElement =
    when (this) {
        null -> {
            JsonNull
        }

        is Boolean -> {
            JsonPrimitive(this)
        }

        is Number -> {
            JsonPrimitive(this.toString())
        }

        is String -> {
            JsonPrimitive(this)
        }

        is List<*> -> {
            JsonArray(this.map { it.toJsonElement() })
        }

        is Map<*, *> -> {
            JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })
        }

        else -> {
            JsonPrimitive(this.toString())
        }
    }

private fun JsonElement.toAny(): Any? =
    when (this) {
        is JsonNull -> {
            null
        }

        is JsonPrimitive -> {
            if (isString) {
                content
            } else {
                booleanOrNull ?: longOrNull ?: doubleOrNull ?: content
            }
        }

        is JsonArray -> {
            this.map { it.toAny() }
        }

        is JsonObject -> {
            this.mapValues { it.value.toAny() }
        }
    }
