package lightrag.kg.memory

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.types.BaseVectorStorage
import lightrag.core.types.EmbeddingFunc
import lightrag.kg.memory.Metadata

private val logger = KotlinLogging.logger {}

class InMemoryVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    override val embeddingFunc: EmbeddingFunc = Any(),
) : BaseVectorStorage {
    override val cosineBetterThanThreshold: Double = 0.8
    override val metaFields: Set<String> = emptySet()

    // Using ConcurrentHashMap for thread safety might be better, but MutableMap is fine for simple impl
    private val vectors = mutableMapOf<String, List<Float>>()
    private val metadata = mutableMapOf<String, Metadata>()

    private fun debug(msg: () -> String) {
        if (logger.isDebugEnabled()) {
            logger.debug(msg)
        }
    }

    override suspend fun indexDoneCallback() {
        // No persistence for in-memory
    }

    override suspend fun drop(): Map<String, String> {
        vectors.clear()
        metadata.clear()
        return mapOf("status" to "success", "message" to "data dropped")
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
                .filter { it.third != null }
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

        val embeddingModel = embeddingFunc as? EmbeddingModel
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
                if (embeddingModel == null) {
                    logger.warn { "No embedding model provided for upsert in '$namespace'" }
                } else if (content == null) {
                    logger.warn { "No content provided for upsert id $id in '$namespace'" }
                }
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
}
