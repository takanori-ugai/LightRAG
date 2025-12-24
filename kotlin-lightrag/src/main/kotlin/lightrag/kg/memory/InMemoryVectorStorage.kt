package lightrag.kg.memory

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import lightrag.core.types.BaseVectorStorage
import lightrag.core.types.EmbeddingFunc

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
    private val metadata = mutableMapOf<String, Map<String, Any>>()

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
        if (queryVec.isEmpty()) return emptyList()

        // Calculate cosine similarity for all vectors
        val results =
            vectors.map { (id, vec) ->
                val similarity = CosineSimilarity.between(Embedding(queryVec.toFloatArray()), Embedding(vec.toFloatArray()))
                Triple(id, similarity, metadata[id])
            }
                .filter { it.third != null }
                .sortedByDescending { it.second }
                .take(topK)

        return results.map { (id, score, meta) ->
            (meta ?: emptyMap()) + mapOf("id" to id, "score" to score, "distance" to score)
        }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        // data keys are IDs, values are metadata maps
        // We expect metadata to contain "content" field which needs to be embedded if not already vectors?
        // In Python LightRAG, upsert logic in vector storage often handles embedding if content is provided.

        val embeddingModel = embeddingFunc as? EmbeddingModel

        data.forEach { (id, meta) ->
            metadata[id] = meta
            val content = meta["content"] as? String

            if (content != null && embeddingModel != null) {
                // Generate embedding
                try {
                    val embedding = embeddingModel.embed(TextSegment.from(content)).content()
                    vectors[id] = embedding.vector().toList()
                } catch (e: Exception) {
                    println("Error embedding content for id $id: ${e.message}")
                }
            } else if (meta.containsKey("vector")) {
                // If vector is provided directly (not typical for this codebase based on usage, but good for completeness)
                @Suppress("UNCHECKED_CAST")
                val vec = meta["vector"] as? List<Float>
                if (vec != null) {
                    vectors[id] = vec
                }
            }
        }
    }

    private fun embed(text: String): List<Float> {
        val embeddingModel = embeddingFunc as? EmbeddingModel ?: return emptyList()
        return try {
            embeddingModel.embed(text).content().vector().toList()
        } catch (e: Exception) {
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
}
