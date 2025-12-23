package lightrag.kg.memory

import lightrag.core.types.BaseVectorStorage
import lightrag.core.types.EmbeddingFunc

class InMemoryVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    // Placeholder
    override val embeddingFunc: EmbeddingFunc = Any(),
) : BaseVectorStorage {
    override val cosineBetterThanThreshold: Double = 0.8
    override val metaFields: Set<String> = emptySet()

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
        // Implement simple cosine similarity search here
        return emptyList()
    }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        // Store vectors and metadata
    }

    override suspend fun deleteEntity(entityName: String) {
        // Remove entities
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        // Remove relations
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
