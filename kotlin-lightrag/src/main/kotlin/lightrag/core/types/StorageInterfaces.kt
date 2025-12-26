package lightrag.core.types

import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

interface StorageNameSpace {
    val namespace: String
    val workspace: String
    val globalConfig: Map<String, Any>

    suspend fun initialize() {}

    suspend fun finalize() {}

    suspend fun indexDoneCallback()

    suspend fun drop(): Map<String, String>
}

interface BaseVectorStorage : StorageNameSpace {
    val embeddingFunc: EmbeddingModel
    val cosineBetterThanThreshold: Double
    val metaFields: Set<String>

    suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>? = null,
    ): List<Map<String, Any>>

    suspend fun upsert(data: Map<String, Map<String, Any>>)

    suspend fun deleteEntity(entityName: String)

    suspend fun deleteEntityRelation(entityName: String)

    suspend fun getById(id: String): Map<String, Any>?

    suspend fun getByIds(ids: List<String>): List<Map<String, Any>>

    suspend fun delete(ids: List<String>)

    suspend fun getVectorsByIds(ids: List<String>): Map<String, List<Float>>
}

interface BaseKVStorage : StorageNameSpace {
    val embeddingFunc: EmbeddingModel

    suspend fun getById(id: String): Map<String, Any>?

    suspend fun getByIds(ids: List<String>): List<Map<String, Any>>

    suspend fun filterKeys(keys: Set<String>): Set<String>

    suspend fun upsert(data: Map<String, Map<String, Any>>)

    suspend fun delete(ids: List<String>)

    suspend fun isEmpty(): Boolean
}

interface BaseGraphStorage : StorageNameSpace {
    val embeddingFunc: EmbeddingModel

    suspend fun hasNode(nodeId: String): Boolean

    suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean

    suspend fun nodeDegree(nodeId: String): Int

    suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int

    suspend fun getNode(nodeId: String): Map<String, String>?

    suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, String>?

    suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>?

    suspend fun getNodesBatch(nodeIds: List<String>): Map<String, Map<String, String>> {
        val result = mutableMapOf<String, Map<String, String>>()
        for (nodeId in nodeIds) {
            getNode(nodeId)?.let { result[nodeId] = it }
        }
        return result
    }

    suspend fun nodeDegreesBatch(nodeIds: List<String>): Map<String, Int> {
        val result = mutableMapOf<String, Int>()
        for (nodeId in nodeIds) {
            result[nodeId] = nodeDegree(nodeId)
        }
        return result
    }

    suspend fun edgeDegreesBatch(edgePairs: List<Pair<String, String>>): Map<Pair<String, String>, Int> {
        val result = mutableMapOf<Pair<String, String>, Int>()
        for ((srcId, tgtId) in edgePairs) {
            result[srcId to tgtId] = edgeDegree(srcId, tgtId)
        }
        return result
    }

    suspend fun getEdgesBatch(pairs: List<Map<String, String>>): Map<Pair<String, String>, Map<String, String>> {
        val result = mutableMapOf<Pair<String, String>, Map<String, String>>()
        for (pair in pairs) {
            val srcId = pair["src"] ?: continue
            val tgtId = pair["tgt"] ?: continue
            getEdge(srcId, tgtId)?.let { result[srcId to tgtId] = it }
        }
        return result
    }

    suspend fun getNodesEdgesBatch(nodeIds: List<String>): Map<String, List<Pair<String, String>>> {
        val result = mutableMapOf<String, List<Pair<String, String>>>()
        for (nodeId in nodeIds) {
            getNodeEdges(nodeId)?.let { result[nodeId] = it }
        }
        return result
    }

    suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, String>,
    )

    suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, String>,
    )

    suspend fun deleteNode(nodeId: String)

    suspend fun removeNodes(nodes: List<String>)

    suspend fun removeEdges(edges: List<Pair<String, String>>)

    suspend fun getAllLabels(): List<String>

    suspend fun getKnowledgeGraph(
        nodeLabel: String,
        maxDepth: Int = 3,
        maxNodes: Int = 1000,
    ): KnowledgeGraph

    suspend fun getAllNodes(): List<Map<String, Any>>

    suspend fun getAllEdges(): List<Map<String, Any>>

    suspend fun getPopularLabels(limit: Int = 300): List<String>

    suspend fun searchLabels(
        query: String,
        limit: Int = 50,
    ): List<String>
}

@Serializable
data class KnowledgeGraph(
    val nodes: List<Map<String, @Contextual Any>>,
    val edges: List<Map<String, @Contextual Any>>,
    val isTruncated: Boolean = false,
)

enum class DocStatus(val value: String) {
    PENDING("pending"),
    PROCESSING("processing"),
    PREPROCESSED("preprocessed"),
    PROCESSED("processed"),
    FAILED("failed"),
}

@Serializable
data class DocProcessingStatus(
    @SerialName("content_summary")
    val contentSummary: String,
    @SerialName("content_length")
    val contentLength: Int,
    @SerialName("file_path")
    val filePath: String,
    val status: DocStatus,
    @SerialName("created_at")
    val createdAt: String,
    @SerialName("updated_at")
    val updatedAt: String,
    @SerialName("track_id")
    val trackId: String? = null,
    @SerialName("chunks_count")
    val chunksCount: Int? = null,
    @SerialName("chunks_list")
    val chunksList: List<String>? = null,
    @SerialName("error_msg")
    val errorMsg: String? = null,
    // Simplified to String map for now
    val metadata: Map<String, String> = emptyMap(),
    @SerialName("multimodal_processed")
    val multimodalProcessed: Boolean? = null,
)

interface DocStatusStorage : BaseKVStorage {
    suspend fun getStatusCounts(): Map<String, Int>

    suspend fun getDocsByStatus(status: DocStatus): Map<String, DocProcessingStatus>

    suspend fun getDocsByTrackId(trackId: String): Map<String, DocProcessingStatus>

    suspend fun getDocsPaginated(
        statusFilter: DocStatus? = null,
        page: Int = 1,
        pageSize: Int = 50,
        sortField: String = "updated_at",
        sortDirection: String = "desc",
    ): Pair<List<Pair<String, DocProcessingStatus>>, Int>

    suspend fun getAllStatusCounts(): Map<String, Int>

    suspend fun getDocByFilePath(filePath: String): Map<String, Any>?
}
