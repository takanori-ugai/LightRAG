package lightrag.kg.memory

import dev.langchain4j.model.embedding.EmbeddingModel
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.KnowledgeGraph

class InMemoryGraphStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : BaseGraphStorage {
    private val nodes = mutableMapOf<String, Map<String, String>>()
    private val edges = mutableMapOf<String, MutableMap<String, Map<String, String>>>()

    override suspend fun indexDoneCallback() {}

    override suspend fun drop(): Map<String, String> {
        nodes.clear()
        edges.clear()
        return mapOf("status" to "success", "message" to "data dropped")
    }

    override suspend fun hasNode(nodeId: String): Boolean = nodes.containsKey(nodeId)

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean {
        return edges[sourceNodeId]?.containsKey(targetNodeId) == true
    }

    override suspend fun nodeDegree(nodeId: String): Int {
        return (edges[nodeId]?.size ?: 0) // Simplified directed degree
    }

    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int {
        return nodeDegree(srcId) + nodeDegree(tgtId)
    }

    override suspend fun getNode(nodeId: String): Map<String, String>? = nodes[nodeId]

    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, String>? {
        return edges[sourceNodeId]?.get(targetNodeId)
    }

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>? {
        if (!nodes.containsKey(sourceNodeId)) return null
        return edges[sourceNodeId]?.keys?.map { sourceNodeId to it } ?: emptyList()
    }

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, String>,
    ) {
        nodes[nodeId] = nodeData
    }

    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, String>,
    ) {
        edges.computeIfAbsent(sourceNodeId) { mutableMapOf() }[targetNodeId] = edgeData
        // Undirected graph usually stores both ways or handles it logic wise.
        // BaseGraphStorage says "All operations related to edges in graph should be undirected."
        // So we should add the reverse edge too.
        edges.computeIfAbsent(targetNodeId) { mutableMapOf() }[sourceNodeId] = edgeData
    }

    override suspend fun deleteNode(nodeId: String) {
        nodes.remove(nodeId)
        edges.remove(nodeId)
        edges.values.forEach { it.remove(nodeId) }
    }

    override suspend fun removeNodes(nodes: List<String>) {
        nodes.forEach { deleteNode(it) }
    }

    override suspend fun removeEdges(edgesList: List<Pair<String, String>>) {
        edgesList.forEach { (src, tgt) ->
            edges[src]?.remove(tgt)
            edges[tgt]?.remove(src)
        }
    }

    override suspend fun getAllLabels(): List<String> = nodes.keys.toList().sorted()

    override suspend fun getKnowledgeGraph(
        nodeLabel: String,
        maxDepth: Int,
        maxNodes: Int,
    ): KnowledgeGraph {
        // BFS Implementation placeholder
        return KnowledgeGraph(emptyList(), emptyList())
    }

    override suspend fun getAllNodes(): List<Map<String, Any>> = nodes.values.map { it as Map<String, Any> }

    override suspend fun getAllEdges(): List<Map<String, Any>> {
        val result = mutableListOf<Map<String, Any>>()
        val seen = mutableSetOf<Pair<String, String>>()

        edges.forEach { (src, targets) ->
            targets.forEach { (tgt, data) ->
                val key = if (src < tgt) src to tgt else tgt to src
                if (key !in seen) {
                    seen.add(key)
                    result.add(data + mapOf("source" to src, "target" to tgt))
                }
            }
        }
        return result
    }

    override suspend fun getPopularLabels(limit: Int): List<String> {
        // nodeDegree is suspend function, so we need to map first then sort
        val degrees = nodes.keys.associateWith { nodeDegree(it) }
        return nodes.keys.sortedByDescending { degrees[it] }.take(limit)
    }

    override suspend fun searchLabels(
        query: String,
        limit: Int,
    ): List<String> {
        return nodes.keys.filter { it.contains(query, ignoreCase = true) }.take(limit)
    }
}
