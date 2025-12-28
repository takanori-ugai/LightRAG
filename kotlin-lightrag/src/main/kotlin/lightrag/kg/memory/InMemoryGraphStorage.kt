package lightrag.kg.memory

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.KnowledgeGraph
import java.io.File

private val logger = KotlinLogging.logger {}

class InMemoryGraphStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : BaseGraphStorage {
    private val nodes = mutableMapOf<String, Map<String, String>>()
    private val edges = mutableMapOf<String, MutableMap<String, Map<String, String>>>()
    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "graph_$namespace.json")
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    @Suppress("UNCHECKED_CAST")
    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (!file.exists()) return
        runCatching {
            val content = file.readText()
            if (content.isBlank()) return@runCatching
            val parsed = json.parseToJsonElement(content)
            if (parsed is JsonObject) {
                val loadedNodes =
                    (parsed["nodes"] as? JsonObject)?.entries?.associate { (k, v) ->
                        k to ((v.toAny() as? Map<String, String>) ?: emptyMap())
                    } ?: emptyMap()
                val loadedEdges =
                    (parsed["edges"] as? JsonObject)?.entries?.associate { (src, tgtObj) ->
                        val tgtMap =
                            (tgtObj as? JsonObject)?.entries?.associate { (tgt, data) ->
                                tgt to ((data.toAny() as? Map<String, String>) ?: emptyMap())
                            }?.toMutableMap() ?: mutableMapOf()
                        src to tgtMap
                    } ?: emptyMap()
                nodes.putAll(loadedNodes)
                edges.putAll(loadedEdges)
                logger.info {
                    "Loaded graph '$namespace' with ${nodes.size} nodes and ${edges.size} edge buckets from ${file.absolutePath}"
                }
            }
        }.onFailure { logger.error(it) { "Error loading graph storage from ${file.absolutePath}" } }
    }

    override suspend fun indexDoneCallback() {
        runCatching {
            if (!workingDir.exists()) {
                workingDir.mkdirs()
            }
            val nodesObj = JsonObject(nodes.mapValues { (_, v) -> v.toJsonElement() as JsonObject })
            val edgesObj =
                JsonObject(
                    edges.mapValues { (_, targets) ->
                        JsonObject(targets.mapValues { (_, data) -> data.toJsonElement() as JsonObject })
                    },
                )
            val payload =
                JsonObject(
                    mapOf(
                        "nodes" to nodesObj,
                        "edges" to edgesObj,
                    ),
                )
            val content = json.encodeToString(JsonElement.serializer(), payload)
            file.writeText(content)
            logger.debug { "Persisted graph '$namespace' with ${nodes.size} nodes to ${file.absolutePath}" }
        }.onFailure { logger.error(it) { "Error saving graph storage to ${file.absolutePath}" } }
    }

    override suspend fun drop(): Map<String, String> {
        nodes.clear()
        edges.clear()
        if (file.exists()) {
            runCatching { file.delete() }.onFailure {
                logger.warn(it) { "Failed to delete graph file ${file.absolutePath}" }
            }
        }
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

    override suspend fun removeEdges(edges: List<Pair<String, String>>) {
        edges.forEach { (src, tgt) ->
            this.edges[src]?.remove(tgt)
            this.edges[tgt]?.remove(src)
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

    private fun Any?.toJsonElement(): JsonElement {
        return when (this) {
            null -> JsonNull
            is Boolean -> JsonPrimitive(this)
            is Number -> JsonPrimitive(this)
            is String -> JsonPrimitive(this)
            is List<*> -> kotlinx.serialization.json.JsonArray(this.map { it.toJsonElement() })
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
            is kotlinx.serialization.json.JsonArray -> this.map { it.toAny() }
            is JsonObject -> this.mapValues { it.value.toAny() }
        }
    }
}
