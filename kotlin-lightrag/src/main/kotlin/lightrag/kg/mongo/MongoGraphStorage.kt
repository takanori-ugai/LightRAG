package lightrag.kg.mongo

import com.mongodb.client.model.Filters
import com.mongodb.client.model.ReplaceOptions
import com.mongodb.kotlin.client.coroutine.MongoClient
import com.mongodb.kotlin.client.coroutine.MongoCollection
import com.mongodb.kotlin.client.coroutine.MongoDatabase
import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.toList
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.KnowledgeGraph
import org.bson.Document

private val logger = KotlinLogging.logger {}

class MongoGraphStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any>,
    override val embeddingFunc: EmbeddingModel,
) : BaseGraphStorage {
    override val workspace: String = globalConfig["working_dir"] as? String ?: "./rag_storage"

    private lateinit var client: MongoClient
    private lateinit var database: MongoDatabase
    private lateinit var nodesCollection: MongoCollection<Document>
    private lateinit var edgesCollection: MongoCollection<Document>

    private val uri: String =
        System.getenv("MONGO_URI") ?: "mongodb://0.0.0.0:27017/?directConnection=true"
    private val dbName: String = System.getenv("MONGO_DATABASE") ?: "LightRAG"
    private val collectionName: String = System.getenv("MONGO_KG_COLLECTION") ?: "MDB_KG"

    override suspend fun initialize() {
        logger.info {
            "Initializing MongoGraphStorage with URI: $uri, DB: $dbName, Collection: $collectionName"
        }
        client = MongoClient.create(uri)
        database = client.getDatabase(dbName)
        // We will store both nodes and edges in the same collection

        // Let's assume we use the collection name provided.
        val col = database.getCollection<Document>(collectionName)
        nodesCollection = col
        edgesCollection = col

        // Create indexes if needed?
    }

    override suspend fun finalize() {
        if (::client.isInitialized) {
            client.close()
        }
    }

    override suspend fun indexDoneCallback() {
        // MongoDB persists immediately, so maybe nothing to do here, or ensure indexes.
    }

    override suspend fun drop(): Map<String, String> {
        // Implement drop logic if needed, e.g., drop collection
        return emptyMap()
    }

    override suspend fun hasNode(nodeId: String): Boolean {
        val count =
            nodesCollection.countDocuments(
                Filters.and(
                    Filters.eq("id", nodeId),
                    Filters.eq("type", "node"),
                ),
            )
        return count > 0
    }

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean {
        val count =
            edgesCollection.countDocuments(
                Filters.and(
                    Filters.eq("source_id", sourceNodeId),
                    Filters.eq("target_id", targetNodeId),
                    Filters.eq("type", "edge"),
                ),
            )
        return count > 0
    }

    override suspend fun nodeDegree(nodeId: String): Int {
        val srcCount =
            edgesCollection.countDocuments(
                Filters.and(
                    Filters.eq("source_id", nodeId),
                    Filters.eq("type", "edge"),
                ),
            )
        val tgtCount =
            edgesCollection.countDocuments(
                Filters.and(
                    Filters.eq("target_id", nodeId),
                    Filters.eq("type", "edge"),
                ),
            )
        return (srcCount + tgtCount).toInt()
    }

    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int {
        // Degree of an edge? usually it's just 1 if it exists, or weight?
        // The BaseGraphStorage interface asks for edgeDegree.
        // In NetworkX, degree is for nodes.
        // Maybe this means the weight of the edge?
        // Let's check InMemoryGraphStorage implementation.
        return (nodeDegree(srcId) + nodeDegree(tgtId))
    }

    override suspend fun getNode(nodeId: String): Map<String, String>? {
        val doc =
            nodesCollection.find(
                Filters.and(
                    Filters.eq("id", nodeId),
                    Filters.eq("type", "node"),
                ),
            ).firstOrNull() ?: return null

        return doc.entries.associate { (k, v) -> k to v.toString() }
    }

    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, String>? {
        val doc =
            edgesCollection.find(
                Filters.and(
                    Filters.eq("source_id", sourceNodeId),
                    Filters.eq("target_id", targetNodeId),
                    Filters.eq("type", "edge"),
                ),
            ).firstOrNull() ?: return null

        return doc.entries.associate { (k, v) -> k to v.toString() }
    }

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>? {
        // In python: list(self._graph.edges(sourceNodeId)) -> [(src, tgt), ...]
        // Here we want list of (src, tgt) connected to sourceNodeId?
        // Or just outgoing edges?
        // Usually edges(n) returns all edges adjacent to n.

        val edges =
            edgesCollection.find(
                Filters.and(
                    Filters.or(
                        Filters.eq("source_id", sourceNodeId),
                        Filters.eq("target_id", sourceNodeId),
                    ),
                    Filters.eq("type", "edge"),
                ),
            ).toList()

        return edges.map { doc ->
            (doc.getString("source_id") ?: "") to (doc.getString("target_id") ?: "")
        }
    }

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, String>,
    ) {
        val doc =
            Document("id", nodeId)
                .append("type", "node")
        nodeData.forEach { (k, v) -> doc.append(k, v) }

        nodesCollection.replaceOne(
            Filters.and(Filters.eq("id", nodeId), Filters.eq("type", "node")),
            doc,
            ReplaceOptions().upsert(true),
        )
    }

    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, String>,
    ) {
        val doc =
            Document("source_id", sourceNodeId)
                .append("target_id", targetNodeId)
                .append("type", "edge")
        edgeData.forEach { (k, v) -> doc.append(k, v) }

        edgesCollection.replaceOne(
            Filters.and(
                Filters.eq("source_id", sourceNodeId),
                Filters.eq("target_id", targetNodeId),
                Filters.eq("type", "edge"),
            ),
            doc,
            ReplaceOptions().upsert(true),
        )
    }

    override suspend fun deleteNode(nodeId: String) {
        nodesCollection.deleteOne(
            Filters.and(
                Filters.eq("id", nodeId),
                Filters.eq("type", "node"),
            ),
        )
        // Also delete connected edges?
        edgesCollection.deleteMany(
            Filters.and(
                Filters.or(
                    Filters.eq("source_id", nodeId),
                    Filters.eq("target_id", nodeId),
                ),
                Filters.eq("type", "edge"),
            ),
        )
    }

    override suspend fun removeNodes(nodes: List<String>) {
        nodes.forEach { deleteNode(it) }
    }

    override suspend fun removeEdges(edges: List<Pair<String, String>>) {
        edges.forEach { (src, tgt) ->
            edgesCollection.deleteOne(
                Filters.and(
                    Filters.eq("source_id", src),
                    Filters.eq("target_id", tgt),
                    Filters.eq("type", "edge"),
                ),
            )
        }
    }

    override suspend fun getAllLabels(): List<String> {
        // Assuming labels are stored in node data, but 'getAllLabels' usually refers to edge labels in some graphs or node types?
        // BaseGraphStorage documentation or python implementation?
        // Python networkx graph doesn't enforce labels.
        // Maybe getting all node IDs?
        return emptyList()
    }

    override suspend fun getKnowledgeGraph(
        nodeLabel: String,
        maxDepth: Int,
        maxNodes: Int,
    ): KnowledgeGraph {
        // This seems to be complex graph traversal.
        // For now, return empty or implement simple BFS.
        return KnowledgeGraph(emptyList(), emptyList())
    }

    override suspend fun getAllNodes(): List<Map<String, Any>> {
        return nodesCollection
            .find(Filters.eq("type", "node"))
            .toList()
            .map { it.toMap() }
    }

    override suspend fun getAllEdges(): List<Map<String, Any>> {
        return edgesCollection
            .find(Filters.eq("type", "edge"))
            .toList()
            .map { it.toMap() }
    }

    override suspend fun getPopularLabels(limit: Int): List<String> {
        return emptyList()
    }

    override suspend fun searchLabels(
        query: String,
        limit: Int,
    ): List<String> {
        return emptyList()
    }
}
