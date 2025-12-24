package lightrag.kg.neo4j

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.EmbeddingFunc
import lightrag.core.types.KnowledgeGraph
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Session
import org.neo4j.driver.SessionConfig
import org.neo4j.driver.exceptions.ClientException
import org.neo4j.driver.exceptions.Neo4jException
import org.neo4j.driver.exceptions.ServiceUnavailableException
import org.neo4j.driver.types.Node
import java.util.concurrent.TimeUnit
import kotlin.math.min

private val logger = KotlinLogging.logger {}

class Neo4jGraphStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any>,
    override val embeddingFunc: EmbeddingFunc?,
) : BaseGraphStorage {
    override val workspace: String

    private var driver: Driver? = null
    private var database: String?

    companion object {
        private const val DEFAULT_POOL_SIZE = 100
        private const val DEFAULT_TIMEOUT_MS = 30000L
        private const val DEFAULT_MAX_LIFETIME_MS = 300000L
        private const val SCORE_EXACT_MATCH = 1000
        private const val SCORE_STARTS_WITH = 500
        private const val SCORE_CONTAINS = 500
        private const val SCORE_PARTIAL = 50
        private const val DEFAULT_MAX_NODES = 1000
        private const val FALLBACK_SCORE_BASE = 100
    }

    init {
        // Read env and override the arg if present
        val neo4jWorkspace = System.getenv("NEO4J_WORKSPACE")
        val originalWorkspace = globalConfig["workspace"] as? String
        var ws = originalWorkspace
        if (!neo4jWorkspace.isNullOrBlank()) {
            ws = neo4jWorkspace
        }

        // Default to 'base' when both arg and env are empty
        if (ws.isNullOrBlank()) {
            ws = "base"
        }
        workspace = ws ?: "base"

        if (!neo4jWorkspace.isNullOrBlank()) {
            logger.info {
                "Using NEO4J_WORKSPACE environment variable: '$neo4jWorkspace' " +
                    "(overriding '$originalWorkspace/$namespace')"
            }
        }

        database = System.getenv("NEO4J_DATABASE")
    }

    private fun getWorkspaceLabel(): String {
        return workspace
    }

    private fun normalizeIndexSuffix(workspaceLabel: String): String {
        var normalized = workspaceLabel.replace(Regex("[^A-Za-z0-9_]+"), "_").trim('_')
        if (normalized.isEmpty()) {
            normalized = "base"
        }
        if (!normalized[0].isLetter() && normalized[0] != '_') {
            normalized = "ws_$normalized"
        }
        return normalized
    }

    private fun getFulltextIndexName(workspaceLabel: String): String {
        val suffix = normalizeIndexSuffix(workspaceLabel)
        return "entity_id_fulltext_idx_$suffix"
    }

    private fun isChineseText(text: String): Boolean {
        val cjkPattern = Regex("[\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]|[\uD840-\uD87F][\uDC00-\uDFFF]")
        return cjkPattern.containsMatchIn(text)
    }

    private fun getSessionConfig(): SessionConfig {
        return if (database != null) {
            SessionConfig.forDatabase(database)
        } else {
            SessionConfig.defaultConfig()
        }
    }

    override suspend fun initialize() {
        val uri = System.getenv("NEO4J_URI") ?: (globalConfig["neo4j"] as? Map<*, *>)?.get("uri") as? String
        val username =
            System.getenv("NEO4J_USERNAME") ?: (globalConfig["neo4j"] as? Map<*, *>)?.get("username") as? String
        val password =
            System.getenv("NEO4J_PASSWORD") ?: (globalConfig["neo4j"] as? Map<*, *>)?.get("password") as? String

        if (uri == null) {
            logger.error { "NEO4J_URI is not set" }
            return
        }

        val authToken =
            if (username != null && password != null) {
                AuthTokens.basic(username, password)
            } else {
                AuthTokens.none()
            }

        // Configuration mapping
        val maxConnectionPoolSize =
            (System.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE")?.toIntOrNull() ?: DEFAULT_POOL_SIZE)
        val connectionTimeout =
            (System.getenv("NEO4J_CONNECTION_TIMEOUT")?.toLongOrNull() ?: DEFAULT_TIMEOUT_MS) // ms
        val maxConnectionLifetime =
            (System.getenv("NEO4J_MAX_CONNECTION_LIFETIME")?.toLongOrNull() ?: DEFAULT_MAX_LIFETIME_MS) // ms

        val config =
            org.neo4j.driver.Config.builder()
                .withMaxConnectionPoolSize(maxConnectionPoolSize)
                .withConnectionTimeout(connectionTimeout, TimeUnit.MILLISECONDS)
                .withMaxConnectionLifetime(maxConnectionLifetime, TimeUnit.MILLISECONDS)
                .build()

        driver = GraphDatabase.driver(uri, authToken, config)

        withContext(Dispatchers.IO) {
            var connected = false
            // Try to connect to the database and create it if it doesn't exist
            val databasesToCheck = listOf(database, null) // null means default database

            for (db in databasesToCheck) {
                try {
                    val sessionConfig = if (db != null) SessionConfig.forDatabase(db) else SessionConfig.defaultConfig()
                    driver!!.session(sessionConfig).use { neoSession: Session ->
                        try {
                            neoSession.run("MATCH (n) RETURN n LIMIT 0").consume()
                            logger.info { "[$workspace] Connected to ${db ?: "default"} at $uri" }
                            connected = true
                            this@Neo4jGraphStorage.database = db
                        } catch (e: ServiceUnavailableException) {
                            logger.error { "[$workspace] Database ${db ?: "default"} at $uri is not available" }
                            throw e
                        }
                    }
                } catch (e: Neo4jException) {
                    if (e is ClientException && e.code() == "Neo.ClientError.Database.DatabaseNotFound") {
                        logger.info { "[$workspace] Database $db at $uri not found. Try to create specified database." }
                        try {
                            driver!!.session().use { neoSession: Session ->
                                neoSession.run("CREATE DATABASE `$db` IF NOT EXISTS").consume()
                                logger.info { "[$workspace] Database $db at $uri created" }
                                connected = true
                                this@Neo4jGraphStorage.database = db
                            }
                        } catch (createEx: Neo4jException) {
                            if (db == null) {
                                logger.error { "[$workspace] Failed to create $db at $uri" }
                                throw createEx
                            }
                            logger.warn {
                                "[$workspace] Failed to create database $db, fallback to default. " +
                                    "Error: ${createEx.message}"
                            }
                        }
                    } else if (db == null && !connected) {
                        // If we failed on default DB too
                        logger.error { "[$workspace] Authentication or connection failed: ${e.message}" }
                        throw e
                    }
                }
                if (connected) break
            }

            if (connected) {
                val workspaceLabel = getWorkspaceLabel()
                // Create B-Tree index
                try {
                    driver!!.session(getSessionConfig()).use { neoSession: Session ->
                        neoSession.run(
                            "CREATE INDEX IF NOT EXISTS FOR (n:`$workspaceLabel`) ON (n.entity_id)",
                        ).consume()
                        logger.info {
                            "[$workspace] Ensured B-Tree index on entity_id for $workspaceLabel in $database"
                        }
                    }
                } catch (e: Neo4jException) {
                    logger.warn { "[$workspace] Failed to create B-Tree index: ${e.message}" }
                }

                // Create full-text index
                createFulltextIndex(driver!!, database, workspaceLabel)
            }
        }
    }

    private fun createFulltextIndex(
        driver: Driver,
        database: String?,
        workspaceLabel: String,
    ) {
        val indexName = getFulltextIndexName(workspaceLabel)
        val legacyIndexName = "entity_id_fulltext_idx"

        try {
            val sessionConfig = if (database != null) SessionConfig.forDatabase(database) else SessionConfig.defaultConfig()
            driver.session(sessionConfig).use { neoSession: Session ->
                val checkIndexQuery = "SHOW FULLTEXT INDEXES"
                val result = neoSession.run(checkIndexQuery)
                val indexes = result.list().map { it.asMap() }

                var existingIndex: Map<String, Any>? = null
                var legacyIndex: Map<String, Any>? = null

                for (idx in indexes) {
                    if (idx["name"] == indexName) {
                        existingIndex = idx
                    } else if (idx["name"] == legacyIndexName) {
                        legacyIndex = idx
                    }
                }

                if (legacyIndex != null && existingIndex == null) {
                    logger.info { "[$workspace] Found legacy index '$legacyIndexName'. Migrating to '$indexName'." }
                    try {
                        neoSession.run("DROP INDEX $legacyIndexName IF EXISTS").consume()
                        logger.info { "[$workspace] Dropped legacy index '$legacyIndexName'" }
                    } catch (e: Neo4jException) {
                        logger.warn { "[$workspace] Failed to drop legacy index: ${e.message}" }
                    }
                }

                if (existingIndex != null) {
                    val indexState = existingIndex["state"] as? String ?: "UNKNOWN"
                    logger.info { "[$workspace] Found existing index '$indexName' with state: $indexState" }
                    if (indexState == "ONLINE") {
                        logger.info {
                            "[$workspace] Full-text index '$indexName' already exists and is online. " +
                                "Skipping recreation."
                        }
                        return
                    } else {
                        logger.warn {
                            "[$workspace] Existing index '$indexName' is not online (state: $indexState). " +
                                "Will recreate."
                        }
                    }
                } else {
                    logger.info { "[$workspace] No existing index '$indexName' found. Creating new index." }
                }

                val needsRecreation = existingIndex != null && existingIndex["state"] != "ONLINE"
                val needsCreation = existingIndex == null

                if (needsRecreation || needsCreation) {
                    if (needsRecreation) {
                        try {
                            neoSession.run("DROP INDEX $indexName IF EXISTS").consume()
                            logger.info { "[$workspace] Dropped existing index '$indexName'" }
                        } catch (e: Neo4jException) {
                            logger.warn { "[$workspace] Failed to drop existing index: ${e.message}" }
                        }
                    }

                    logger.info {
                        "[$workspace] Creating full-text index '$indexName' with Chinese tokenizer support."
                    }
                    try {
                        val createIndexQuery =
                            """
                            CREATE FULLTEXT INDEX $indexName
                            FOR (n:`$workspaceLabel`) ON EACH [n.entity_id]
                            OPTIONS {
                                indexConfig: {
                                    `fulltext.analyzer`: 'cjk',
                                    `fulltext.eventually_consistent`: true
                                }
                            }
                            """.trimIndent()
                        neoSession.run(createIndexQuery).consume()
                        logger.info {
                            "[$workspace] Successfully created full-text index '$indexName' with CJK analyzer."
                        }
                    } catch (cjkError: Neo4jException) {
                        logger.warn {
                            "[$workspace] CJK analyzer not supported: ${cjkError.message}. " +
                                "Falling back to standard analyzer."
                        }
                        val createIndexQuery =
                            """
                            CREATE FULLTEXT INDEX $indexName
                            FOR (n:`$workspaceLabel`) ON EACH [n.entity_id]
                            """.trimIndent()
                        neoSession.run(createIndexQuery).consume()
                        logger.info {
                            "[$workspace] Successfully created full-text index '$indexName' with standard analyzer."
                        }
                    }
                }
            }
        } catch (e: Neo4jException) {
            if (e.message?.contains("Unknown command") == true ||
                e.message?.lowercase()?.contains("invalid syntax") == true
            ) {
                logger.warn {
                    "[$workspace] Could not create or verify full-text index '$indexName'. " +
                        "Use Neo4j version that supports it."
                }
            } else {
                logger.error {
                    "[$workspace] Failed to create or verify full-text index '$indexName': ${e.message}"
                }
            }
        }
    }

    override suspend fun finalize() {
        driver?.close()
        driver = null
    }

    override suspend fun indexDoneCallback() {
        // Neo4j handles persistence
    }

    override suspend fun drop(): Map<String, String> {
        val workspaceLabel = getWorkspaceLabel()
        return withContext(Dispatchers.IO) {
            try {
                driver!!.session(getSessionConfig()).use { neoSession: Session ->
                    val query = "MATCH (n:`$workspaceLabel`) DETACH DELETE n"
                    neoSession.run(query).consume()
                }
                mapOf("status" to "success", "message" to "workspace '$workspaceLabel' data dropped")
            } catch (e: Neo4jException) {
                logger.error { "[$workspace] Error dropping Neo4j workspace '$workspaceLabel': ${e.message}" }
                mapOf("status" to "error", "message" to e.message.toString())
            }
        }
    }

    override suspend fun hasNode(nodeId: String): Boolean =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query = "MATCH (n:`$workspaceLabel` {entity_id: \$entity_id}) RETURN count(n) > 0 AS node_exists"
                val result = neoSession.run(query, mapOf("entity_id" to nodeId))
                if (result.hasNext()) {
                    result.single().get("node_exists").asBoolean()
                } else {
                    false
                }
            }
        }

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (a:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})-[r]-(b:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                    RETURN COUNT(r) > 0 AS edgeExists
                    """.trimIndent()
                val result =
                    neoSession.run(
                        query,
                        mapOf(
                            "source_entity_id" to sourceNodeId,
                            "target_entity_id" to targetNodeId,
                        ),
                    )
                if (result.hasNext()) {
                    result.single().get("edgeExists").asBoolean()
                } else {
                    false
                }
            }
        }

    override suspend fun nodeDegree(nodeId: String): Int =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN COUNT(r) AS degree
                    """.trimIndent()
                val result = neoSession.run(query, mapOf("entity_id" to nodeId))
                if (result.hasNext()) {
                    result.single().get("degree").asInt()
                } else {
                    0
                }
            }
        }

    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int {
        val srcDegree = nodeDegree(srcId)
        val tgtDegree = nodeDegree(tgtId)
        return srcDegree + tgtDegree
    }

    override suspend fun getNode(nodeId: String): Map<String, String>? =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query = "MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id}) RETURN n"
                val result = neoSession.run(query, mapOf("entity_id" to nodeId))
                if (result.hasNext()) {
                    val record = result.next()
                    val node = record.get("n").asNode()
                    val nodeMap = node.asMap().toMutableMap()
                    if (nodeMap.containsKey("labels")) {
                        @Suppress("UNCHECKED_CAST")
                        val labels = nodeMap["labels"] as? List<String>
                        if (labels != null) {
                            nodeMap["labels"] = labels.filter { it != workspaceLabel }
                        }
                    }
                    nodeMap.entries.associate { it.key to it.value.toString() }
                } else {
                    null
                }
            }
        }

    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, String>? =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (start:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})-[r]-(end:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                    RETURN properties(r) as edge_properties
                    """.trimIndent()
                val result =
                    neoSession.run(
                        query,
                        mapOf(
                            "source_entity_id" to sourceNodeId,
                            "target_entity_id" to targetNodeId,
                        ),
                    )

                if (result.hasNext()) {
                    val record = result.next()
                    val edgeProps = record.get("edge_properties").asMap().toMutableMap()

                    val requiredKeys =
                        mapOf(
                            "weight" to 1.0,
                            "source_id" to null,
                            "description" to null,
                            "keywords" to null,
                        )
                    requiredKeys.forEach { (key, defaultVal) ->
                        if (!edgeProps.containsKey(key)) {
                            edgeProps[key] = defaultVal
                        }
                    }
                    edgeProps.entries.associate { it.key to it.value.toString() }
                } else {
                    null
                }
            }
        }

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>? =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                    OPTIONAL MATCH (n)-[r]-(connected:`$workspaceLabel`)
                    WHERE connected.entity_id IS NOT NULL
                    RETURN n, r, connected
                    """.trimIndent()
                val result = neoSession.run(query, mapOf("entity_id" to sourceNodeId))
                val edges = mutableListOf<Pair<String, String>>()
                while (result.hasNext()) {
                    val record = result.next()
                    val sourceNode = record.get("n").asNode()
                    val connectedNode = record.get("connected")

                    if (!connectedNode.isNull) {
                        val cNode = connectedNode.asNode()
                        val sourceLabel = sourceNode.get("entity_id").asString()
                        val targetLabel = cNode.get("entity_id").asString()
                        if (sourceLabel != null && targetLabel != null) {
                            edges.add(sourceLabel to targetLabel)
                        }
                    }
                }
                edges
            }
        }

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, String>,
    ) = withContext(Dispatchers.IO) {
        val workspaceLabel = getWorkspaceLabel()
        val properties = nodeData.toMutableMap()
        val entityType = properties["entity_type"] ?: "UNKNOWN"
        require(properties.containsKey("entity_id")) { "Neo4j: node properties must contain an 'entity_id' field" }

        driver!!.session(getSessionConfig()).use { neoSession: Session ->
            neoSession.executeWrite { tx ->
                val query =
                    """
                    MERGE (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                    SET n += ${'$'}properties
                    SET n:`$entityType`
                    """.trimIndent()
                tx.run(query, mapOf("entity_id" to nodeId, "properties" to properties)).consume()
            }
        }
        Unit
    }

    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, String>,
    ) = withContext(Dispatchers.IO) {
        val workspaceLabel = getWorkspaceLabel()
        driver!!.session(getSessionConfig()).use { neoSession: Session ->
            neoSession.executeWrite { tx ->
                val query =
                    """
                    MATCH (source:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})
                    WITH source
                    MATCH (target:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                    MERGE (source)-[r:DIRECTED]-(target)
                    SET r += ${'$'}properties
                    RETURN r
                    """.trimIndent()
                tx.run(
                    query,
                    mapOf(
                        "source_entity_id" to sourceNodeId,
                        "target_entity_id" to targetNodeId,
                        "properties" to edgeData,
                    ),
                ).consume()
            }
        }
        Unit
    }

    override suspend fun deleteNode(nodeId: String) =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                neoSession.executeWrite { tx ->
                    val query =
                        """
                        MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id})
                        DETACH DELETE n
                        """.trimIndent()
                    tx.run(query, mapOf("entity_id" to nodeId)).consume()
                }
            }
            Unit
        }

    override suspend fun removeNodes(nodes: List<String>) {
        nodes.forEach { deleteNode(it) }
    }

    override suspend fun removeEdges(edges: List<Pair<String, String>>) =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                neoSession.executeWrite { tx ->
                    val query =
                        """
                        MATCH (source:`$workspaceLabel` {entity_id: ${'$'}source_entity_id})-[r]-(target:`$workspaceLabel` {entity_id: ${'$'}target_entity_id})
                        DELETE r
                        """.trimIndent()
                    edges.forEach { (src, tgt) ->
                        tx.run(query, mapOf("source_entity_id" to src, "target_entity_id" to tgt)).consume()
                    }
                }
            }
            Unit
        }

    override suspend fun getAllLabels(): List<String> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel`)
                    WHERE n.entity_id IS NOT NULL
                    RETURN DISTINCT n.entity_id AS label
                    ORDER BY label
                    """.trimIndent()
                val result = neoSession.run(query)
                result.list().map { it.get("label").asString() }
            }
        }

    override suspend fun getKnowledgeGraph(
        nodeLabel: String,
        maxDepth: Int,
        maxNodes: Int,
    ): KnowledgeGraph =
        withContext(Dispatchers.IO) {
            // Check maxNodes global config if needed, here we assume argument is correct
            var effectiveMaxNodes = maxNodes
            val maxGraphNodes = (globalConfig["max_graph_nodes"] as? Number)?.toInt() ?: DEFAULT_MAX_NODES
            effectiveMaxNodes = min(effectiveMaxNodes, maxGraphNodes)

            val workspaceLabel = getWorkspaceLabel()

            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                if (nodeLabel == "*") {
                    // Wildcard case
                    val countQuery = "MATCH (n:`$workspaceLabel`) RETURN count(n) as total"
                    val countResult = neoSession.run(countQuery).single()
                    val total = countResult.get("total").asInt()

                    if (total > effectiveMaxNodes) {
                        logger.info {
                            "[$workspace] Graph truncated: $total nodes found, limited to $effectiveMaxNodes"
                        }
                    }
                }
            }

            if (nodeLabel == "*") {
                return@withContext getKnowledgeGraphWildcard(effectiveMaxNodes)
            }

            // Implementation using Robust BFS (Python's _robust_fallback)
            robustFallback(nodeLabel, maxDepth, effectiveMaxNodes)
        }

    @Suppress("LoopWithTooManyJumpStatements", "ReturnCount")
    private suspend fun robustFallback(
        nodeLabel: String,
        maxDepth: Int,
        maxNodes: Int,
    ): KnowledgeGraph {
        val nodes = mutableListOf<Map<String, Any>>()
        val edges = mutableListOf<Map<String, Any>>()
        var isTruncated = false
        val visitedNodes = mutableSetOf<String>() // entity_id
        val visitedEdges = mutableSetOf<String>() // edge elementId
        val visitedEdgePairs = mutableSetOf<Set<String>>()

        val queue = ArrayDeque<Triple<Map<String, Any>, Map<String, Any>?, Int>>() // Node, Edge?, Depth

        val workspaceLabel = getWorkspaceLabel()

        if (nodeLabel == "*") {
            return getKnowledgeGraphWildcard(maxNodes)
        }

        driver!!.session(getSessionConfig()).use { neoSession: Session ->
            val query = "MATCH (n:`$workspaceLabel` {entity_id: ${'$'}entity_id}) RETURN n"
            val result = neoSession.run(query, mapOf("entity_id" to nodeLabel))
            if (result.hasNext()) {
                val node = result.next().get("n").asNode()
                val props = node.asMap().toMutableMap()
                val entityId = props["entity_id"]?.toString() ?: return@use
                queue.add(Triple(props, null, 0))
            }
        }

        while (queue.isNotEmpty() && visitedNodes.size < maxNodes) {
            val (currentNode, currentEdge, currentDepth) = queue.removeFirst()
            val currentEntityId = currentNode["entity_id"].toString()

            if (shouldSkipNode(currentEntityId, currentDepth, maxDepth, visitedNodes)) continue

            nodes.add(currentNode)
            visitedNodes.add(currentEntityId)

            if (currentEdge != null) {
                edges.add(currentEdge)
            }

            if (visitedNodes.size >= maxNodes) {
                isTruncated = true
                break
            }

            processNeighbors(
                currentEntityId,
                currentDepth,
                maxDepth,
                visitedNodes,
                visitedEdges,
                visitedEdgePairs,
                edges,
                queue,
                workspaceLabel,
            )
        }

        return KnowledgeGraph(nodes, edges, isTruncated)
    }

    private fun shouldSkipNode(
        entityId: String,
        depth: Int,
        maxDepth: Int,
        visitedNodes: Set<String>,
    ): Boolean {
        return visitedNodes.contains(entityId) || depth > maxDepth
    }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun processNeighbors(
        currentEntityId: String,
        currentDepth: Int,
        maxDepth: Int,
        visitedNodes: Set<String>,
        visitedEdges: MutableSet<String>,
        visitedEdgePairs: MutableSet<Set<String>>,
        edges: MutableList<Map<String, Any>>,
        queue: ArrayDeque<Triple<Map<String, Any>, Map<String, Any>?, Int>>,
        workspaceLabel: String,
    ) {
        driver!!.session(getSessionConfig()).use { neoSession: Session ->
            val query =
                """
                MATCH (a:`$workspaceLabel` {entity_id: ${'$'}entity_id})-[r]-(b)
                WITH r, b
                RETURN r, b
                """.trimIndent()
            val result = neoSession.run(query, mapOf("entity_id" to currentEntityId))
            val records = result.list()

            for (record in records) {
                val r = record.get("r").asRelationship()
                val b = record.get("b").asNode()
                val bProps = b.asMap().toMutableMap()
                val bEntityId = bProps["entity_id"]?.toString() ?: continue
                val rProps = r.asMap().toMutableMap()
                val edgeId = r.elementId()

                if (visitedEdges.contains(edgeId)) continue

                val pair = setOf(currentEntityId, bEntityId)
                val edgeMap = rProps
                edgeMap["source"] = currentEntityId
                edgeMap["target"] = bEntityId

                if (pair !in visitedEdgePairs) {
                    if (visitedNodes.contains(bEntityId) || currentDepth < maxDepth) {
                        edges.add(edgeMap)
                        visitedEdges.add(edgeId)
                        visitedEdgePairs.add(pair)
                    }
                }

                if (!visitedNodes.contains(bEntityId)) {
                    if (currentDepth < maxDepth) {
                        queue.add(Triple(bProps, null, currentDepth + 1))
                    }
                }
            }
        }
    }

    private fun getKnowledgeGraphWildcard(maxNodes: Int): KnowledgeGraph {
        val workspaceLabel = getWorkspaceLabel()
        val nodes = mutableListOf<Map<String, Any>>()
        val edges = mutableListOf<Map<String, Any>>()
        var isTruncated = false

        driver!!.session(getSessionConfig()).use { neoSession: Session ->
            val countQuery = "MATCH (n:`$workspaceLabel`) RETURN count(n) as total"
            val total = neoSession.run(countQuery).single().get("total").asInt()

            if (total > maxNodes) {
                isTruncated = true
            }

            val pyQuery =
                """
                MATCH (n:`$workspaceLabel`)
                OPTIONAL MATCH (n)-[r]-()
                WITH n, COALESCE(count(r), 0) AS degree
                ORDER BY degree DESC
                LIMIT ${'$'}max_nodes
                WITH collect({node: n}) AS filtered_nodes
                UNWIND filtered_nodes AS node_info
                WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                OPTIONAL MATCH (a)-[r]-(b)
                WHERE a IN kept_nodes AND b IN kept_nodes
                RETURN filtered_nodes AS node_info,
                       collect(DISTINCT r) AS relationships
                """.trimIndent()

            val result = neoSession.run(pyQuery, mapOf("max_nodes" to maxNodes))
            if (result.hasNext()) {
                val record = result.single()
                val nodeInfos = record.get("node_info").asList { it.asMap() }
                val relationships = record.get("relationships").asList { it.asRelationship() }

                val idToEntityId = mutableMapOf<String, String>()

                nodeInfos.forEach { info ->
                    val node = info["node"] as Node
                    val props = node.asMap().toMutableMap()
                    val entityId = props["entity_id"] as? String
                    if (entityId != null) {
                        idToEntityId[node.elementId()] = entityId
                        nodes.add(props)
                    }
                }

                relationships.forEach { rel ->
                    val startId = rel.startNodeElementId()
                    val endId = rel.endNodeElementId()

                    val srcEntityId = idToEntityId[startId]
                    val tgtEntityId = idToEntityId[endId]

                    if (srcEntityId != null && tgtEntityId != null) {
                        val edgeProps = rel.asMap().toMutableMap()
                        edgeProps["source"] = srcEntityId
                        edgeProps["target"] = tgtEntityId
                        edges.add(edgeProps)
                    }
                }
            }
        }
        return KnowledgeGraph(nodes, edges, isTruncated)
    }

    override suspend fun getAllNodes(): List<Map<String, Any>> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query = "MATCH (n:`$workspaceLabel`) RETURN n"
                val result = neoSession.run(query)
                result.list().map { record ->
                    val node = record.get("n").asNode()
                    val map = node.asMap().toMutableMap()
                    map["id"] = map["entity_id"] ?: ""
                    map
                }
            }
        }

    override suspend fun getAllEdges(): List<Map<String, Any>> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (a:`$workspaceLabel`)-[r]-(b:`$workspaceLabel`)
                    RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
                    """.trimIndent()
                val result = neoSession.run(query)
                result.list().map { record ->
                    val props = record.get("properties").asMap().toMutableMap()
                    props["source"] = record.get("source").asString()
                    props["target"] = record.get("target").asString()
                    props
                }
            }
        }

    override suspend fun getPopularLabels(limit: Int): List<String> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                val query =
                    """
                    MATCH (n:`$workspaceLabel`)
                    WHERE n.entity_id IS NOT NULL
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n.entity_id AS label, count(r) AS degree
                    ORDER BY degree DESC, label ASC
                    LIMIT ${'$'}limit
                    RETURN label
                    """.trimIndent()
                val result = neoSession.run(query, mapOf("limit" to limit))
                result.list().map { it.get("label").asString() }
            }
        }

    override suspend fun searchLabels(
        query: String,
        limit: Int,
    ): List<String> =
        withContext(Dispatchers.IO) {
            val workspaceLabel = getWorkspaceLabel()
            val queryStrip = query.trim()
            if (queryStrip.isEmpty()) return@withContext emptyList<String>()

            val queryLower = queryStrip.lowercase()
            val isChinese = isChineseText(queryStrip)
            val indexName = getFulltextIndexName(workspaceLabel)

            driver!!.session(getSessionConfig()).use { neoSession: Session ->
                try {
                    // Try fulltext search
                    val cypherQuery: String
                    val params: MutableMap<String, Any> =
                        mutableMapOf(
                            "index_name" to indexName,
                            "query_strip" to queryStrip,
                            "query_lower" to queryLower,
                            "limit" to limit,
                        )

                    if (isChinese) {
                        cypherQuery =
                            """
                            CALL db.index.fulltext.queryNodes(${'$'}index_name, ${'$'}search_query) YIELD node, score
                            WITH node, score
                            WHERE node:`$workspaceLabel`
                            WITH node.entity_id AS label, score
                            WITH label, score,
                                 CASE
                                     WHEN label = ${'$'}query_strip THEN score + $SCORE_EXACT_MATCH
                                     WHEN label CONTAINS ${'$'}query_strip THEN score + $SCORE_CONTAINS
                                     ELSE score
                                 END AS final_score
                            RETURN label
                            ORDER BY final_score DESC, label ASC
                            LIMIT ${'$'}limit
                            """.trimIndent()
                        params["search_query"] = queryStrip
                    } else {
                        cypherQuery =
                            """
                            CALL db.index.fulltext.queryNodes(${'$'}index_name, ${'$'}search_query) YIELD node, score
                            WITH node, score
                            WHERE node:`$workspaceLabel`
                            WITH node.entity_id AS label, toLower(node.entity_id) AS label_lower, score
                            WITH label, label_lower, score,
                                 CASE
                                     WHEN label_lower = ${'$'}query_lower THEN score + $SCORE_EXACT_MATCH
                                     WHEN label_lower STARTS WITH ${'$'}query_lower THEN score + $SCORE_STARTS_WITH
                                     WHEN label_lower CONTAINS ' ' + ${'$'}query_lower OR label_lower CONTAINS '_' + ${'$'}query_lower THEN score + $SCORE_PARTIAL
                                     ELSE score
                                 END AS final_score
                            RETURN label
                            ORDER BY final_score DESC, label ASC
                            LIMIT ${'$'}limit
                            """.trimIndent()
                        params["search_query"] = "$queryStrip*"
                    }

                    val result = neoSession.run(cypherQuery, params)
                    result.list().map { it.get("label").asString() }
                } catch (e: Neo4jException) {
                    // Fallback
                    logger.warn {
                        "[$workspace] Full-text search failed: ${e.message}. Falling back to standard search."
                    }
                    val fallbackQuery: String
                    val fallbackParams = mutableMapOf<String, Any>("limit" to limit)

                    if (isChinese) {
                        fallbackQuery =
                            """
                            MATCH (n:`$workspaceLabel`)
                            WHERE n.entity_id IS NOT NULL
                            WITH n.entity_id AS label
                            WHERE label CONTAINS ${'$'}query_strip
                            WITH label,
                                 CASE
                                     WHEN label = ${'$'}query_strip THEN $SCORE_EXACT_MATCH
                                     WHEN label STARTS WITH ${'$'}query_strip THEN $SCORE_STARTS_WITH
                                     ELSE $FALLBACK_SCORE_BASE - size(label)
                                 END AS score
                            ORDER BY score DESC, label ASC
                            LIMIT ${'$'}limit
                            RETURN label
                            """.trimIndent()
                        fallbackParams["query_strip"] = queryStrip
                    } else {
                        fallbackQuery =
                            """
                            MATCH (n:`$workspaceLabel`)
                            WHERE n.entity_id IS NOT NULL
                            WITH n.entity_id AS label, toLower(n.entity_id) AS label_lower
                            WHERE label_lower CONTAINS ${'$'}query_lower
                            WITH label, label_lower,
                                 CASE
                                     WHEN label_lower = ${'$'}query_lower THEN $SCORE_EXACT_MATCH
                                     WHEN label_lower STARTS WITH ${'$'}query_lower THEN $SCORE_STARTS_WITH
                                     ELSE $FALLBACK_SCORE_BASE - size(label)
                                 END AS score
                            ORDER BY score DESC, label ASC
                            LIMIT ${'$'}limit
                            RETURN label
                            """.trimIndent()
                        fallbackParams["query_lower"] = queryLower
                    }

                    val result = neoSession.run(fallbackQuery, fallbackParams)
                    result.list().map { it.get("label").asString() }
                }
            }
        }
}
