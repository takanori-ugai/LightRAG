package lightrag.kg.neo4j

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import lightrag.core.Neo4jConfig
import lightrag.core.types.BaseVectorStorage
import lightrag.kg.memory.Metadata
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.SessionConfig
import java.util.concurrent.TimeUnit

private val logger = KotlinLogging.logger {}

/**
 * A simple Neo4j-backed vector storage. It keeps an in-memory cache for similarity
 * calculations and persists vectors/metadata as nodes in Neo4j for durability.
 */
class Neo4jVectorStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
    private val cosineThreshold: Double? = null,
) : BaseVectorStorage {
    override val cosineBetterThanThreshold: Double =
        cosineThreshold ?: (globalConfig["cosine_better_than_threshold"] as? Double ?: 0.2)
    override val metaFields: Set<String> = emptySet()

    private var driver: Driver? = null
    private val label: String = sanitizeLabel("${namespace}_$workspace")

    private val vectors = mutableMapOf<String, List<Float>>()
    private val metadata = mutableMapOf<String, Metadata>()

    private fun sanitizeLabel(raw: String): String {
        var result = raw.replace(Regex("[^A-Za-z0-9_]+"), "_").trim('_')
        if (result.isEmpty()) result = "base"
        if (!result.first().isLetter() && result.first() != '_') {
            result = "l_$result"
        }
        return result
    }

    private fun parseNeo4jConfig(): Neo4jConfig? {
        return when (val cfg = globalConfig["neo4j"]) {
            is Neo4jConfig -> cfg
            is Map<*, *> ->
                Neo4jConfig(
                    uri = cfg["uri"] as? String,
                    username = cfg["username"] as? String,
                    password = cfg["password"] as? String,
                    database = cfg["database"] as? String,
                )
            else -> null
        }
    }

    private fun sessionConfig(): SessionConfig {
        val database = System.getenv("NEO4J_DATABASE") ?: parseNeo4jConfig()?.database
        return database?.let { SessionConfig.forDatabase(it) } ?: SessionConfig.defaultConfig()
    }

    override suspend fun initialize() {
        val cfg = parseNeo4jConfig()
        val uri = System.getenv("NEO4J_URI") ?: cfg?.uri
        val username = System.getenv("NEO4J_USERNAME") ?: cfg?.username
        val password = System.getenv("NEO4J_PASSWORD") ?: cfg?.password

        if (uri == null) {
            logger.error { "Neo4jVectorStorage[$namespace] NEO4J_URI not configured" }
            return
        }

        val auth =
            if (username != null && password != null) {
                AuthTokens.basic(username, password)
            } else {
                AuthTokens.none()
            }

        val maxPool = System.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE")?.toIntOrNull() ?: 50
        val timeoutMs = System.getenv("NEO4J_CONNECTION_TIMEOUT")?.toLongOrNull() ?: 30_000L
        val maxLifetimeMs = System.getenv("NEO4J_MAX_CONNECTION_LIFETIME")?.toLongOrNull() ?: 300_000L

        val config =
            org.neo4j.driver.Config.builder()
                .withMaxConnectionPoolSize(maxPool)
                .withConnectionTimeout(timeoutMs, TimeUnit.MILLISECONDS)
                .withMaxConnectionLifetime(maxLifetimeMs, TimeUnit.MILLISECONDS)
                .build()

        driver = GraphDatabase.driver(uri, auth, config)

        withContext(Dispatchers.IO) {
            driver?.session(sessionConfig())?.use { session ->
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:$label) REQUIRE n.id IS UNIQUE",
                ).consume()
            }
        }

        loadFromNeo()
    }

    override suspend fun finalize() {
        driver?.close()
    }

    private suspend fun loadFromNeo() {
        val sessionCfg = sessionConfig()
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                val result =
                    session.run("MATCH (n:$label) RETURN n.id AS id, n.vector AS vector, n.metadata AS metadata")
                while (result.hasNext()) {
                    val record = result.next()
                    val id = record["id"].asString()
                    val vec =
                        record["vector"]?.let { value ->
                            value.asList { it.asDouble() }.map { it.toFloat() }
                        } ?: emptyList()
                    val rawMeta =
                        record["metadata"]?.asList { it.asString() }?.associate { entry ->
                            val parts = entry.split(":", limit = 2)
                            if (parts.size == 2) parts[0] to parts[1] else entry to ""
                        } ?: emptyMap()
                    vectors[id] = vec
                    metadata[id] = mapToMetadata(rawMeta)
                }
                if (logger.isInfoEnabled()) {
                    logger.info { "[$namespace/$workspace] Loaded ${vectors.size} vectors from Neo4j label '$label'" }
                }
            }
        }
    }

    override suspend fun indexDoneCallback() {
        // no-op; data is persisted on each upsert
    }

    override suspend fun drop(): Map<String, String> {
        withContext(Dispatchers.IO) {
            driver?.session(sessionConfig())?.use { session ->
                session.run("MATCH (n:$label) DETACH DELETE n").consume()
            }
        }
        vectors.clear()
        metadata.clear()
        return mapOf("status" to "success", "message" to "Deleted all nodes for label $label")
    }

    override suspend fun query(
        query: String,
        topK: Int,
        queryEmbedding: List<Float>?,
    ): List<Map<String, Any>> {
        val qVec = queryEmbedding ?: embed(query)
        if (qVec.isEmpty()) return emptyList()

        val scored =
            vectors.mapNotNull { (id, vec) ->
                if (vec.isEmpty()) return@mapNotNull null
                val score = CosineSimilarity.between(Embedding.from(qVec), Embedding.from(vec))
                if (score >= cosineBetterThanThreshold) {
                    Triple(id, score, metadata[id])
                } else {
                    null
                }
            }.sortedByDescending { it.second }
                .take(topK)

        return scored.map { (id, score, meta) ->
            val raw = meta?.raw ?: emptyMap()
            raw + mapOf("id" to id, "score" to score, "distance" to score)
        }
    }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        data.forEach { (id, metaMap) ->
            val meta = mapToMetadata(metaMap)
            metadata[id] = meta
            val vec =
                when {
                    meta.vector != null -> meta.vector
                    meta.content != null -> embed(meta.content)
                    else -> emptyList()
                }
            if (vec.isEmpty()) {
                logger.warn { "[$namespace/$workspace] No vector for id=$id; skipping persist" }
                return@forEach
            }
            vectors[id] = vec
            persistNode(id, vec, meta.raw)
        }
    }

    override suspend fun deleteEntity(entityName: String) {
        val idsToDelete =
            metadata.filter { it.value.entityName == entityName }.keys
        delete(idsToDelete.toList())
    }

    override suspend fun deleteEntityRelation(entityName: String) {
        val idsToDelete =
            metadata.filter { it.value.srcId == entityName || it.value.tgtId == entityName }.keys
        delete(idsToDelete.toList())
    }

    override suspend fun getById(id: String): Map<String, Any>? {
        val meta = metadata[id]?.raw ?: return null
        return meta + mapOf("id" to id, "vector" to (vectors[id] ?: emptyList<Float>()))
    }

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> {
        return ids.mapNotNull { getById(it) }
    }

    override suspend fun delete(ids: List<String>) {
        if (ids.isEmpty()) return
        withContext(Dispatchers.IO) {
            driver?.session(sessionConfig())?.use { session ->
                session.run(
                    "MATCH (n:$label) WHERE n.id IN \$ids DETACH DELETE n",
                    mapOf("ids" to ids),
                ).consume()
            }
        }
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

    private fun persistNode(
        id: String,
        vector: List<Float>,
        rawMeta: Map<String, Any>,
    ) {
        val metaList =
            rawMeta.entries.map { (k, v) ->
                val value =
                    when (v) {
                        null -> ""
                        is Number, is Boolean -> v.toString()
                        is List<*> -> v.filterNotNull().joinToString(",")
                        else -> v.toString()
                    }
                "$k:$value"
            }
        val params =
            mapOf(
                "id" to id,
                "vector" to vector.map { it.toDouble() },
                "metadata" to metaList,
            )
        driver?.session(sessionConfig())?.use { session ->
            session.run(
                "MERGE (n:$label {id: \$id}) SET n.vector = \$vector, n.metadata = \$metadata",
                params,
            ).consume()
        }
    }

    private fun embed(text: String): List<Float> {
        return try {
            val response = embeddingFunc.embed(text)
            val content = response.content()
            if (content is Embedding) {
                content.vector().toList()
            } else {
                emptyList()
            }
        } catch (e: Exception) {
            logger.error(e) { "Error embedding text for Neo4jVectorStorage" }
            emptyList()
        }
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
}
