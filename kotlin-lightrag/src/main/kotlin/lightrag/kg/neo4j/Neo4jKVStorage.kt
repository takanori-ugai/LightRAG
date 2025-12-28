package lightrag.kg.neo4j

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull
import lightrag.core.Neo4jConfig
import lightrag.core.types.BaseKVStorage
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.SessionConfig
import org.neo4j.driver.Values
import java.util.concurrent.TimeUnit

private val logger = KotlinLogging.logger {}

/**
 * Simple Neo4j-backed KV storage.
 *
 * Values are stored under a label derived from namespace/workspace with properties:
 *  - id   : String (primary key)
 *  - data : Map<String, Any> (payload)
 *
 * A lightweight in-memory cache is kept for fast reads and to satisfy BaseKVStorage contract.
 */
class Neo4jKVStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : BaseKVStorage {
    private var driver: Driver? = null
    private val label: String = sanitizeLabel("${namespace}_${workspace}_kv")
    private val data = mutableMapOf<String, Map<String, Any>>()
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    private fun sanitizeLabel(raw: String): String {
        var result = raw.replace(Regex("[^A-Za-z0-9_]+"), "_").trim('_')
        if (result.isEmpty()) result = "base"
        if (!result.first().isLetter() && result.first() != '_') {
            result = "l_$result"
        }
        return result
    }

    private fun parseNeo4jConfig(): Neo4jConfig? =
        when (val cfg = globalConfig["neo4j"]) {
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
            logger.error { "Neo4jKVStorage[$namespace] NEO4J_URI not configured" }
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
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:$label) REQUIRE n.id IS UNIQUE").consume()
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
                val result = session.run("MATCH (n:$label) RETURN n.id AS id, n.data_json AS data_json")
                result.list().forEach { record ->
                    val id = record["id"].asString()
                    val rawJson = record["data_json"].asString(null)
                    val kv =
                        if (rawJson != null && rawJson.isNotBlank()) {
                            val parsed = json.parseToJsonElement(rawJson)
                            (parsed.toAny() as? Map<String, Any>) ?: emptyMap()
                        } else {
                            emptyMap()
                        }
                    data[id] = kv
                }
                logger.info { "Loaded ${data.size} KV records for '$namespace/$workspace' from Neo4j" }
            }
        }
    }

    override suspend fun indexDoneCallback() {
        // Writes happen eagerly in upsert/delete; nothing extra to do.
    }

    override suspend fun getById(id: String): Map<String, Any>? = data[id]

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> = ids.mapNotNull { data[it] }

    override suspend fun filterKeys(keys: Set<String>): Set<String> = keys.filter { !data.containsKey(it) }.toSet()

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        if (data.isEmpty()) return
        this.data.putAll(data)

        val rows =
            data.map { (id, payload) ->
                mapOf("id" to id, "data" to payload)
            }
        val sessionCfg = sessionConfig()
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                rows.forEach { row ->
                    val jsonStr =
                        json.encodeToString(
                            JsonElement.serializer(),
                            (row["data"] ?: emptyMap<String, Any>()).toJsonElement(),
                        )
                    session.run(
                        "MERGE (n:$label {id: \$id}) SET n.data_json = \$data_json",
                        Values.parameters("id", row["id"], "data_json", jsonStr),
                    ).consume()
                }
            }
        }
    }

    override suspend fun delete(ids: List<String>) {
        ids.forEach { data.remove(it) }
        if (ids.isEmpty()) return
        val sessionCfg = sessionConfig()
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                val cypher = "MATCH (n:$label) WHERE n.id IN " + "$" + "ids DETACH DELETE n"
                session.run(
                    cypher,
                    Values.parameters("ids", ids),
                ).consume()
            }
        }
    }

    override suspend fun drop(): Map<String, String> {
        data.clear()
        val sessionCfg = sessionConfig()
        withContext(Dispatchers.IO) {
            driver?.session(sessionCfg)?.use { session ->
                session.run("MATCH (n:$label) DETACH DELETE n").consume()
            }
        }
        return mapOf("status" to "success", "message" to "Neo4j KV data dropped for $label")
    }

    override suspend fun isEmpty(): Boolean = data.isEmpty()

    private fun Any?.toJsonElement(): JsonElement =
        when (this) {
            null -> JsonNull
            is Boolean -> JsonPrimitive(this)
            is Number -> JsonPrimitive(this)
            is String -> JsonPrimitive(this)
            is List<*> -> JsonArray(this.map { it.toJsonElement() })
            is Map<*, *> -> JsonObject(this.entries.associate { it.key.toString() to it.value.toJsonElement() })
            else -> JsonPrimitive(this.toString())
        }

    private fun JsonElement.toAny(): Any? =
        when (this) {
            is JsonNull -> null
            is JsonPrimitive ->
                if (isString) {
                    content
                } else {
                    booleanOrNull ?: longOrNull ?: doubleOrNull ?: content
                }
            is JsonArray -> this.map { it.toAny() }
            is JsonObject -> this.mapValues { it.value.toAny() }
        }
}
