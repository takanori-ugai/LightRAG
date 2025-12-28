package lightrag.kg.json

import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull
import lightrag.core.types.BaseKVStorage
import java.io.File

private val logger = KotlinLogging.logger {}

/**
 * A JSON-backed key-value storage.
 * @property namespace The namespace of the storage.
 * @property workspace The workspace of the storage.
 * @property globalConfig The global configuration for the storage.
 * @property embeddingFunc The embedding model to use.
 */
class JsonKVStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any?> = emptyMap(),
    override val embeddingFunc: EmbeddingModel,
) : BaseKVStorage {
    private val data = mutableMapOf<String, KVEntry>()
    private val mutex = Mutex()
    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "kv_store_$namespace.json")

    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = true
        }

    /**
     * Initializes the storage by loading data from the JSON file.
     */
    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (file.exists()) {
            try {
                val content = file.readText()
                if (content.isNotBlank()) {
                    val jsonElement = json.parseToJsonElement(content)
                    if (jsonElement is JsonObject) {
                        val loadedData =
                            jsonElement.entries.associate { (k, v) ->
                                @Suppress("UNCHECKED_CAST")
                                val rawMap = (v.toAny() as? Map<String, Any>) ?: emptyMap()
                                k to KVEntry(KVValue(rawMap))
                            }
                        mutex.withLock {
                            data.putAll(loadedData)
                        }
                        logger.info { "Loaded ${loadedData.size} records from ${file.absolutePath}" }
                    }
                }
            } catch (e: Exception) {
                logger.error(e) { "Error loading KV storage from ${file.absolutePath}" }
            }
        }
    }

    /**
     * Saves the current state of the storage to the JSON file.
     */
    override suspend fun indexDoneCallback() {
        mutex.withLock {
            try {
                val jsonObject = JsonObject(data.mapValues { it.value.value.data.toJsonElement() as JsonObject })
                val content = json.encodeToString(JsonElement.serializer(), jsonObject)
                file.writeText(content)
                logger.debug { "Saved ${data.size} records to ${file.absolutePath}" }
            } catch (e: Exception) {
                logger.error(e) { "Error saving KV storage to ${file.absolutePath}" }
            }
        }
    }

    /**
     * Gets an item by its ID.
     * @param id The ID of the item to get.
     * @return A map representing the item.
     */
    override suspend fun getById(id: String): Map<String, Any>? =
        mutex.withLock {
            data[id]?.value?.data
        }

    /**
     * Gets items by their IDs.
     * @param ids The IDs of the items to get.
     * @return A list of maps representing the items.
     */
    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> =
        mutex.withLock {
            ids.mapNotNull { data[it]?.value?.data }
        }

    /**
     * Filters keys from the storage.
     * @param keys The keys to filter.
     * @return A set of the filtered keys.
     */
    override suspend fun filterKeys(keys: Set<String>): Set<String> =
        mutex.withLock {
            keys.filter { !data.containsKey(it) }.toSet()
        }

    /**
     * Upserts data into the storage.
     * @param data The data to upsert.
     */
    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        mutex.withLock {
            val wrapped = data.mapValues { KVEntry(KVValue(it.value)) }
            this.data.putAll(wrapped)
        }
    }

    /**
     * Deletes items by their IDs.
     * @param ids The IDs of the items to delete.
     */
    override suspend fun delete(ids: List<String>) {
        mutex.withLock {
            ids.forEach { this.data.remove(it) }
        }
    }

    /**
     * Drops the storage.
     * @return A map with the status of the operation.
     */
    override suspend fun drop(): Map<String, String> {
        mutex.withLock {
            data.clear()
            if (file.exists()) {
                file.delete()
            }
        }
        return mapOf("status" to "success", "message" to "data dropped")
    }

    /**
     * Checks if the storage is empty.
     * @return True if the storage is empty, false otherwise.
     */
    override suspend fun isEmpty(): Boolean =
        mutex.withLock {
            data.isEmpty()
        }

    private fun Any?.toJsonElement(): JsonElement {
        return when (this) {
            null -> JsonNull
            is Boolean -> JsonPrimitive(this)
            is Number -> JsonPrimitive(this)
            is String -> JsonPrimitive(this)
            is List<*> -> JsonArray(this.map { it.toJsonElement() })
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
            is JsonArray -> this.map { it.toAny() }
            is JsonObject -> this.mapValues { it.value.toAny() }
        }
    }
}
