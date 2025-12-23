package lightrag.kg.json

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.EmbeddingFunc
import java.io.File

class JsonKVStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    override val embeddingFunc: EmbeddingFunc? = null,
) : BaseKVStorage {
    private val data = mutableMapOf<String, Map<String, Any>>()
    private val mutex = Mutex()
    private val workingDir = File(globalConfig["working_dir"] as? String ?: "./rag_storage")
    private val file = File(workingDir, "kv_store_$namespace.json")

    override suspend fun initialize() {
        if (!workingDir.exists()) {
            workingDir.mkdirs()
        }
        if (file.exists()) {
            try {
                val content = file.readText()
                // Simple JSON deserialization (needs refining for complex types)
                // For now assuming simplistic map structure for prototype
                // Real implementation would need robust JSON handling for Map<String, Any>
            } catch (e: Exception) {
                println("Error loading KV storage: ${e.message}")
            }
        }
    }

    override suspend fun indexDoneCallback() {
        mutex.withLock {
            // Serialize and save to file
            // Note: kotlinx.serialization with Map<String, Any> is tricky.
            // Often requires custom serializers or Contextual serialization.
            // For this skeleton, we assume in-memory persistence only or stub file IO
        }
    }

    override suspend fun getById(id: String): Map<String, Any>? =
        mutex.withLock {
            data[id]
        }

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> =
        mutex.withLock {
            ids.mapNotNull { data[it] }
        }

    override suspend fun filterKeys(keys: Set<String>): Set<String> =
        mutex.withLock {
            keys.filter { !data.containsKey(it) }.toSet()
        }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        mutex.withLock {
            this.data.putAll(data)
        }
    }

    override suspend fun delete(ids: List<String>) {
        mutex.withLock {
            ids.forEach { this.data.remove(it) }
        }
    }

    override suspend fun drop(): Map<String, String> {
        mutex.withLock {
            data.clear()
            if (file.exists()) {
                file.delete()
            }
        }
        return mapOf("status" to "success", "message" to "data dropped")
    }

    override suspend fun isEmpty(): Boolean =
        mutex.withLock {
            data.isEmpty()
        }
}
