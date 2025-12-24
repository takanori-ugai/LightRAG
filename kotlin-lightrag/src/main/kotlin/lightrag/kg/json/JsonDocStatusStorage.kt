package lightrag.kg.json

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import lightrag.core.types.DocProcessingStatus
import lightrag.core.types.DocStatus
import lightrag.core.types.DocStatusStorage
import lightrag.core.types.EmbeddingFunc
import java.lang.Math.min

class JsonDocStatusStorage(
    override val namespace: String,
    override val workspace: String,
    override val globalConfig: Map<String, Any> = emptyMap(),
    override val embeddingFunc: EmbeddingFunc? = null,
) : DocStatusStorage {
    private val docs = mutableMapOf<String, DocProcessingStatus>()
    private val mutex = Mutex()

    override suspend fun initialize() {
        // Load from file if exists
    }

    override suspend fun indexDoneCallback() {
        // Save to file
    }

    override suspend fun getById(id: String): Map<String, Any>? =
        mutex.withLock {
            docs[id]?.let {
                // Convert DocProcessingStatus to Map<String, Any>
                mapOf(
                    "status" to it.status.value,
                    "content_summary" to it.contentSummary,
                    "content_length" to it.contentLength,
                    "created_at" to it.createdAt,
                    "updated_at" to it.updatedAt,
                    "file_path" to it.filePath,
                    "track_id" to (it.trackId ?: ""),
                    "chunks_count" to (it.chunksCount ?: 0),
                    "error_msg" to (it.errorMsg ?: ""),
                )
            }
        }

    override suspend fun getByIds(ids: List<String>): List<Map<String, Any>> {
        return ids.mapNotNull { getById(it) }
    }

    override suspend fun filterKeys(keys: Set<String>): Set<String> =
        mutex.withLock {
            keys.filter { !docs.containsKey(it) }.toSet()
        }

    override suspend fun upsert(data: Map<String, Map<String, Any>>) {
        mutex.withLock {
            data.forEach { (id, map) ->
                val existing = docs[id]
                val statusStr = map["status"] as? String ?: existing?.status?.value ?: DocStatus.PENDING.value
                val status = DocStatus.values().find { it.value == statusStr } ?: DocStatus.PENDING

                val newDoc =
                    DocProcessingStatus(
                        status = status,
                        contentSummary = map["content_summary"] as? String ?: existing?.contentSummary ?: "",
                        contentLength =
                            (map["content_length"]?.toString()?.toIntOrNull())
                                ?: existing?.contentLength ?: 0,
                        createdAt = map["created_at"] as? String ?: existing?.createdAt ?: "",
                        updatedAt = map["updated_at"] as? String ?: existing?.updatedAt ?: "",
                        filePath = map["file_path"] as? String ?: existing?.filePath ?: "",
                        trackId = map["track_id"] as? String ?: existing?.trackId,
                        errorMsg = map["error_msg"] as? String ?: existing?.errorMsg,
                    )
                docs[id] = newDoc
            }
        }
    }

    override suspend fun delete(ids: List<String>) {
        mutex.withLock {
            ids.forEach { docs.remove(it) }
        }
    }

    override suspend fun drop(): Map<String, String> {
        mutex.withLock {
            docs.clear()
        }
        return mapOf("status" to "success", "message" to "data dropped")
    }

    override suspend fun isEmpty(): Boolean =
        mutex.withLock {
            docs.isEmpty()
        }

    override suspend fun getStatusCounts(): Map<String, Int> =
        mutex.withLock {
            docs.values.groupingBy { it.status.value }.eachCount()
        }

    override suspend fun getDocsByStatus(status: DocStatus): Map<String, DocProcessingStatus> =
        mutex.withLock {
            docs.filterValues { it.status == status }
        }

    override suspend fun getDocsByTrackId(trackId: String): Map<String, DocProcessingStatus> =
        mutex.withLock {
            docs.filterValues { it.trackId == trackId }
        }

    override suspend fun getDocsPaginated(
        statusFilter: DocStatus?,
        page: Int,
        pageSize: Int,
        sortField: String,
        sortDirection: String,
    ): Pair<List<Pair<String, DocProcessingStatus>>, Int> =
        mutex.withLock {
            var filtered =
                if (statusFilter != null) {
                    docs.filterValues { it.status == statusFilter }.toList()
                } else {
                    docs.toList()
                }

            val total = filtered.size

            // Sorting logic (simplified)
            filtered = filtered.sortedBy { it.second.updatedAt }
            if (sortDirection == "desc") {
                filtered = filtered.reversed()
            }

            val start = (page - 1) * pageSize
            val end = min(start + pageSize, total)

            if (start >= total) {
                return@withLock Pair(emptyList(), total)
            }

            Pair(filtered.subList(start, end), total)
        }

    override suspend fun getAllStatusCounts(): Map<String, Int> = getStatusCounts()

    override suspend fun getDocByFilePath(filePath: String): Map<String, Any>? =
        mutex.withLock {
            val entry = docs.entries.find { it.value.filePath == filePath }
            entry?.let { getById(it.key) }
        }
}
