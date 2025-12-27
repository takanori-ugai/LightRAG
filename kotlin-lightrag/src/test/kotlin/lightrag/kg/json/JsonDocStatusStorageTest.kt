package lightrag.kg.json

import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import lightrag.core.types.DocStatus
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class JsonDocStatusStorageTest {
    private val embeddingModel = TestEmbeddings.mockEmbeddingModel()

    @Test
    fun `initialize loads persisted statuses`() {
        runBlocking {
            val tempDir = File("build/tmp/json_doc_status_init_${System.currentTimeMillis()}").apply { mkdirs() }
            try {
                val storage =
                    JsonDocStatusStorage(
                        namespace = "init-ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )

                storage.upsert(
                    mapOf(
                        "doc-1" to
                            mapOf(
                                "status" to "processed",
                                "content_summary" to "summary",
                                "content_length" to 123,
                                "created_at" to "2024-01-01T00:00:00Z",
                                "updated_at" to "2024-01-02T00:00:00Z",
                                "file_path" to "/tmp/file.txt",
                                "track_id" to "track-42",
                                "chunks_count" to 2,
                            ),
                    ),
                )
                storage.indexDoneCallback()

                val reloaded =
                    JsonDocStatusStorage(
                        namespace = "init-ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )
                reloaded.initialize()

                val loaded = reloaded.getById("doc-1")
                assertNotNull(loaded)
                assertEquals("processed", loaded["status"])
                assertEquals("/tmp/file.txt", loaded["file_path"])
                assertEquals("track-42", loaded["track_id"])
                assertEquals(1, reloaded.getStatusCounts()["processed"])
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }

    @Test
    fun `upsert merges existing values when fields missing`() {
        runBlocking {
            val tempDir = File("build/tmp/json_doc_status_merge_${System.currentTimeMillis()}").apply { mkdirs() }
            try {
                val storage =
                    JsonDocStatusStorage(
                        namespace = "merge-ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )

                storage.upsert(
                    mapOf(
                        "doc-1" to
                            mapOf(
                                "status" to "pending",
                                "content_summary" to "initial summary",
                                "content_length" to 10,
                                "created_at" to "2024-02-01T10:00:00Z",
                                "updated_at" to "2024-02-01T10:00:00Z",
                                "file_path" to "/files/one",
                                "track_id" to "track-A",
                            ),
                    ),
                )

                storage.upsert(
                    mapOf(
                        "doc-1" to
                            mapOf(
                                "status" to "failed",
                                "error_msg" to "failed to parse",
                            ),
                    ),
                )

                val merged = storage.getById("doc-1")
                assertNotNull(merged)
                assertEquals("failed", merged["status"])
                assertEquals("initial summary", merged["content_summary"])
                assertEquals(10, merged["content_length"])
                assertEquals("/files/one", merged["file_path"])
                assertEquals("track-A", merged["track_id"])
                assertEquals("failed to parse", merged["error_msg"])
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }

    @Test
    fun `status queries and pagination respect updated timestamps`() {
        runBlocking {
            val tempDir = File("build/tmp/json_doc_status_query_${System.currentTimeMillis()}").apply { mkdirs() }
            try {
                val storage =
                    JsonDocStatusStorage(
                        namespace = "query-ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )

                storage.upsert(
                    mapOf(
                        "doc-a" to
                            mapOf(
                                "status" to "processed",
                                "content_summary" to "A",
                                "content_length" to 1,
                                "created_at" to "2024-03-01T00:00:00Z",
                                "updated_at" to "2024-03-01T00:00:00Z",
                                "file_path" to "/files/a",
                            ),
                        "doc-b" to
                            mapOf(
                                "status" to "processed",
                                "content_summary" to "B",
                                "content_length" to 2,
                                "created_at" to "2024-02-01T00:00:00Z",
                                "updated_at" to "2024-02-01T00:00:00Z",
                                "file_path" to "/files/b",
                            ),
                        "doc-c" to
                            mapOf(
                                "status" to "pending",
                                "content_summary" to "C",
                                "content_length" to 3,
                                "created_at" to "2024-01-15T00:00:00Z",
                                "updated_at" to "2024-01-15T00:00:00Z",
                                "file_path" to "/files/c",
                            ),
                    ),
                )

                val processed = storage.getDocsByStatus(DocStatus.PROCESSED)
                assertEquals(setOf("doc-a", "doc-b"), processed.keys)

                val missing = storage.filterKeys(setOf("doc-a", "doc-x"))
                assertEquals(setOf("doc-x"), missing)

                val (page1, total) =
                    storage.getDocsPaginated(
                        statusFilter = DocStatus.PROCESSED,
                        page = 1,
                        pageSize = 1,
                        sortField = "updated_at",
                        sortDirection = "desc",
                    )
                assertEquals(2, total)
                assertEquals(1, page1.size)
                assertEquals("doc-a", page1.first().first)

                val (page2, totalAgain) =
                    storage.getDocsPaginated(
                        statusFilter = DocStatus.PROCESSED,
                        page = 2,
                        pageSize = 1,
                        sortField = "updated_at",
                        sortDirection = "desc",
                    )
                assertEquals(2, totalAgain)
                assertEquals(1, page2.size)
                assertEquals("doc-b", page2.first().first)
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }

    @Test
    fun `drop clears state and deletes persisted file`() {
        runBlocking {
            val tempDir = File("build/tmp/json_doc_status_drop_${System.currentTimeMillis()}").apply { mkdirs() }
            try {
                val storage =
                    JsonDocStatusStorage(
                        namespace = "drop-ns",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                        embeddingFunc = embeddingModel,
                    )

                storage.upsert(
                    mapOf(
                        "doc-1" to
                            mapOf(
                                "status" to "processing",
                                "content_summary" to "drop me",
                                "content_length" to 5,
                                "created_at" to "2024-04-01T00:00:00Z",
                                "updated_at" to "2024-04-01T00:00:00Z",
                                "file_path" to "/files/drop",
                            ),
                    ),
                )
                storage.indexDoneCallback()

                val persistedFile = File(tempDir, "doc_status_drop-ns.json")
                assertTrue(persistedFile.exists())

                val result = storage.drop()

                assertEquals("success", result["status"])
                assertTrue(storage.isEmpty())
                assertTrue(!persistedFile.exists())
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }
}
