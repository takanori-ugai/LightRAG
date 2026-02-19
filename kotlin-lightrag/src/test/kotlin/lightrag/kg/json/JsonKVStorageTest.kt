package lightrag.kg.json

import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Tests covering JSON-backed KV storage behaviors needed by query context assembly. */
class JsonKVStorageTest {
    @get:Rule
    val tempFolder = TemporaryFolder()

    private val embeddingModel = TestEmbeddings.mockEmbeddingModel()

    /** Ensures read APIs return payload data for requested keys. */
    @Test
    fun `getById and getByIds return payload`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "text_chunks",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()

            storage.upsert(
                mapOf(
                    "chunk-1" to mapOf("content" to "first chunk"),
                    "chunk-2" to mapOf("content" to "second chunk"),
                ),
            )

            val single = storage.getById("chunk-1")
            assertNotNull(single)
            assertEquals("first chunk", single["content"])
            assertNull(single["id"])

            val results = storage.getByIds(listOf("chunk-1", "chunk-2"))
            val contents = results.mapNotNull { it["content"] as? String }.toSet()
            assertEquals(setOf("first chunk", "second chunk"), contents)
        }
    }

    /** Tests filterKeys returns keys that are not present in storage. */
    @Test
    fun `filterKeys returns missing keys`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(
                mapOf(
                    "key-1" to mapOf("value" to "data1"),
                    "key-2" to mapOf("value" to "data2"),
                ),
            )

            val result = storage.filterKeys(setOf("key-1", "key-2", "key-3", "key-4"))
            assertEquals(setOf("key-3", "key-4"), result)
        }
    }

    /** Tests filterKeys with empty storage returns all keys. */
    @Test
    fun `filterKeys with empty storage returns all keys`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()

            val result = storage.filterKeys(setOf("key-1", "key-2"))
            assertEquals(setOf("key-1", "key-2"), result)
        }
    }

    /** Tests delete removes items from storage. */
    @Test
    fun `delete removes items`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(
                mapOf(
                    "key-1" to mapOf("value" to "data1"),
                    "key-2" to mapOf("value" to "data2"),
                    "key-3" to mapOf("value" to "data3"),
                ),
            )

            storage.delete(listOf("key-1", "key-3"))

            assertNull(storage.getById("key-1"))
            assertNotNull(storage.getById("key-2"))
            assertNull(storage.getById("key-3"))
        }
    }

    /** Tests drop clears all data and deletes the file. */
    @Test
    fun `drop clears data and deletes file`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))
            storage.indexDoneCallback()

            val file = File(tempDir, "kv_store_test.json")
            assertTrue(file.exists())

            val result = storage.drop()
            assertEquals("success", result["status"])
            assertTrue(storage.isEmpty())
            assertFalse(file.exists())
        }
    }

    /** Tests isEmpty returns correct status. */
    @Test
    fun `isEmpty returns correct status`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            assertTrue(storage.isEmpty())

            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))
            assertFalse(storage.isEmpty())

            storage.delete(listOf("key-1"))
            assertTrue(storage.isEmpty())
        }
    }

    /** Tests persistence round-trip saves and loads data correctly. */
    @Test
    fun `persistence round-trip preserves data`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val namespace = "persist_test"

            // Create and save data
            val storage1 =
                JsonKVStorage(
                    namespace = namespace,
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage1.initialize()
            storage1.upsert(
                mapOf(
                    "key-1" to mapOf("name" to "Alice", "age" to 30),
                    "key-2" to mapOf("name" to "Bob", "age" to 25),
                ),
            )
            storage1.indexDoneCallback()

            // Load data in a new instance
            val storage2 =
                JsonKVStorage(
                    namespace = namespace,
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage2.initialize()
            assertFalse(storage2.isEmpty())

            val item1 = storage2.getById("key-1")
            assertNotNull(item1)
            assertEquals("Alice", item1["name"])
            assertEquals(30L, item1["age"])

            val item2 = storage2.getById("key-2")
            assertNotNull(item2)
            assertEquals("Bob", item2["name"])
            assertEquals(25L, item2["age"])
        }
    }

    /** Tests getById returns null for non-existent key. */
    @Test
    fun `getById returns null for non-existent key`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            assertNull(storage.getById("non-existent"))
        }
    }

    /** Tests getByIds returns only existing items. */
    @Test
    fun `getByIds returns only existing items`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))

            val results = storage.getByIds(listOf("key-1", "key-2", "key-3"))
            assertEquals(1, results.size)
            assertEquals("data1", results[0]["value"])
        }
    }

    /** Tests upsert updates existing values. */
    @Test
    fun `upsert updates existing values`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(mapOf("key-1" to mapOf("value" to "original")))

            var item = storage.getById("key-1")
            assertEquals("original", item?.get("value"))

            storage.upsert(mapOf("key-1" to mapOf("value" to "updated")))

            item = storage.getById("key-1")
            assertEquals("updated", item?.get("value"))
        }
    }

    /** Tests storage handles various data types. */
    @Test
    fun `handles various data types`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(
                mapOf(
                    "key-1" to
                        mapOf(
                            "string" to "text",
                            "number" to 42,
                            "boolean" to true,
                            "list" to listOf(1, 2, 3),
                            "map" to mapOf("nested" to "value"),
                        ),
                ),
            )

            val item = storage.getById("key-1")
            assertNotNull(item)
            assertEquals("text", item["string"])
            assertTrue(item["boolean"] as Boolean)
            assertEquals(3, (item["list"] as List<*>).size)
        }
    }

    /** Tests initialization creates working directory if it doesn't exist. */
    @Test
    fun `initialize creates working directory`() {
        runBlocking {
            val tempDir = File(tempFolder.newFolder(), "subdir/nested")
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            assertTrue(tempDir.exists())
            assertTrue(tempDir.isDirectory)
        }
    }

    /** Tests empty file doesn't cause errors. */
    @Test
    fun `initialize handles empty file`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val file = File(tempDir, "kv_store_test.json")
            file.writeText("")

            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            assertTrue(storage.isEmpty())
        }
    }

    /** Tests delete with empty list does nothing. */
    @Test
    fun `delete with empty list does nothing`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))

            storage.delete(emptyList())

            assertNotNull(storage.getById("key-1"))
        }
    }

    /** Tests upsert with empty map does nothing. */
    @Test
    fun `upsert with empty map does nothing`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(emptyMap())
            assertTrue(storage.isEmpty())
        }
    }

    /** Tests storage with complex nested structures. */
    @Test
    fun `handles complex nested structures`() {
        runBlocking {
            val tempDir = tempFolder.newFolder()
            val storage =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage.initialize()
            storage.upsert(
                mapOf(
                    "complex" to
                        mapOf(
                            "level1" to
                                mapOf(
                                    "level2" to
                                        mapOf(
                                            "level3" to listOf("a", "b", "c"),
                                        ),
                                ),
                        ),
                ),
            )

            storage.indexDoneCallback()

            // Reload to test persistence
            val storage2 =
                JsonKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    globalConfig = mapOf("working_dir" to tempDir.absolutePath),
                    embeddingFunc = embeddingModel,
                )

            storage2.initialize()
            val item = storage2.getById("complex")
            assertNotNull(item)

            @Suppress("UNCHECKED_CAST")
            val level1 = item["level1"] as Map<String, Any>
            @Suppress("UNCHECKED_CAST")
            val level2 = level1["level2"] as Map<String, Any>
            @Suppress("UNCHECKED_CAST")
            val level3 = level2["level3"] as List<String>
            assertEquals(listOf("a", "b", "c"), level3)
        }
    }
}
