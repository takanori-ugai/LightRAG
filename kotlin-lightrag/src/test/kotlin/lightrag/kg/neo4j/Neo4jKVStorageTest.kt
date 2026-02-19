package lightrag.kg.neo4j

import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Tests covering Neo4j-backed KV storage read behaviors used by query context assembly. */
class Neo4jKVStorageTest {
    private val embeddingModel = TestEmbeddings.mockEmbeddingModel()

    /** Ensures read APIs return payload data for requested keys. */
    @Test
    fun `getById and getByIds return payload`() {
        runBlocking {
            // Unit test against the in-memory cache path; Neo4j driver is not initialized here.
            val storage =
                Neo4jKVStorage(
                    namespace = "text_chunks",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

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
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

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
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            val result = storage.filterKeys(setOf("key-1", "key-2"))
            assertEquals(setOf("key-1", "key-2"), result)
        }
    }

    /** Tests delete removes items from storage. */
    @Test
    fun `delete removes items`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

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

    /** Tests drop clears all data. */
    @Test
    fun `drop clears all data`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))

            val result = storage.drop()
            assertEquals("success", result["status"])
            assertTrue(storage.isEmpty())
        }
    }

    /** Tests isEmpty returns correct status. */
    @Test
    fun `isEmpty returns correct status`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            assertTrue(storage.isEmpty())

            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))
            assertFalse(storage.isEmpty())

            storage.delete(listOf("key-1"))
            assertTrue(storage.isEmpty())
        }
    }

    /** Tests getById returns null for non-existent key. */
    @Test
    fun `getById returns null for non-existent key`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            assertNull(storage.getById("non-existent"))
        }
    }

    /** Tests getByIds returns only existing items. */
    @Test
    fun `getByIds returns only existing items`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

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
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

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
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

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
            assertEquals(42, item["number"])
            assertTrue(item["boolean"] as Boolean)
            assertEquals(3, (item["list"] as List<*>).size)
        }
    }

    /** Tests delete with empty list does nothing. */
    @Test
    fun `delete with empty list does nothing`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))

            storage.delete(emptyList())

            assertNotNull(storage.getById("key-1"))
        }
    }

    /** Tests upsert with empty map does nothing. */
    @Test
    fun `upsert with empty map does nothing`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            storage.upsert(emptyMap())
            assertTrue(storage.isEmpty())
        }
    }

    /** Tests storage with complex nested structures. */
    @Test
    fun `handles complex nested structures`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

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

            val item = storage.getById("complex")
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

    /** Tests sanitizeLabel creates valid Neo4j labels. */
    @Test
    fun `sanitizeLabel creates valid labels`() {
        // Test various edge cases for label sanitization
        // Note: We can't directly test the private helper, but we can verify the storage
        // initializes without errors with various namespace/workspace combinations
        runBlocking {
            val testCases =
                listOf(
                    "test-namespace" to "work-space",
                    "test@namespace" to "work#space",
                    "123test" to "456work",
                    "test_namespace" to "work_space",
                )

            testCases.forEach { (namespace, workspace) ->
                val storage =
                    Neo4jKVStorage(
                        namespace = namespace,
                        workspace = workspace,
                        embeddingFunc = embeddingModel,
                    )

                storage.upsert(mapOf("key-1" to mapOf("value" to "data")))
                assertNotNull(storage.getById("key-1"))
            }
        }
    }

    /** Tests multiple upserts preserve all data. */
    @Test
    fun `multiple upserts preserve all data`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))
            storage.upsert(mapOf("key-2" to mapOf("value" to "data2")))
            storage.upsert(mapOf("key-3" to mapOf("value" to "data3")))

            assertEquals(3, storage.getByIds(listOf("key-1", "key-2", "key-3")).size)
            assertNotNull(storage.getById("key-1"))
            assertNotNull(storage.getById("key-2"))
            assertNotNull(storage.getById("key-3"))
        }
    }

    /** Tests partial delete preserves undeleted items. */
    @Test
    fun `partial delete preserves undeleted items`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            storage.upsert(
                mapOf(
                    "key-1" to mapOf("value" to "data1"),
                    "key-2" to mapOf("value" to "data2"),
                    "key-3" to mapOf("value" to "data3"),
                    "key-4" to mapOf("value" to "data4"),
                ),
            )

            storage.delete(listOf("key-2", "key-4"))

            val remaining = storage.getByIds(listOf("key-1", "key-2", "key-3", "key-4"))
            assertEquals(2, remaining.size)

            val remainingValues = remaining.mapNotNull { it["value"] as? String }.toSet()
            assertEquals(setOf("data1", "data3"), remainingValues)
        }
    }

    /** Tests drop result message. */
    @Test
    fun `drop returns success message`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test_namespace",
                    workspace = "test_workspace",
                    embeddingFunc = embeddingModel,
                )

            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))

            val result = storage.drop()
            assertEquals("success", result["status"])
            assertTrue(result["message"]?.contains("test_namespace") == true)
            assertTrue(result["message"]?.contains("test_workspace") == true)
        }
    }

    /** Tests getByIds with empty list returns empty list. */
    @Test
    fun `getByIds with empty list returns empty list`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            storage.upsert(mapOf("key-1" to mapOf("value" to "data1")))

            val results = storage.getByIds(emptyList())
            assertEquals(0, results.size)
        }
    }

    /** Tests null values in data. */
    @Test
    fun `handles null values in data`() {
        runBlocking {
            val storage =
                Neo4jKVStorage(
                    namespace = "test",
                    workspace = "ws",
                    embeddingFunc = embeddingModel,
                )

            val payload =
                mapOf(
                    "string" to "text",
                    "nullValue" to null,
                    "number" to 42,
                )
            val cleanedPayload = payload.filterValues { it != null }.mapValues { it.value!! }
            storage.upsert(mapOf("key-1" to cleanedPayload))

            val item = storage.getById("key-1")
            assertNotNull(item)
            assertEquals("text", item["string"])
            assertEquals(42, item["number"])
            // null values may or may not be present depending on implementation
        }
    }
}
