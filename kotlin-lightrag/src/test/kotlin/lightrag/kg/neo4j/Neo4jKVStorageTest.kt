package lightrag.kg.neo4j

import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/** Tests covering Neo4j-backed KV storage read behaviors used by query context assembly. */
class Neo4jKVStorageTest {
    private val embeddingModel = TestEmbeddings.mockEmbeddingModel()

    /** Ensures read APIs include the key as `id` to preserve chunk uniqueness. */
    @Test
    fun `getById and getByIds include id field`() {
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
            assertEquals("chunk-1", single["id"])

            val results = storage.getByIds(listOf("chunk-1", "chunk-2"))
            val ids = results.mapNotNull { it["id"] as? String }.toSet()
            assertEquals(setOf("chunk-1", "chunk-2"), ids)
            assertEquals(2, results.distinctBy { it["id"] }.size)
        }
    }
}
