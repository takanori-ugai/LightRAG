package lightrag.kg.json

import kotlinx.coroutines.runBlocking
import lightrag.TestEmbeddings
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/** Tests covering JSON-backed KV storage behaviors needed by query context assembly. */
class JsonKVStorageTest {
    private val embeddingModel = TestEmbeddings.mockEmbeddingModel()

    /** Ensures read APIs include the key as `id` to preserve chunk uniqueness. */
    @Test
    fun `getById and getByIds include id field`() {
        runBlocking {
            val tempDir = File("build/tmp/json_kv_ids_${System.currentTimeMillis()}").apply { mkdirs() }
            try {
                val storage =
                    JsonKVStorage(
                        namespace = "text_chunks",
                        workspace = "ws",
                        globalConfig = mapOf("working_dir" to tempDir.absolutePath),
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
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }
}
