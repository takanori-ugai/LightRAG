package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.appModule
import lightrag.di.openAiNeo4jVectorExampleModule
import lightrag.services.StorageManager
import org.koin.core.context.startKoin
import org.koin.java.KoinJavaComponent.get

/**
 * Demo showing LightRAG with both graph storage and vector storage backed by Neo4j.
 *
 * Requirements:
 * - Neo4j running (defaults to bolt://localhost:7687 with neo4j/neo4j)
 * - OPENAI_API_KEY set
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule, openAiNeo4jVectorExampleModule)
        }

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        val apiKey = System.getenv("OPENAI_API_KEY")
        if (apiKey.isNullOrBlank()) {
            println("Error: OPENAI_API_KEY is not set.")
            return@runBlocking
        }

        println("Initializing Neo4j vector/graph storage...")
        storageManager.initialize()

        val content =
            """
            The capital of France is Paris. The Eiffel Tower is a landmark in Paris.
            The Louvre Museum houses famous artworks like the Mona Lisa.
            """.trimIndent()

        println("Inserting content...")
        rag.insert(content)

        val queryText = "What are key attractions in the capital of France?"
        val modes = listOf("naive", "local", "global")

        modes.forEach { mode ->
            println("\n=== Query mode: $mode ===")
            val result =
                rag.query(
                    queryText,
                    QueryParam(
                        mode = mode,
                        includeReferences = true,
                        topK = 3,
                        chunkTopK = 3,
                    ),
                )
            println(result?.content ?: "No result")
        }

        storageManager.persist()
    }
