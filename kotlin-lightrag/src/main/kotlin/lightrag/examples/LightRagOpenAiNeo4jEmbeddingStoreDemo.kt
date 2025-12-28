package lightrag.examples

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.Neo4jConfig
import lightrag.core.QueryParam
import java.time.Duration

/**
 * Demo showing LightRAG with Neo4jEmbeddingStoreVectorStorage (langchain4j community Neo4j embedding store).
 *
 * Requirements:
 * - Neo4j 5.15+ with vector index capability (defaults to bolt://localhost:7687 with neo4j/neo4j)
 * - OPENAI_API_KEY set
 */
fun main() =
    runBlocking {
        val apiKey = System.getenv("OPENAI_API_KEY")
        if (apiKey.isNullOrBlank()) {
            println("Error: OPENAI_API_KEY is not set.")
            return@runBlocking
        }

        // Neo4j settings (can override via env NEO4J_URI/USERNAME/PASSWORD/DATABASE)
        val neo4jUri = System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"
        val neo4jUser = System.getenv("NEO4J_USERNAME") ?: "neo4j"
        val neo4jPass = System.getenv("NEO4J_PASSWORD") ?: "neo4j"

        val chatModel =
            OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build()

        val embeddingModel =
            OpenAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-3-small")
                .build()

        val addonConfig =
            AddonConfig(
                neo4j =
                    Neo4jConfig(
                        uri = neo4jUri,
                        username = neo4jUser,
                        password = neo4jPass,
                    ),
                overrides =
                    LightRagOverrides(
                        chunkTokenSize = 256,
                        chunkOverlapTokenSize = 16,
                        cosineBetterThreshold = 0.2,
                    ),
            )

        val rag =
            LightRAG(
                workingDir = "./neo4j_embedding_store_demo",
                chatModel = chatModel,
                embeddingModel = embeddingModel,
                vectorStorageName = "Neo4jEmbeddingStoreVectorStorage",
                addonConfig = addonConfig,
            )

        println("Initializing Neo4j embedding store vector storage...")
        rag.chunksVdb.initialize()
        rag.entitiesVdb.initialize()
        rag.relationshipsVdb.initialize()

        // start clean for the demo
        rag.chunksVdb.drop()
        rag.entitiesVdb.drop()
        rag.relationshipsVdb.drop()

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

        rag.chunksVdb.finalize()
        rag.entitiesVdb.finalize()
        rag.relationshipsVdb.finalize()
    }
