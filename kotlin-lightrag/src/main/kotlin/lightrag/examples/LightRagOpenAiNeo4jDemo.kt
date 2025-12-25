package lightrag.examples

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.Neo4jConfig
import lightrag.core.QueryParam
import java.io.File
import java.net.URI
import java.time.Duration

fun main() =
    runBlocking {
        // Check OpenAI environment variable
        val apiKey = System.getenv("OPENAI_API_KEY")
        if (apiKey.isNullOrBlank()) {
            println("Error: OPENAI_API_KEY environment variable is not set.")
            return@runBlocking
        }

        // Neo4j Configuration
        // User requested default: http://localhost:7474
        val defaultUri = "http://localhost:7474"
        val defaultUser = "neo4j"
        val defaultPass = "neo4j"

        var neo4jUri = System.getenv("NEO4J_URI") ?: defaultUri
        val neo4jUser = System.getenv("NEO4J_USERNAME") ?: defaultUser
        val neo4jPass = System.getenv("NEO4J_PASSWORD") ?: defaultPass

        println("Using Neo4j configuration:")
        println("  Input URI: $neo4jUri")
        println("  User: $neo4jUser")

        // Sanitize URI for Neo4j Driver (requires bolt/neo4j scheme, not http)
        if (neo4jUri.startsWith("http")) {
            println("Warning: The Neo4j Driver requires a binary protocol (bolt/neo4j), but an HTTP URI was provided.")
            try {
                val uriObj = URI(neo4jUri)
                val host = uriObj.host ?: "localhost"
                // If port is standard HTTP console (7474), switch to standard Bolt (7687)
                val port = if (uriObj.port == 7474) 7687 else uriObj.port
                val newUri = "bolt://$host:$port"
                println("Converting '$neo4jUri' to '$newUri' for driver connection.")
                neo4jUri = newUri
            } catch (e: Exception) {
                println("Error parsing URI '$neo4jUri': ${e.message}. Falling back to 'bolt://localhost:7687'.")
                neo4jUri = "bolt://localhost:7687"
            }
        }

        val workingDir = "./neo4j_test_dir"
        val dir = File(workingDir)
        if (!dir.exists()) {
            dir.mkdirs()
        }

        // Configure OpenAI Models
        val chatModel =
            OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build()

        val embeddingModel =
            OpenAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-3-large")
                .dimensions(3072)
                .build()

        // Pass configuration to LightRAG via addonConfig
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
                        chunkTokenSize = 50,
                        chunkOverlapTokenSize = 2,
                    ),
            )

        val rag =
            LightRAG(
                workingDir = workingDir,
                chatModel = chatModel,
                embeddingModel = embeddingModel,
                graphStorageName = "Neo4jGraphStorage",
                addonConfig = addonConfig,
            )

        // Initialize Neo4j Storage (Create indexes etc.)
        println("Initializing Neo4j Graph Storage...")
        try {
            rag.chunkEntityRelationGraph.initialize()
        } catch (e: Exception) {
            println("Error initializing Neo4j storage: ${e.message}")
            println("Please ensure Neo4j is running at $neo4jUri")
            return@runBlocking
        }

        // Prepare content
        val bookFile = File("book.txt")
        val content =
            if (bookFile.exists()) {
                bookFile.readText()
            } else {
                println("Warning: book.txt not found. Using dummy content.")
                """
                Neo4j is a graph database management system developed by Neo4j, Inc.
                It is an ACID-compliant transactional database with native graph storage and processing.
                LightRAG is a retrieval-augmented generation system that can use Neo4j as a backend.
                Kotlin is a cross-platform, statically typed, general-purpose programming language with type inference.
                """.trimIndent()
            }

        // Insert
        println("Inserting content...")
        try {
            rag.insert(content)
        } catch (e: Exception) {
            println("Error inserting content: ${e.message}")
        }

        // Query
        val modes = listOf("naive", "local", "global", "hybrid")
        val queryText = "What are the top themes in this story?"

        modes.forEach { mode ->
            println("\n=====================")
            println("Query mode: $mode")
            println("=====================")
            try {
                val result =
                    rag.query(
                        queryText,
                        QueryParam(
                            mode = mode,
                            includeReferences = true,
                            topK = 2,
                            chunkTopK = 2,
                        ),
                    )
                println(result?.content)
            } catch (e: Exception) {
                println("Error querying mode $mode: ${e.message}")
            }
        }

        // Finalize
        rag.chunkEntityRelationGraph.finalize()
        println("\nDone!")
    }
