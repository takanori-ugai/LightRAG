package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.llm.LLMFactory
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.dsl.module
import java.io.File

/**
 * The main function for the LightRAG OpenAI Neo4j demo.
 * This function demonstrates how to use LightRAG with OpenAI models and a Neo4j-backed graph storage.
 * It initializes the models and storage, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val koin = startKoin {
            allowOverride(true)
            modules(appModule)
        }.koin

        // Enable HTTP request/response logging for the OpenAI chat model in this demo.
        val loggingModule =
            module {
                single<dev.langchain4j.model.chat.ChatModel> {
                    val cfg = koin.get<LightRagConfig>()
                    LLMFactory.createChatModel(
                        binding = "openai",
                        modelName = cfg.openai.chatModelName,
                        apiKey = cfg.openai.apiKey,
                        logRequests = true,
                        logResponses = true,
                    )
                }
            }
        loadKoinModules(loggingModule)

        val rag: LightRAG = koin.get<LightRAG>()
        val storageManager: StorageManager = koin.get<StorageManager>()

        // All configurations are loaded via Koin from application.conf
        // Initialize Neo4j Storage (Create indexes etc.)
        println("Initializing Neo4j Graph Storage...")
        try {
            storageManager.initialize()
            storageManager.drop()
        } catch (e: Exception) {
            println("Error initializing Neo4j storage: ${e.message}")
            println("Please ensure Neo4j is running and configured correctly in application.conf")
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
        storageManager.persist()
        println("\nDone!")
    }
