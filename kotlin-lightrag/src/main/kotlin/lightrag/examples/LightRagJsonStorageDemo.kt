package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.appModule
import lightrag.services.StorageManager
import org.koin.core.context.startKoin
import org.koin.java.KoinJavaComponent.get
import java.io.File

/**
 * The main function for the LightRAG JSON storage demo.
 * This function demonstrates how to use LightRAG with JSON-backed storages.
 * It initializes the storages, inserts a document, queries it using different modes, and persists the data.
 */
fun main() =
    runBlocking {
        startKoin {
            modules(appModule)
        }

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        // Configure API key, chat model, embedding model through Koin modules

        // Initialize storages and reload any persisted state
        storageManager.initialize()

        val bookFile = File("./book.txt")
        val content =
            if (bookFile.exists()) {
                bookFile.readText()
            } else {
                println("Warning: ./book.txt not found. Using dummy content.")
                "It was the best of times, it was the worst of times."
            }

        println("Inserting document into JSON storage...")
        rag.insert(content)

        println("\nDoc status snapshot (JsonDocStatusStorage):")
        println(storageManager.docStatusStorage.getStatusCounts())

        println("\nFull docs snapshot (JsonKVStorage):")
        println(storageManager.fullDocs.getByIds(listOf("0")))

        val modes = listOf("naive", "local", "global", "hybrid")
        val queryText = "What are the top themes in this story?"

        modes.forEach { mode ->
            println("\n=====================")
            println("Query mode: $mode")
            println("=====================")
            val result =
                rag.query(
                    queryText,
                    QueryParam(
                        mode = mode,
                        includeReferences = true,
                        topK = 5,
                        chunkTopK = 2,
                    ),
                )
            println(result?.content)
        }

        // Persist JSON-backed storages (KV, DocStatus, Vectors)
        storageManager.persist()

        println("\nDone! Data persisted using JSON-backed storages.")
    }
