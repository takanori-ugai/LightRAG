package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.appModule
import lightrag.di.mongodbGraphExampleModule
import lightrag.services.StorageManager
import org.koin.core.context.startKoin
import org.koin.java.KoinJavaComponent.get
import java.io.File

/**
 * The main function for the LightRAG OpenAI MongoDB Graph demo.
 * This function demonstrates how to use LightRAG with OpenAI models and a MongoDB-backed graph storage.
 * It initializes the models and storage, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule, mongodbGraphExampleModule)
        }

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        val workingDir = "./mongodb_test_dir"
        val dir = File(workingDir)
        if (!dir.exists()) {
            dir.mkdirs()
        }

        // Ensure Mongo env vars are set or default to localhost

        runBlocking {
            // Initialize storages
            storageManager.initialize()

            // Also kvStorage and others might need init if they were persistent
            // (JsonKVStorage loads from file in init block usually or constructor).

            val bookFile = File("book.txt")
            if (bookFile.exists()) {
                val content = bookFile.readText()
                rag.insert(content)
            } else {
                println("book.txt not found. Please place 'book.txt' in the working directory.")
                return@runBlocking
            }

            // Perform naive search
            println("Naive Search:")
            println(
                rag.query(
                    "What are the top themes in this story?",
                    QueryParam(mode = "naive"),
                )?.content,
            )

            // Perform local search
            println("\nLocal Search:")
            println(
                rag.query(
                    "What are the top themes in this story?",
                    QueryParam(mode = "local"),
                )?.content,
            )

            // Perform global search
            println("\nGlobal Search:")
            println(
                rag.query(
                    "What are the top themes in this story?",
                    QueryParam(mode = "global"),
                )?.content,
            )

            // Perform hybrid search
            println("\nHybrid Search:")
            println(
                rag.query(
                    "What are the top themes in this story?",
                    QueryParam(mode = "hybrid"),
                )?.content,
            )

            // Finalize (Close connection)
            storageManager.persist()
        }
    }
