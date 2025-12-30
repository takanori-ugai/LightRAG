package lightrag.examples

import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.appModule
import org.koin.core.context.startKoin
import java.io.File

/**
 * The main function for the LightRAG OpenAI demo.
 * This function demonstrates how to use LightRAG with OpenAI models.
 * It initializes the models, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val koin =
            startKoin {
                modules(appModule)
            }.koin

        val rag: LightRAG = koin.get<LightRAG>()

        val workingDir = "./dickens"
        val workingDirFile = File(workingDir)
        if (!workingDirFile.exists()) {
            workingDirFile.mkdirs()
        }

        // Clear old data files
        val filesToDelete =
            listOf(
                "graph_chunk_entity_relation.graphml",
                "kv_store_doc_status.json",
                "kv_store_full_docs.json",
                "kv_store_text_chunks.json",
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
            )

        filesToDelete.forEach { fileName ->
            val file = File(workingDirFile, fileName)
            if (file.exists()) {
                file.delete()
                println("Deleting old file: ${file.absolutePath}")
            }
        }

        // Test embedding function
        val embeddingModel: EmbeddingModel = koin.get()
        val testText = "This is a test string for embedding."
        try {
            val embeddingResponse = embeddingModel.embed(testText)
            val embedding = embeddingResponse.content()
            val embeddingDim = embedding.dimension()
            println("\n=======================")
            println("Test embedding function")
            println("========================")
            println("Test text: $testText")
            println("Detected embedding dimension: $embeddingDim\n\n")
        } catch (e: IllegalStateException) {
            println("Error testing embedding: ${e.message}")
        } catch (e: IllegalArgumentException) {
            println("Error testing embedding: ${e.message}")
        }

        // Read book.txt
        val bookFile = File("./book.txt")
        val bookContent =
            if (bookFile.exists()) {
                bookFile.readText()
            } else {
                println("Warning: ./book.txt not found. Using dummy content.")
                "This is a story about a developer converting Python code to Kotlin. " +
                    "It was a long and arduous journey, " +
                    "but eventually, the code compiled and ran successfully. " +
                    "The themes involve persistence, programming languages, and AI assistants."
            }

        rag.insert(bookContent)

        // Perform queries
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
                            topK = 5,
                            chunkTopK = 2,
                        ),
                    )
                println(result?.content)
            } catch (e: IllegalStateException) {
                println("Error querying mode $mode: ${e.message}")
            } catch (e: IllegalArgumentException) {
                println("Error querying mode $mode: ${e.message}")
            }
        }

        println("\nDone!")
    }
