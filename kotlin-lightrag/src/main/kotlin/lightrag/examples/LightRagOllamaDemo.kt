package lightrag.examples

import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.QueryParam
import java.io.File

fun main() =
    runBlocking {
        val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: "http://localhost:11434"
        val workingDir = "./ollama-demo"
        val workingDirFile = File(workingDir)
        if (!workingDirFile.exists()) {
            workingDirFile.mkdirs()
        }

        // Clear old data files (mirrors LightRagOpenAiDemo)
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

        // Initialize models served by Ollama
        val chatModel =
            OllamaChatModel.builder()
                .modelName("llama3")
                .baseUrl(baseUrl)
                .temperature(0.0)
                .build()

        val embeddingModel =
            OllamaEmbeddingModel.builder()
                .modelName("all-minilm")
                .baseUrl(baseUrl)
                .build()

        // Initialize RAG
        val rag =
            LightRAG(
                workingDir = workingDir,
                chatModel = chatModel,
                embeddingModel = embeddingModel,
                addonConfig =
                    AddonConfig(
                        overrides =
                            LightRagOverrides(
                                chunkTokenSize = 50,
                                chunkOverlapTokenSize = 2,
                            ),
                    ),
            )

        // Test embedding function
        val testText = "This is a test string for embedding."
        @Suppress("TooGenericExceptionCaught")
        try {
            val embeddingResponse = embeddingModel.embed(testText)
            val embedding = embeddingResponse.content()
            val embeddingDim = embedding.dimension()
            println("\n=======================")
            println("Test embedding function")
            println("========================")
            println("Test text: $testText")
            println("Detected embedding dimension: $embeddingDim\n\n")
        } catch (e: Exception) {
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
            @Suppress("TooGenericExceptionCaught")
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
            } catch (e: Exception) {
                println("Error querying mode $mode: ${e.message}")
            }
        }

        println("\nDone!")
    }
