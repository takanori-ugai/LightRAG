package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.llm.LLMFactory
import java.io.File

fun main() =
    runBlocking {
        val apiKey = System.getenv("OPENAI_API_KEY")
        if (apiKey.isNullOrBlank()) {
            println("Error: OPENAI_API_KEY environment variable is not set.")
            return@runBlocking
        }

        val workingDir = "./json_demo_storage"
        val chatModel =
            LLMFactory.createChatModel(
                binding = "openai",
                modelName = "gpt-4o-mini",
                apiKey = apiKey,
            )
        val embeddingModel =
            LLMFactory.createEmbeddingModel(
                binding = "openai",
                modelName = "text-embedding-3-small",
                apiKey = apiKey,
            )

        val rag =
            LightRAG(
                workingDir = workingDir,
                chatModel = chatModel,
                embeddingModel = embeddingModel,
            )

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
        println(rag.docStatusStorage.getStatusCounts())

        println("\nFull docs snapshot (JsonKVStorage):")
        println(rag.fullDocs.getByIds(listOf("0")))

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
                    ),
                )
            println(result?.content)
        }

        // Persist JSON-backed storages (KV, DocStatus, Vectors)
        rag.docStatusStorage.indexDoneCallback()
        rag.fullDocs.indexDoneCallback()
        rag.textChunks.indexDoneCallback()
        rag.fullEntities.indexDoneCallback()
        rag.fullRelations.indexDoneCallback()
        rag.chunksVdb.indexDoneCallback()
        rag.entitiesVdb.indexDoneCallback()
        rag.relationshipsVdb.indexDoneCallback()

        println("\nDone! Data persisted under $workingDir using JSON-backed storages.")
    }
