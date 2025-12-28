package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.QueryParam
import lightrag.kg.neo4j.Neo4jDocStatusStorage
import lightrag.llm.LLMFactory
import java.io.File

/**
 * Example showing how to use Neo4jDocStatusStorage for doc status persistence.
 *
 * Prerequisites:
 *  - Neo4j reachable via NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD (or set in globalConfig["neo4j"])
 *  - OPENAI_API_KEY set
 */
fun main() =
    runBlocking {
        val apiKey = System.getenv("OPENAI_API_KEY")
        if (apiKey.isNullOrBlank()) {
            println("Error: OPENAI_API_KEY environment variable is not set.")
            return@runBlocking
        }

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

        val docStatusStorage =
            Neo4jDocStatusStorage(
                namespace = "doc_status",
                workspace = "default",
                globalConfig =
                    mapOf(
                        "neo4j" to
                            mapOf(
                                "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                                "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                                "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                            ),
                    ),
                embeddingFunc = embeddingModel,
            )
        docStatusStorage.initialize()

        val rag =
            LightRAG(
                chatModel = chatModel,
                embeddingModel = embeddingModel,
                docStatusStorageOverride = docStatusStorage,
                addonConfig =
                    AddonConfig(
                        overrides =
                            LightRagOverrides(
                                chunkTokenSize = 50,
                                chunkOverlapTokenSize = 2,
                            ),
                    ),
            )

        // Initialize other storages
        rag.fullDocs.initialize()
        rag.textChunks.initialize()
        rag.fullEntities.initialize()
        rag.fullRelations.initialize()
        rag.chunkEntityRelationGraph.initialize()
        rag.chunksVdb.initialize()
        rag.entitiesVdb.initialize()
        rag.relationshipsVdb.initialize()

        val bookFile = File("./book.txt")
        val content =
            if (bookFile.exists()) {
                bookFile.readText()
            } else {
                println("Warning: ./book.txt not found. Using dummy content.")
                "Neo4j is great for graphs; LightRAG can store doc status in Neo4j."
            }

        println("Inserting document (status tracked in Neo4j)...")
        rag.insert(content)

        println("Doc status snapshot (Neo4jDocStatusStorage):")
        println(docStatusStorage.getStatusCounts())

        val queryText = "How does LightRAG use Neo4j?"
        val modes = listOf("global", "local")
        modes.forEach { mode ->
            println("\n=== Query mode: $mode ===")
            val result =
                rag.query(
                    queryText,
                    QueryParam(
                        mode = mode,
                        includeReferences = true,
                        topK = 3,
                        chunkTopK = 2,
                    ),
                )
            println(result?.content ?: "No result")
        }

        println("Persisting storages...")
        docStatusStorage.indexDoneCallback()
        rag.fullDocs.indexDoneCallback()
        rag.textChunks.indexDoneCallback()
        rag.fullEntities.indexDoneCallback()
        rag.fullRelations.indexDoneCallback()
        rag.chunkEntityRelationGraph.indexDoneCallback()
        rag.chunksVdb.indexDoneCallback()
        rag.entitiesVdb.indexDoneCallback()
        rag.relationshipsVdb.indexDoneCallback()
        println("Done. Doc statuses persisted in Neo4j.")
    }
