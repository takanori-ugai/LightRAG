package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.kg.neo4j.Neo4jDocStatusStorage
import lightrag.kg.neo4j.Neo4jKVStorage
import lightrag.llm.LLMFactory

/**
 * Minimal example showing how to use Neo4jKVStorage for KV persistence.
 *
 * Prerequisites:
 *  - Neo4j running and accessible via NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD (or globalConfig["neo4j"])
 *  - OPENAI_API_KEY set for LLMs
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

        val neo4jConfig =
            mapOf(
                "neo4j" to
                    mapOf(
                        "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                        "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                        "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                    ),
            )

        // Use Neo4j-backed storages for KV and doc status
        val hashingKv =
            Neo4jKVStorage(
                namespace = "hash_cache",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            ).also { it.initialize() }
        val docStatusStorage =
            Neo4jDocStatusStorage(
                namespace = "doc_status",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            ).also { it.initialize() }
        val fullDocs =
            Neo4jKVStorage(
                namespace = "full_docs",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            ).also { it.initialize() }
        val textChunks =
            Neo4jKVStorage(
                namespace = "text_chunks",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            ).also { it.initialize() }
        val fullEntities =
            Neo4jKVStorage(
                namespace = "full_entities",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            ).also { it.initialize() }
        val fullRelations =
            Neo4jKVStorage(
                namespace = "full_relations",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            ).also { it.initialize() }

        val rag =
            LightRAG(
                chatModel = chatModel,
                embeddingModel = embeddingModel,
                hashingKv = hashingKv,
                docStatusStorageOverride = docStatusStorage,
                fullDocsStorageOverride = fullDocs,
                textChunksStorageOverride = textChunks,
                fullEntitiesStorageOverride = fullEntities,
                fullRelationsStorageOverride = fullRelations,
                addonConfig =
                    AddonConfig(
                        overrides =
                            LightRagOverrides(
                                chunkTokenSize = 50,
                                chunkOverlapTokenSize = 2,
                            ),
                    ),
            )

        // Initialize remaining storages to load any persisted state
        rag.chunkEntityRelationGraph.initialize()
        rag.chunksVdb.initialize()
        rag.entitiesVdb.initialize()
        rag.relationshipsVdb.initialize()

        val text =
            """
            Neo4j is a native graph database.
            LightRAG can use Neo4j for vectors, graph, or KV cache.
            """.trimIndent()

        println("Inserting text...")
        rag.insert(text)

        val queryText = "How does LightRAG use Neo4j?"
        val modes = listOf("global", "local")

        modes.forEach { mode ->
            println("\n=====================")
            println("Query mode: $mode")
            println("=====================")
            val result =
                rag.query(
                    queryText,
                    lightrag.core.QueryParam(
                        mode = mode,
                        includeReferences = true,
                        topK = 3,
                        chunkTopK = 3,
                    ),
                )
            println(result?.content ?: "No result")
        }

        println("Persisting storages...")
        hashingKv.indexDoneCallback()
        docStatusStorage.indexDoneCallback()
        fullDocs.indexDoneCallback()
        textChunks.indexDoneCallback()
        fullEntities.indexDoneCallback()
        fullRelations.indexDoneCallback()
        rag.chunkEntityRelationGraph.indexDoneCallback()
        rag.chunksVdb.indexDoneCallback()
        rag.entitiesVdb.indexDoneCallback()
        rag.relationshipsVdb.indexDoneCallback()
        println("Done. Hashing cache persisted in Neo4j and other stores saved locally.")
    }
