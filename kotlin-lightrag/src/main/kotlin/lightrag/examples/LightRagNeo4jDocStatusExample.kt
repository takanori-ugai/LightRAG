package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.appModule
import lightrag.kg.neo4j.Neo4jDocStatusStorage
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get
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
        startKoin {
            allowOverride(true)
            modules(appModule)
        }

        // Override StorageManager to use Neo4jDocStatusStorage instead of the default JSON store.
        val neo4jDocStatusModule =
            module {
                single<StorageManager> {
                    val appConfig = get<AppConfig>()
                    val globalConfig = get<Map<String, Any?>>(named("globalConfig"))
                    val embeddingModel = appConfig.embeddingModel
                    StorageManager(
                        workingDir = appConfig.workingDir,
                        embeddingModel = embeddingModel,
                        graphStorageName = appConfig.graphStorageName,
                        vectorStorageName = appConfig.vectorStorageName,
                        addonConfig = appConfig.addonConfig,
                        globalConfig = globalConfig,
                        docStatusStorageOverride =
                            Neo4jDocStatusStorage(
                                namespace = "doc_status",
                                workspace = System.getenv("NEO4J_WORKSPACE") ?: "default",
                                globalConfig = globalConfig,
                                embeddingFunc = embeddingModel,
                            ),
                        fullDocsStorageOverride = appConfig.fullDocsStorageOverride,
                        textChunksStorageOverride = appConfig.textChunksStorageOverride,
                        fullEntitiesStorageOverride = appConfig.fullEntitiesStorageOverride,
                        fullRelationsStorageOverride = appConfig.fullRelationsStorageOverride,
                    )
                }
            }
        loadKoinModules(neo4jDocStatusModule)

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

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
        println(storageManager.docStatusStorage.getStatusCounts())

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
        storageManager.persist()
        println("Done. Doc statuses persisted in Neo4j.")
    }
