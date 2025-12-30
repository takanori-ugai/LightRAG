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
        loadKoinModules(neo4jDocStatusModule())

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        println("Inserting document (status tracked in Neo4j)...")
        rag.insert(loadBookContent())

        println("Doc status snapshot (Neo4jDocStatusStorage):")
        println(storageManager.docStatusStorage.getStatusCounts())

        runDemoQueries(
            rag,
            "How does LightRAG use Neo4j?",
            modes = listOf("global", "local"),
        ) { mode ->
            QueryParam(
                mode = mode,
                includeReferences = true,
                topK = 3,
                chunkTopK = 2,
            )
        }

        storageManager.persist()
        println("Done. Doc statuses persisted in Neo4j.")
    }

private fun neo4jDocStatusModule() =
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
