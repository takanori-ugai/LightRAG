package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.services.StorageManager
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module

/**
 * The main function for the LightRAG OpenAI demo.
 * This function demonstrates how to use LightRAG with OpenAI models.
 * It initializes the models, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        val koin =
            startKoin {
                allowOverride(true)
                modules(appModule, openAiDemoOverrideModule())
            }.koin
        val rag: LightRAG = koin.get()
        val storageManager: StorageManager = koin.get()

        prepareWorkingDir(
            "./dickens",
            filesToDelete =
                listOf(
                    "graph_chunk_entity_relation.graphml",
                    "kv_store_doc_status.json",
                    "kv_store_full_docs.json",
                    "kv_store_text_chunks.json",
                    "vdb_chunks.json",
                    "vdb_entities.json",
                    "vdb_relationships.json",
                ),
        )

        // Initialize storages (connects Neo4j and loads persisted data) before insert/query.
        storageManager.initialize()

        testEmbeddingModel(koin.get(), "This is a test string for embedding.")
        rag.insert(loadBookContent())
        runDemoQueries(rag, "What are the top themes related with King of England")
        println("\nDone!")
    }

private fun openAiDemoOverrideModule() =
    module {
        single<AppConfig> {
            val cfg = get<LightRagConfig>()
            val baseAddonConfig = addonConfigFrom(cfg)
            val addonConfig =
                baseAddonConfig.copy(
                    extras = baseAddonConfig.extras + mapOf("kg_chunk_pick_method" to "VECTOR"),
                )
            AppConfig(
                workingDir = "./dickens",
                graphStorageName = cfg.storage.graphStorageName,
                vectorStorageName = cfg.storage.vectorStorageName,
                addonConfig = addonConfig,
                chatModel = get(),
                embeddingModel = get(),
            )
        }

        single<Map<String, Any?>>(named("globalConfig")) {
            val appConfig = get<AppConfig>()
            globalConfigFrom(appConfig)
        }
    }
