package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.services.StorageManager
import lightrag.utils.loadCausalCsv
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get
import java.nio.file.Path

fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule, csvDemoOverrideModule())
        }

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        storageManager.initialize()

        val firstRow = loadCausalCsv(Path.of("data.csv")).firstOrNull()
        if (firstRow == null) {
            println("No rows found in data.csv.")
            return@runBlocking
        }

        println("Inserting causal text into LightRAG...")
        rag.insert(firstRow.text)
        rag.rebuildDerivedStorageIfEmpty()

        val queryText = "What is the cause of ${firstRow.result}? Answer in one word or a few words. Strictly answer in given context. Don't use your knowledge."
        println("Query: $queryText")

        val result = rag.query(queryText, QueryParam(mode = "hybrid", topK = 5, chunkTopK = 5))
        println(result?.content ?: "(no result)")
    }

private fun csvDemoOverrideModule() =
    module {
        single<AppConfig> {
            val cfg = get<LightRagConfig>()
            AppConfig(
                workingDir = "./csv_demo_storage",
                graphStorageName = "InMemoryGraphStorage",
                vectorStorageName = "InMemoryVectorStorage",
                addonConfig = addonConfigFrom(cfg),
                chatModel = get(),
                embeddingModel = get(),
            )
        }

        single<Map<String, Any?>>(named("globalConfig")) {
            val appConfig = get<AppConfig>()
            globalConfigFrom(appConfig)
        }
    }
