package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.core.LightRagOverrides
import lightrag.core.AddonConfig
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
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
            modules(appModule)
        }

        // Override AppConfig/globalConfig to use MongoDB graph storage.
        val mongoOverrideModule =
            module {
                single<AppConfig> {
                    val cfg = get<LightRagConfig>()
                    AppConfig(
                        workingDir = cfg.storage.workingDir,
                        graphStorageName = "MongoGraphStorage",
                        vectorStorageName = cfg.storage.vectorStorageName,
                        addonConfig =
                            AddonConfig(
                                overrides =
                                    LightRagOverrides(
                                        chunkTokenSize = cfg.addonConfig.chunkTokenSize,
                                        chunkOverlapTokenSize = cfg.addonConfig.chunkOverlapTokenSize,
                                        entityTypes = cfg.addonConfig.entityTypes,
                                        language = cfg.addonConfig.language,
                                        cosineBetterThreshold = cfg.addonConfig.cosineBetterThreshold,
                                    ),
                                cosineBetterThreshold = cfg.addonConfig.cosineBetterThreshold,
                            ),
                        llmBinding = "openai",
                        llmModelName = cfg.openai.chatModelName,
                        embeddingBinding = "openai",
                        embeddingModelName = cfg.openai.embeddingModelName,
                        chatModel = get(),
                        embeddingModel = get(),
                    )
                }

                // Refresh globalConfig to reflect the updated AppConfig.
                single<Map<String, Any?>>(named("globalConfig")) {
                    val appConfig = get<AppConfig>()
                    val overrides = appConfig.addonConfig.overrides
                    val chunkTokenSize = overrides.chunkTokenSize ?: 1200
                    val chunkOverlapTokenSize = overrides.chunkOverlapTokenSize ?: 100
                    val entityTypes =
                        overrides.entityTypes ?: listOf("Person", "Organization", "Location", "Event", "Concept")
                    val language = overrides.language ?: "English"
                    mapOf(
                        "llm_model_func" to appConfig.chatModel,
                        "embedding_func" to appConfig.embeddingModel,
                        "chunk_token_size" to chunkTokenSize,
                        "chunk_overlap_token_size" to chunkOverlapTokenSize,
                        "entity_types" to entityTypes,
                        "language" to language,
                        "working_dir" to appConfig.workingDir,
                        "enable_llm_cache" to (appConfig.hashingKv != null),
                    ) + appConfig.addonConfig.toMap()
                }
            }
        loadKoinModules(mongoOverrideModule)

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
                rag
                    .query(
                        "What are the top themes in this story?",
                        QueryParam(mode = "naive"),
                    )?.content,
            )

            // Perform local search
            println("\nLocal Search:")
            println(
                rag
                    .query(
                        "What are the top themes in this story?",
                        QueryParam(mode = "local"),
                    )?.content,
            )

            // Perform global search
            println("\nGlobal Search:")
            println(
                rag
                    .query(
                        "What are the top themes in this story?",
                        QueryParam(mode = "global"),
                    )?.content,
            )

            // Perform hybrid search
            println("\nHybrid Search:")
            println(
                rag
                    .query(
                        "What are the top themes in this story?",
                        QueryParam(mode = "hybrid"),
                    )?.content,
            )

            // Finalize (Close connection)
            storageManager.persist()
        }
    }
