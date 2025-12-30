package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get

/**
 * Demo showing LightRAG with Neo4jEmbeddingStoreVectorStorage (langchain4j community Neo4j embedding store).
 *
 * Requirements:
 * - Neo4j 5.15+ with vector index capability (defaults to bolt://localhost:7687 with neo4j/neo4j)
 * - OPENAI_API_KEY set
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule)
        }

        // Force vector storage to Neo4jEmbeddingStoreVectorStorage.
        val neo4jEmbeddingStoreModule =
            module {
                single<AppConfig> {
                    val cfg = get<LightRagConfig>()
                    AppConfig(
                        workingDir = cfg.storage.workingDir,
                        graphStorageName = cfg.storage.graphStorageName,
                        vectorStorageName = "Neo4jEmbeddingStoreVectorStorage",
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
        loadKoinModules(neo4jEmbeddingStoreModule)

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        println("Initializing Neo4j embedding store vector storage...")
        storageManager.initialize()

        // start clean for the demo
        storageManager.drop()

        val content =
            """
            The capital of France is Paris. The Eiffel Tower is a landmark in Paris.
            The Louvre Museum houses famous artworks like the Mona Lisa.
            """.trimIndent()

        println("Inserting content...")
        rag.insert(content)

        val queryText = "What are key attractions in the capital of France?"
        val modes = listOf("naive", "local", "global")

        modes.forEach { mode ->
            println("\n=== Query mode: $mode ===")
            val result =
                rag.query(
                    queryText,
                    QueryParam(
                        mode = mode,
                        includeReferences = true,
                        topK = 3,
                        chunkTopK = 3,
                    ),
                )
            println(result?.content ?: "No result")
        }

        storageManager.persist()
    }
