package lightrag.examples

import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.llm.LLMFactory
import lightrag.services.StorageManager
import org.koin.core.context.loadKoinModules
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get
import java.io.File

/**
 * The main function for the LightRAG Ollama demo.
 * This function demonstrates how to use LightRAG with Ollama-hosted models.
 * It initializes the models, inserts a document, and queries it using different modes.
 */
fun main() =
    runBlocking {
        startKoin {
            allowOverride(true)
            modules(appModule)
        }

        // Override ChatModel/EmbeddingModel/AppConfig to use Ollama instead of OpenAI.
        val ollamaModule =
            module {
                single<dev.langchain4j.model.chat.ChatModel> {
                    val cfg = get<LightRagConfig>()
                    LLMFactory.createChatModel(
                        binding = "ollama",
                        modelName = cfg.ollama.chatModelName,
                        baseUrl = cfg.ollama.baseUrl,
                    )
                }

                single<EmbeddingModel> {
                    val cfg = get<LightRagConfig>()
                    LLMFactory.createEmbeddingModel(
                        binding = "ollama",
                        modelName = cfg.ollama.embeddingModelName,
                        baseUrl = cfg.ollama.baseUrl,
                    )
                }

                single<AppConfig> {
                    val cfg = get<LightRagConfig>()
                    AppConfig(
                        workingDir = cfg.storage.workingDir,
                        graphStorageName = cfg.storage.graphStorageName,
                        vectorStorageName = cfg.storage.vectorStorageName,
                        addonConfig =
                            lightrag.core.AddonConfig(
                                overrides =
                                    lightrag.core.LightRagOverrides(
                                        chunkTokenSize = cfg.addonConfig.chunkTokenSize,
                                        chunkOverlapTokenSize = cfg.addonConfig.chunkOverlapTokenSize,
                                        entityTypes = cfg.addonConfig.entityTypes,
                                        language = cfg.addonConfig.language,
                                        cosineBetterThreshold = cfg.addonConfig.cosineBetterThreshold,
                                    ),
                                cosineBetterThreshold = cfg.addonConfig.cosineBetterThreshold,
                            ),
                        llmBinding = "ollama",
                        llmModelName = cfg.ollama.chatModelName,
                        embeddingBinding = "ollama",
                        embeddingModelName = cfg.ollama.embeddingModelName,
                        chatModel = get(),
                        embeddingModel = get(),
                    )
                }

                // Refresh globalConfig after overriding AppConfig/chat/embedding.
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
        loadKoinModules(ollamaModule)

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

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

        // Test embedding function - get embedding model from Koin
        val embeddingModel: EmbeddingModel = get(EmbeddingModel::class.java)
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
