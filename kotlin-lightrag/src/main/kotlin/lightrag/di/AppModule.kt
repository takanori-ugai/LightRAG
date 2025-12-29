package lightrag.di

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.EncodingType
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.Neo4jConfig // Import Neo4jConfig
import lightrag.core.types.BaseKVStorage // Import BaseKVStorage
import lightrag.kg.neo4j.Neo4jDocStatusStorage
import lightrag.kg.neo4j.Neo4jKVStorage
import lightrag.llm.LLMFactory
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import org.koin.core.qualifier.named
import org.koin.dsl.module
import java.net.URI // Import URI
import java.time.Duration // Import Duration

val appModule =
    module {

        single { io.github.oshai.kotlinlogging.KotlinLogging.logger {} }

        single<ChatModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            LLMFactory.createChatModel(
                binding = "openai",
                modelName = "gpt-4o-mini",
                apiKey = apiKey,
            )
        }

        single<EmbeddingModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            LLMFactory.createEmbeddingModel(
                binding = "openai",
                modelName = "text-embedding-3-small",
                apiKey = apiKey,
            )
        }

        single { AppConfig(chatModel = get(), embeddingModel = get()) }

        single { Encodings.newDefaultEncodingRegistry() }
        single { get<EncodingRegistry>().getEncoding(EncodingType.CL100K_BASE) }

        single(named("tokenizer")) {
            val enc = get<Encoding>()
            val tokenizer: (String) -> List<Int> = { text: String ->
                val intArrayList = enc.encode(text)
                val list = mutableListOf<Int>()
                for (i in 0 until intArrayList.size()) {
                    list.add(intArrayList.get(i))
                }
                list
            }
            tokenizer
        }

        single(named("decoder")) {
            val enc = get<Encoding>()
            val decoder: (List<Int>) -> String = { list ->
                val intArrayList = com.knuddels.jtokkit.api.IntArrayList()
                list.forEach { intArrayList.add(it) }
                enc.decode(intArrayList)
            }
            decoder
        }

        single(named("globalConfig")) {
            val config = get<AppConfig>()
            mapOf(
                "llm_model_func" to config.chatModel,
                "embedding_func" to config.embeddingModel,
                "chunk_token_size" to 1200,
                "chunk_overlap_token_size" to 100,
                "entity_types" to listOf("Person", "Organization", "Location", "Event", "Concept"),
                "language" to "English",
                "working_dir" to config.workingDir,
                "enable_llm_cache" to (config.hashingKv != null),
            ) + config.addonConfig.toMap()
        }

        single {
            val config = get<AppConfig>()
            StorageManager(
                workingDir = config.workingDir,
                embeddingModel = config.embeddingModel,
                graphStorageName = config.graphStorageName,
                vectorStorageName = config.vectorStorageName,
                addonConfig = config.addonConfig,
                globalConfig = get(named("globalConfig")),
                docStatusStorageOverride = config.docStatusStorageOverride,
                fullDocsStorageOverride = config.fullDocsStorageOverride,
                textChunksStorageOverride = config.textChunksStorageOverride,
                fullEntitiesStorageOverride = config.fullEntitiesStorageOverride,
                fullRelationsStorageOverride = config.fullRelationsStorageOverride,
            )
        }

        single {
            IngestionService(
                storageManager = get(),
                globalConfig = get(named("globalConfig")),
                tokenizer = get(named("tokenizer")),
                decoder = get(named("decoder")),
            )
        }

        single {
            val config = get<AppConfig>()
            QueryService(
                storageManager = get(),
                chatModel = get(),
                hashingKv = config.hashingKv,
                globalConfig = get(named("globalConfig")),
                tokenizer = get(named("tokenizer")),
                decoder = get(named("decoder")),
            )
        }

        single {
            LightRAG(
                ingestionService = get(),
                queryService = get(),
                storageManager = get(),
            )
        }
    }

val neo4jDocStatusExampleModule =
    module {
        single {
            val embeddingModel = get<EmbeddingModel>()
            val neo4jConfig =
                mapOf(
                    "neo4j" to
                        mapOf(
                            "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                            "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                            "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                        ),
                )
            Neo4jDocStatusStorage(
                namespace = "doc_status",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            )
        }

        single {
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                docStatusStorageOverride = get(),
            )
        }
    }

val neo4jKVExampleModule =
    module {
        single<BaseKVStorage>(named("hashingKv")) {
            val embeddingModel = get<EmbeddingModel>()
            val neo4jConfig =
                mapOf(
                    "neo4j" to
                        mapOf(
                            "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                            "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                            "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                        ),
                )
            Neo4jKVStorage(
                namespace = "hash_cache",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            )
        }

        single<Neo4jDocStatusStorage> { // This will be provided by neo4jKVExampleModule
            val embeddingModel = get<EmbeddingModel>()
            val neo4jConfig =
                mapOf(
                    "neo4j" to
                        mapOf(
                            "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                            "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                            "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                        ),
                )
            Neo4jDocStatusStorage(
                namespace = "doc_status",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            )
        }

        single(named("fullDocs")) {
            val embeddingModel = get<EmbeddingModel>()
            val neo4jConfig =
                mapOf(
                    "neo4j" to
                        mapOf(
                            "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                            "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                            "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                        ),
                )
            Neo4jKVStorage(
                namespace = "full_docs",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            )
        }

        single(named("textChunks")) {
            val embeddingModel = get<EmbeddingModel>()
            val neo4jConfig =
                mapOf(
                    "neo4j" to
                        mapOf(
                            "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                            "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                            "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                        ),
                )
            Neo4jKVStorage(
                namespace = "text_chunks",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            )
        }

        single(named("fullEntities")) {
            val embeddingModel = get<EmbeddingModel>()
            val neo4jConfig =
                mapOf(
                    "neo4j" to
                        mapOf(
                            "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                            "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                            "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                        ),
                )
            Neo4jKVStorage(
                namespace = "full_entities",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            )
        }

        single(named("fullRelations")) {
            val embeddingModel = get<EmbeddingModel>()
            val neo4jConfig =
                mapOf(
                    "neo4j" to
                        mapOf(
                            "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                            "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                            "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
                        ),
                )
            Neo4jKVStorage(
                namespace = "full_relations",
                workspace = "default",
                globalConfig = neo4jConfig,
                embeddingFunc = embeddingModel,
            )
        }

        single {
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                hashingKv = get(named("hashingKv")),
                docStatusStorageOverride = get<Neo4jDocStatusStorage>(),
                fullDocsStorageOverride = get(named("fullDocs")),
                textChunksStorageOverride = get(named("textChunks")),
                fullEntitiesStorageOverride = get(named("fullEntities")),
                fullRelationsStorageOverride = get(named("fullRelations")),
            )
        }
    }

val ollamaExampleModule =
    module {
        single<ChatModel> {
            val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: "http://localhost:11434"
            OllamaChatModel.builder()
                .modelName("llama3")
                .baseUrl(baseUrl)
                .temperature(0.0)
                .build()
        }

        single<EmbeddingModel> {
            val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: "http://localhost:11434"
            OllamaEmbeddingModel.builder()
                .modelName("all-minilm")
                .baseUrl(baseUrl)
                .build()
        }

        single {
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                workingDir = "./ollama-demo",
            )
        }
    }

val mongodbGraphExampleModule =
    module {
        single<ChatModel> {
            val apiKey = System.getenv("OPENAI_API_KEY") ?: "sk-"
            OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini") // gpt_4o_mini_complete equivalent
                .timeout(Duration.ofSeconds(60))
                .build()
        }

        single<EmbeddingModel> {
            val apiKey = System.getenv("OPENAI_API_KEY") ?: "sk-"
            OpenAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-3-large")
                .dimensions(3072) // Assuming default for text-embedding-3-large
                .build()
        }

        single {
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                workingDir = "./mongodb_test_dir",
                graphStorageName = "MongoGraphStorage",
            )
        }
    }

val openAiNeo4jGraphExampleModule =
    module {
        single<ChatModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build()
        }

        single<EmbeddingModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            OpenAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-3-large")
                .dimensions(3072)
                .build()
        }

        single {
            val neo4jUri = System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"
            val neo4jUser = System.getenv("NEO4J_USERNAME") ?: "neo4j"
            val neo4jPass = System.getenv("NEO4J_PASSWORD") ?: "neo4j"

            var sanitizedNeo4jUri = neo4jUri
            if (neo4jUri.startsWith("http")) {
                val uriObj = URI(neo4jUri)
                val host = uriObj.host ?: "localhost"
                val port = if (uriObj.port == 7474) 7687 else uriObj.port
                sanitizedNeo4jUri = "bolt://$host:$port"
            }

            AddonConfig(
                neo4j =
                    Neo4jConfig(
                        uri = sanitizedNeo4jUri,
                        username = neo4jUser,
                        password = neo4jPass,
                    ),
                overrides =
                    LightRagOverrides(
                        chunkTokenSize = 50,
                        chunkOverlapTokenSize = 2,
                    ),
            )
        }

        single {
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                workingDir = "./neo4j_test_dir",
                graphStorageName = "Neo4jGraphStorage",
                addonConfig = get(),
            )
        }
    }

val openAiNeo4jEmbeddingStoreExampleModule =
    module {
        single<ChatModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build()
        }

        single<EmbeddingModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            OpenAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-3-small") // changed from large to small for consistency with other examples
                .build()
        }

        single {
            val neo4jUri = System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"
            val neo4jUser = System.getenv("NEO4J_USERNAME") ?: "neo4j"
            val neo4jPass = System.getenv("NEO4J_PASSWORD") ?: "neo4j"

            AddonConfig(
                neo4j =
                    Neo4jConfig(
                        uri = neo4jUri,
                        username = neo4jUser,
                        password = neo4jPass,
                    ),
                overrides =
                    LightRagOverrides(
                        chunkTokenSize = 256,
                        chunkOverlapTokenSize = 16,
                        cosineBetterThreshold = 0.2,
                    ),
            )
        }

        single {
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                workingDir = "./neo4j_embedding_store_demo",
                vectorStorageName = "Neo4jEmbeddingStoreVectorStorage",
                addonConfig = get(),
            )
        }
    }

val openAiNeo4jVectorExampleModule =
    module {
        single<ChatModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .timeout(Duration.ofSeconds(60))
                .build()
        }

        single<EmbeddingModel> {
            val apiKey =
                System.getenv("OPENAI_API_KEY")
                    ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")
            OpenAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-3-small")
                .build()
        }

        single {
            val neo4jUri = System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"
            val neo4jUser = System.getenv("NEO4J_USERNAME") ?: "neo4j"
            val neo4jPass = System.getenv("NEO4J_PASSWORD") ?: "neo4j"

            AddonConfig(
                neo4j =
                    Neo4jConfig(
                        uri = neo4jUri,
                        username = neo4jUser,
                        password = neo4jPass,
                    ),
                overrides =
                    LightRagOverrides(
                        chunkTokenSize = 256,
                        chunkOverlapTokenSize = 16,
                        cosineBetterThreshold = 0.2,
                    ),
            )
        }

        single {
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                workingDir = "./neo4j_vector_demo",
                graphStorageName = "Neo4jGraphStorage",
                vectorStorageName = "Neo4jVectorStorage",
                addonConfig = get(),
            )
        }
    }
