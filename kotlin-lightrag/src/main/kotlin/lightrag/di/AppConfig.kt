package lightrag.di

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import lightrag.core.AddonConfig
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.DocStatusStorage

data class AppConfig(
    val workingDir: String = "./rag_storage",
    val graphStorageName: String = "InMemoryGraphStorage",
    val vectorStorageName: String = "InMemoryVectorStorage",
    val addonConfig: AddonConfig = AddonConfig(),
    val llmBinding: String = "ollama",
    val llmModelName: String = "llama3",
    val embeddingBinding: String = "ollama",
    val embeddingModelName: String = "all-minilm",
    val chatModel: ChatModel,
    val embeddingModel: EmbeddingModel,
    val hashingKv: BaseKVStorage? = null,
    val docStatusStorageOverride: DocStatusStorage? = null,
    val fullDocsStorageOverride: BaseKVStorage? = null,
    val textChunksStorageOverride: BaseKVStorage? = null,
    val fullEntitiesStorageOverride: BaseKVStorage? = null,
    val fullRelationsStorageOverride: BaseKVStorage? = null,
)
