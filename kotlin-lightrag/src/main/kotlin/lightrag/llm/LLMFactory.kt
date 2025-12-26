package lightrag.llm

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import java.time.Duration

object LLMFactory {
    fun createChatModel(
        binding: String,
        modelName: String,
        baseUrl: String? = null,
        apiKey: String? = null,
        timeout: Long = 60,
    ): ChatModel {
        return when (binding) {
            "openai" -> {
                val builder =
                    OpenAiChatModel.builder()
                        .modelName(modelName)
                        .apiKey(apiKey ?: "demo")
                        .logRequests(true)
                        .logResponses(true)
                        .timeout(Duration.ofSeconds(timeout))
                if (baseUrl != null) {
                    builder.baseUrl(baseUrl)
                }
                builder.build()
            }
            "ollama" -> {
                val builder =
                    OllamaChatModel.builder()
                        .modelName(modelName)
                        .baseUrl(baseUrl ?: "http://localhost:11434")
                        .timeout(Duration.ofSeconds(timeout))
                builder.build()
            }
            else -> throw IllegalArgumentException("Unsupported LLM binding: $binding")
        }
    }

    fun createEmbeddingModel(
        binding: String,
        modelName: String,
        baseUrl: String? = null,
        apiKey: String? = null,
        timeout: Long = 60,
    ): EmbeddingModel {
        return when (binding) {
            "openai" -> {
                val builder =
                    OpenAiEmbeddingModel.builder()
                        .modelName(modelName)
                        .apiKey(apiKey ?: "demo")
                        .logRequests(true)
                        .timeout(Duration.ofSeconds(timeout))
                if (baseUrl != null) {
                    builder.baseUrl(baseUrl)
                }
                builder.build()
            }
            "ollama" -> {
                val builder =
                    OllamaEmbeddingModel.builder()
                        .modelName(modelName)
                        .baseUrl(baseUrl ?: "http://localhost:11434")
                        .timeout(Duration.ofSeconds(timeout))
                builder.build()
            }
            else -> throw IllegalArgumentException("Unsupported embedding binding: $binding")
        }
    }
}
