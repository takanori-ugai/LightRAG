package lightrag.core

import kotlinx.serialization.Serializable
import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.ollama.OllamaChatModel

@Serializable
data class QueryParam(
    val mode: String = "global",
    val only_need_context: Boolean = false,
    val response_type: String? = null,
    val top_k: Int = 10,
    val max_token_for_text_unit: Int = 4000,
    val max_token_for_global_context: Int = 4000,
    val max_token_for_local_context: Int = 4000
)

@Serializable
data class DocStatus(
    val id: String,
    val status: String,
    val content_summary: String? = null,
    val content_length: Int? = 0,
    val created_at: String? = null,
    val updated_at: String? = null
)

class LightRAG(
    val workingDir: String = "./rag_storage",
    private val chatModel: ChatLanguageModel? = null
) {
    // Default model if none provided (e.g. Ollama or OpenAI)
    private val model: ChatLanguageModel = chatModel ?: OllamaChatModel.builder()
        .baseUrl("http://localhost:11434")
        .modelName("llama3")
        .build()

    suspend fun insert(input: String): String {
        // Implementation would go here
        return "track_id_placeholder"
    }

    suspend fun insert(input: List<String>): String {
         // Implementation would go here
        return "track_id_placeholder"
    }

    suspend fun query(query: String, param: QueryParam): String {
        // Use LangChain4j model to generate a response (mocking RAG logic)
        try {
             return model.generate(query)
        } catch (e: Exception) {
             return "Error generating response: ${e.message}"
        }
    }

    suspend fun getProcessingStatus(): Map<String, Int> {
         return mapOf("PENDING" to 0, "PROCESSING" to 0, "PROCESSED" to 0, "FAILED" to 0)
    }

    suspend fun deleteByDocId(docId: String): Map<String, String> {
        return mapOf("status" to "success", "doc_id" to docId)
    }
}
