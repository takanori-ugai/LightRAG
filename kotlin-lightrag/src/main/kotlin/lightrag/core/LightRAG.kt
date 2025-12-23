package lightrag.core

import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.ollama.OllamaChatModel
import kotlinx.serialization.Serializable
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.core.types.DocStatusStorage
import lightrag.kg.json.JsonDocStatusStorage
import lightrag.kg.json.JsonKVStorage
import lightrag.kg.memory.InMemoryGraphStorage
import lightrag.kg.memory.InMemoryVectorStorage

@Serializable
data class QueryParam(
    val mode: String = "global",
    val only_need_context: Boolean = false,
    val response_type: String? = null,
    val top_k: Int = 10,
    val max_token_for_text_unit: Int = 4000,
    val max_token_for_global_context: Int = 4000,
    val max_token_for_local_context: Int = 4000,
)

class LightRAG(
    val workingDir: String = "./rag_storage",
    private val chatModel: ChatLanguageModel? = null,
) {
    // Default model if none provided (e.g. Ollama or OpenAI)
    private val model: ChatLanguageModel =
        chatModel ?: OllamaChatModel.builder()
            .baseUrl("http://localhost:11434")
            .modelName("llama3")
            .build()

    // Initialize Storages
    // In a real app, these would be injected or configured via a factory
    val docStatusStorage: DocStatusStorage = JsonDocStatusStorage(namespace = "doc_status", workspace = "default")
    val kvStorage: BaseKVStorage = JsonKVStorage(namespace = "kv_storage", workspace = "default")
    val vectorStorage: BaseVectorStorage = InMemoryVectorStorage(namespace = "vector_storage", workspace = "default")
    val graphStorage: BaseGraphStorage = InMemoryGraphStorage(namespace = "graph_storage", workspace = "default")

    suspend fun insert(input: String): String {
        // Implementation would go here
        return "track_id_placeholder"
    }

    suspend fun insert(input: List<String>): String {
        // Implementation would go here
        return "track_id_placeholder"
    }

    suspend fun query(
        query: String,
        param: QueryParam,
    ): String {
        // Use LangChain4j model to generate a response (mocking RAG logic)
        try {
            return model.generate(query)
        } catch (e: Exception) {
            return "Error generating response: ${e.message}"
        }
    }

    suspend fun getProcessingStatus(): Map<String, Int> {
        return docStatusStorage.getStatusCounts()
    }

    suspend fun deleteByDocId(docId: String): Map<String, String> {
        // Mock deletion logic using storage
        docStatusStorage.delete(listOf(docId))
        return mapOf("status" to "success", "doc_id" to docId)
    }
}
