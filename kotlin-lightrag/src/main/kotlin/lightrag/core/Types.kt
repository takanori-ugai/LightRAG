package lightrag.core

import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

data class QueryResult(
    val content: String? = null,
    val responseIterator: Flow<String>? = null,
    val rawData: Map<String, Any?>? = null,
    val isStreaming: Boolean = false,
) {
    val referenceList: List<Map<String, String>>
        get() =
            rawData?.get("data")
                ?.let { it as? Map<String, Any?> }
                ?.get("references")
                ?.let { it as? List<Map<String, String>> }
                ?: emptyList()

    val metadata: Map<String, Any?>
        get() =
            rawData?.get("metadata")
                ?.let { it as? Map<String, Any?> }
                ?: emptyMap()
}

data class QueryContextResult(
    val context: String,
    val rawData: Map<String, Any?>,
) {
    val referenceList: List<Map<String, String>>
        get() =
            rawData["data"]
                ?.let { it as? Map<String, Any?> }
                ?.get("references")
                ?.let { it as? List<Map<String, String>> }
                ?: emptyList()
}

@Serializable
data class TextChunkSchema(
    val tokens: Int,
    val content: String,
    @SerialName("full_doc_id")
    val fullDocId: String,
    @SerialName("chunk_order_index")
    val chunkOrderIndex: Int,
)

@Serializable
data class QueryParamCache(
    val mode: String,
    val responseType: String?,
    val topK: Int,
    val chunkTopK: Int,
    val maxEntityTokens: Int,
    val maxRelationTokens: Int,
    val maxTotalTokens: Int,
    val hlKeywords: String,
    val llKeywords: String,
    val userPrompt: String,
    val enableRerank: Boolean,
)

@Serializable
data class CacheData(
    val argsHash: String,
    val content: String,
    val prompt: String,
    val mode: String,
    val cacheType: String,
    val queryParam: QueryParamCache? = null,
    val historyMessages: List<Map<String, String>>? = null,
)
