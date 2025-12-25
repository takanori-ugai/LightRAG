package lightrag.core

import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.Serializable

data class QueryResult(
    val content: String? = null,
    val responseIterator: Flow<String>? = null,
    val rawData: Any? = null,
    val isStreaming: Boolean = false,
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
    val queryParam: QueryParamCache,
)
