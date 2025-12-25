package lightrag.operate

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatLanguageModel
import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.QueryParam
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.utils.JsonUtils
import lightrag.utils.Prompts
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

data class NaiveQueryParams(
    val query: String,
    val chunksVdb: BaseVectorStorage,
    val queryParam: QueryParam,
    val globalConfig: Map<String, Any?>,
    val hashingKv: BaseKVStorage? = null,
    val systemPrompt: String? = null,
    val chatModel: ChatLanguageModel? = null,
)

suspend fun naiveQuery(params: NaiveQueryParams): String? {
    if (params.query.isBlank()) {
        return Prompts.FAIL_RESPONSE
    }

    // Basic vector search (naive: direct topK from chunks VDB)
    val searchTopK = params.queryParam.chunkTopK.coerceAtMost(params.queryParam.topK)

    val results = params.chunksVdb.query(params.query, searchTopK)
    if (results.isEmpty()) {
        return null
    }

    val contextBuilder = StringBuilder()
    contextBuilder.append(Prompts.NAIVE_QUERY_CONTEXT)
    // We need to fill in text_chunks_str and reference_list_str

    val docChunks =
        results.mapIndexed { index, res ->
            val content = res["content"] ?: ""
            mapOf(
                "reference_id" to "${index + 1}",
                "content" to content,
                "file_path" to (res["file_path"] ?: "unknown_source"),
            )
        }

    // For now, simple context building (simplification of Python logic)
    val textChunksStr =
        docChunks.joinToString("\n") { chunk ->
            "{\"reference_id\": \"${chunk["reference_id"]}\", \"content\": \"${
                JsonUtils.escape(chunk["content"].toString())
            }\"}"
        }

    val referenceListStr =
        docChunks.joinToString("\n") { chunk ->
            "[${chunk["reference_id"]}] ${chunk["file_path"] ?: "unknown_source"}"
        }

    val contextContent =
        contextBuilder.toString()
            .replace("{text_chunks_str}", textChunksStr)
            .replace("{reference_list_str}", referenceListStr)

    val sysPromptTemplate = params.systemPrompt ?: Prompts.NAIVE_RAG_RESPONSE

    val userPrompt =
        buildString {
            if (!params.queryParam.userPrompt.isNullOrBlank()) {
                append(params.queryParam.userPrompt)
            } else {
                append("n/a")
            }
            if (!params.queryParam.responseType.isNullOrBlank()) {
                append("\n\n")
                append(params.queryParam.responseType)
            }
        }

    val sysPrompt =
        sysPromptTemplate
            .replace(
                "{response_type}",
                params.queryParam.responseType ?: "Multiple Paragraphs",
            )
            .replace("{user_prompt}", userPrompt)
            .replace("{content_data}", contextContent)

    if (params.queryParam.onlyNeedContext) {
        return contextContent
    }

    if (params.queryParam.onlyNeedPrompt) {
        return listOf(sysPrompt, "---", params.query).joinToString("\n")
    }

    // Call LLM
    val model = params.chatModel ?: params.globalConfig["llm_model_func"] as? ChatLanguageModel

    if (model == null) {
        logger.error { "No ChatLanguageModel provided for naiveQuery" }
        return "Error: No LLM model configured."
    }

    // Build cache key
    val cacheKeySeed =
        listOf(
            params.queryParam.mode,
            params.query,
            params.queryParam.responseType ?: "",
            params.queryParam.topK.toString(),
            params.queryParam.chunkTopK.toString(),
            params.queryParam.userPrompt ?: "",
            params.queryParam.enableRerank.toString(),
        ).joinToString("|")
    val cacheKey = "query_cache_${computeMd5(cacheKeySeed)}"

    // Try cache if provided
    if (params.hashingKv != null) {
        val cached = params.hashingKv.getById(cacheKey)
        val cachedContent = cached?.get("content") as? String
        if (!cachedContent.isNullOrEmpty()) {
            logger.info { " == LLM cache == Query cache hit, using cached response as query result" }
            return cachedContent
        }
    }

    // Call LLM
    return try {
        val messages =
            listOf(
                SystemMessage(sysPrompt),
                UserMessage(params.query),
            )
        val response: AiMessage = model.generate(messages).content()
        val text = response.text()

        if (params.hashingKv != null) {
            params.hashingKv.upsert(mapOf(cacheKey to mapOf("content" to text)))
        }
        text
    } catch (e: Exception) {
        logger.error(e) { "Error generating response in naiveQuery" }
        "Error generating response."
    }
}
