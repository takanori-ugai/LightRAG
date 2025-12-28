package lightrag.operate

import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.CacheData
import lightrag.core.QueryParam
import lightrag.core.QueryParamCache
import lightrag.core.QueryResult
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.utils.JsonUtils
import lightrag.utils.Prompts
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

@Serializable
data class ChunkContext(
    @SerialName("reference_id")
    val referenceId: String?,
    val content: String?,
)

const val DEFAULT_MAX_TOTAL_TOKENS = 30000 // From python/constants.py

data class NaiveQueryParams(
    val query: String,
    val chunksVdb: BaseVectorStorage,
    val queryParam: QueryParam,
    val globalConfig: Map<String, Any?>,
    val hashingKv: BaseKVStorage? = null,
    val systemPrompt: String? = null,
    val chatModel: ChatModel? = null,
    val tokenizer: ((String) -> List<Int>),
    val decoder: ((List<Int>) -> String),
)

// Equivalent to python's _get_vector_context
private suspend fun getVectorContext(
    query: String,
    chunksVdb: BaseVectorStorage,
    queryParam: QueryParam,
    // Not used in naive, so omitting for now
): List<Map<String, Any?>> {
    try {
        val searchTopK = queryParam.chunkTopK.coerceAtMost(queryParam.topK)
        val cosineThreshold = chunksVdb.cosineBetterThanThreshold // Assuming this exists

        val results = chunksVdb.query(query, searchTopK)
        if (results.isEmpty()) {
            logger.info { "Naive query: 0 chunks (chunk_top_k:$searchTopK cosine:$cosineThreshold)" }
            return emptyList()
        }

        val validChunks = mutableListOf<Map<String, Any?>>()
        for (result in results) {
            if (result.containsKey("content")) {
                val chunkWithMetadata =
                    mapOf(
                        "content" to result["content"],
                        "created_at" to result["created_at"],
                        "file_path" to (result["file_path"] ?: "unknown_source"),
                        // Mark the source type so consumers know this came from the vector store
                        "source_type" to "vector",
                        "chunk_id" to (result["id"] ?: result["chunk_id"]),
                    )
                validChunks.add(chunkWithMetadata)
            }
        }

        logger.info { "Naive query: ${validChunks.size} chunks (chunk_top_k:$searchTopK cosine:$cosineThreshold)" }
        return validChunks
    } catch (e: Exception) {
        logger.error(e) { "Error in _getVectorContext" }
        return emptyList()
    }
}

// Equivalent to python's generate_reference_list_from_chunks

/**
 * This function assumes chunk_id and file_path are present in the chunk map.
 * It adds "reference_id" to each processed chunk.
 */
fun generateReferenceListFromChunks(chunks: List<Map<String, Any?>>): Pair<List<Map<String, String>>, List<Map<String, Any?>>> {
    val referenceList = mutableListOf<Map<String, String>>()
    val processedChunks = mutableListOf<Map<String, Any?>>()
    val seenFilePaths = mutableSetOf<String>()

    for ((index, chunk) in chunks.withIndex()) {
        val referenceId = (index + 1).toString()
        val filePath = chunk["file_path"] as? String ?: "unknown_source"

        if (!seenFilePaths.contains(filePath)) {
            referenceList.add(mapOf("reference_id" to referenceId, "file_path" to filePath))
            seenFilePaths.add(filePath)
        }

        processedChunks.add(chunk + ("reference_id" to referenceId))
    }
    return referenceList to processedChunks
}

// Simplified version of python's process_chunks_unified, focusing on truncation
// This will just apply token limit, no reranking for now.
fun processChunksUnified(
    @Suppress("UNUSED_PARAMETER") query: String,
    // Not directly used in this simplified version for reranking, but kept for signature
    uniqueChunks: List<Map<String, Any?>>,
    @Suppress("UNUSED_PARAMETER") queryParam: QueryParam,
    @Suppress("UNUSED_PARAMETER") globalConfig: Map<String, Any?>,
    @Suppress("UNUSED_PARAMETER") sourceType: String,
    // e.g., "vector", "entity", "relation"
    chunkTokenLimit: Int,
    tokenizer: ((String) -> List<Int>),
    decoder: ((List<Int>) -> String),
): List<Map<String, Any?>> {
    val resultChunks = mutableListOf<Map<String, Any?>>()
    var currentTokens = 0

    // Ensure chunks are unique by content, preserving order
    val distinctChunks = uniqueChunks.distinctBy { it["content"] as? String }

    for (chunk in distinctChunks) {
        val content = chunk["content"] as? String ?: ""
        val chunkTokens = tokenizer(content).size

        if (currentTokens + chunkTokens <= chunkTokenLimit) {
            resultChunks.add(chunk)
            currentTokens += chunkTokens
        } else {
            // If adding the whole chunk exceeds the limit, try to truncate it
            val remainingTokens = chunkTokenLimit - currentTokens
            if (remainingTokens > 0) {
                val truncatedContent = truncateTextByTokenSize(content, remainingTokens, tokenizer, decoder)
                if (truncatedContent.isNotBlank()) {
                    resultChunks.add(chunk + ("content" to truncatedContent))
                    currentTokens += tokenizer(truncatedContent).size
                }
            }
            break // No more chunks can be added or partially added
        }
    }
    logger.debug { "Processed chunks: ${resultChunks.size} (total tokens: $currentTokens / $chunkTokenLimit)" }
    return resultChunks
}

fun truncateTextByTokenSize(
    text: String,
    maxTokenSize: Int,
    tokenizer: ((String) -> List<Int>),
    decoder: ((List<Int>) -> String),
): String {
    val tokens = tokenizer(text)
    if (tokens.size <= maxTokenSize) {
        return text
    }
    return decoder(tokens.subList(0, maxTokenSize))
}

// Function to convert map to JSON string, similar to Python's json.dumps
// This function needs to be properly implemented based on the Python version.
// For now, a placeholder that constructs a basic rawData map.
fun convertToJsonFormat(
    entities: List<Map<String, Any?>>,
    relations: List<Map<String, Any?>>,
    chunks: List<Map<String, Any?>>,
    references: List<Map<String, String>>,
    queryMode: String,
    @Suppress("UNUSED_PARAMETER") relationIdToOriginal: Map<Pair<String, String>, Any?> = emptyMap(),
): Map<String, Any?> {
    val dataMap =
        mutableMapOf<String, Any?>(
            "entities" to entities,
            "relationships" to relations,
            "chunks" to chunks,
            "references" to references,
        )
    val metadataMap =
        mutableMapOf<String, Any?>(
            "query_mode" to queryMode,
        )
    return mutableMapOf(
        "data" to dataMap,
        "metadata" to metadataMap,
    )
}

// Cache helpers could live in a dedicated Cache.kt or Utils.kt module
// For now, embedding them here for a self-contained replacement.

private suspend fun handleCache(
    hashingKv: BaseKVStorage?,
    argsHash: String,
    @Suppress("UNUSED_PARAMETER") prompt: String,
    @Suppress("UNUSED_PARAMETER") mode: String,
    @Suppress("UNUSED_PARAMETER") cacheType: String,
): Pair<String, Long>? { // Returns content and timestamp
    if (hashingKv == null) return null

    val cached = hashingKv.getById(argsHash)
    if (cached != null) {
        val cachedContent = cached["content"] as? String
        val timestamp = (cached["create_time"] as? Number)?.toLong() ?: 0L
        if (cachedContent != null) {
            return cachedContent to timestamp
        }
    }
    return null
}

private suspend fun saveToCache(
    hashingKv: BaseKVStorage?,
    cacheData: CacheData,
) {
    if (hashingKv == null) return
    hashingKv.upsert(
        mapOf(
            cacheData.argsHash to
                mapOf(
                    "content" to cacheData.content,
                    "prompt" to cacheData.prompt,
                    "mode" to cacheData.mode,
                    "cache_type" to cacheData.cacheType,
                    "queryparam" to cacheData.queryParam,
                    "history_messages" to cacheData.historyMessages,
                    // Unix timestamp in seconds
                    "create_time" to System.currentTimeMillis() / 1000,
                ).filterValues { it != null }.mapValues { it.value!! },
        ),
    )
}

private suspend fun saveQueryCache(
    params: NaiveQueryParams,
    argsHash: String,
    content: String,
    userQuery: String,
    maxTotalTokens: Int,
) {
    if (params.hashingKv != null && (params.globalConfig["enable_llm_cache"] as? Boolean == true)) {
        val queryParamDict =
            QueryParamCache(
                mode = params.queryParam.mode,
                responseType = params.queryParam.responseType,
                topK = params.queryParam.topK,
                chunkTopK = params.queryParam.chunkTopK,
                maxEntityTokens = params.queryParam.maxEntityTokens,
                maxRelationTokens = params.queryParam.maxRelationTokens,
                maxTotalTokens = maxTotalTokens,
                hlKeywords = params.queryParam.hlKeywords.joinToString(", "),
                llKeywords = params.queryParam.llKeywords.joinToString(", "),
                userPrompt = params.queryParam.userPrompt ?: "",
                enableRerank = params.queryParam.enableRerank,
            )
        saveToCache(
            params.hashingKv,
            CacheData(
                argsHash = argsHash,
                content = content,
                prompt = userQuery,
                mode = params.queryParam.mode,
                cacheType = "query",
                queryParam = queryParamDict,
                historyMessages = params.queryParam.conversationHistory,
            ),
        )
    }
}

// This function needs to be imported or replicated from utils.
// For now, a placeholder. The actual implementation in utils.kt will need to handle the varargs
fun computeArgsHash(vararg args: Any?): String {
    return computeMd5(args.joinToString("|"))
}

@Suppress("UNUSED")
fun removeThinkTags(text: String): String {
    // This is a simplified placeholder. In Python, it would remove specific XML-like tags.
    // Assuming for now that the LLM is not generating these tags in Kotlin or they are removed differently.
    return text.replace("<THINK>", "").replace("</THINK>", "")
}

suspend fun naiveQuery(params: NaiveQueryParams): QueryResult? {
    if (params.query.isBlank()) {
        return QueryResult(content = Prompts.FAIL_RESPONSE)
    }

    val model = params.chatModel ?: params.globalConfig["llm_model_func"] as? ChatModel
    if (model == null) {
        logger.error { "No ChatModel provided for naiveQuery" }
        return QueryResult(content = "Error: No LLM model configured.")
    }

    val tokenizer = params.tokenizer // Use the injected tokenizer
    val decoder = params.decoder

    // Get chunks using vector context (equivalent to _get_vector_context)
    val chunks = getVectorContext(params.query, params.chunksVdb, params.queryParam)

    if (chunks.isEmpty()) {
        logger.info { "[naive_query] No relevant document chunks found; returning no-result." }
        return null
    }

    // Calculate dynamic token limit for chunks
    val maxTotalTokens =
        params.queryParam.maxTotalTokens.coerceAtMost(
            (params.globalConfig["max_total_tokens"] as? Int) ?: DEFAULT_MAX_TOTAL_TOKENS,
        )

    val userPromptStr =
        if (!params.queryParam.userPrompt.isNullOrBlank()) {
            "\n\n${params.queryParam.userPrompt}"
        } else {
            "n/a"
        }
    val responseType = params.queryParam.responseType ?: "Multiple Paragraphs"

    val sysPromptTemplate = params.systemPrompt ?: Prompts.NAIVE_RAG_RESPONSE

    // Create a preliminary system prompt with empty content_data to calculate overhead
    val preSysPrompt =
        sysPromptTemplate
            .replace("{response_type}", responseType)
            .replace("{user_prompt}", userPromptStr)
            .replace("{content_data}", "") // Empty for overhead calculation

    val sysPromptTokens = tokenizer(preSysPrompt).size
    val queryTokens = tokenizer(params.query).size
    val bufferTokens = 200 // reserved for reference list and safety buffer
    val availableChunkTokens = maxTotalTokens - (sysPromptTokens + queryTokens + bufferTokens)

    logger.debug {
        "Naive query token allocation - Total: $maxTotalTokens, " +
            "SysPrompt: $sysPromptTokens, Query: $queryTokens, " +
            "Buffer: $bufferTokens, Available for chunks: $availableChunkTokens"
    }

    // Process chunks using unified processing with dynamic token limit
    val processedChunks =
        processChunksUnified(
            query = params.query,
            uniqueChunks = chunks,
            queryParam = params.queryParam,
            globalConfig = params.globalConfig,
            sourceType = "vector",
            chunkTokenLimit = availableChunkTokens,
            tokenizer = tokenizer,
            decoder = decoder,
        )

    // Generate reference list from processed chunks
    val (referenceList, processedChunksWithRefIds) = generateReferenceListFromChunks(processedChunks)

    logger.info { "Final context: ${processedChunksWithRefIds.size} chunks" }

    // Build raw data structure for naive mode
    // Entities and relationships stay empty for naive mode
    val rawData =
        convertToJsonFormat(
            emptyList(),
            emptyList(),
            processedChunksWithRefIds,
            referenceList,
            "naive",
        )

    // Add complete metadata for naive mode
    val metadata = mutableMapOf<String, Any?>()
    metadata["keywords"] = mapOf("high_level" to emptyList<String>(), "low_level" to emptyList<String>())
    metadata["processing_info"] =
        mapOf(
            "total_chunks_found" to chunks.size,
            "final_chunks_count" to processedChunksWithRefIds.size,
        )
    (rawData as MutableMap)["metadata"] = metadata
    // Add metadata to rawData

    // Build chunks_context from processed chunks with reference IDs
    val chunksContext =
        processedChunksWithRefIds.map {
            ChunkContext(
                referenceId = it["reference_id"] as? String,
                content = it["content"] as? String,
            )
        }

    val textUnitsStr = JsonUtils.convertObjectToJson(chunksContext)
    val referenceListStr =
        referenceList.joinToString("\n") { ref ->
            "[${ref["reference_id"]}] ${ref["file_path"]}"
        }

    val naiveContextTemplate = Prompts.NAIVE_QUERY_CONTEXT
    val contextContent =
        naiveContextTemplate
            .replace("{text_chunks_str}", textUnitsStr)
            .replace("{reference_list_str}", referenceListStr)

    if (params.queryParam.onlyNeedContext) {
        return QueryResult(content = contextContent, rawData = rawData)
    }

    val sysPrompt =
        sysPromptTemplate
            .replace("{response_type}", responseType)
            .replace("{user_prompt}", userPromptStr)
            .replace("{content_data}", contextContent)

    val userQuery = params.query

    if (params.queryParam.onlyNeedPrompt) {
        val promptContent =
            listOf(sysPrompt, "---User Query---", userQuery)
                .joinToString("\n\n")
        return QueryResult(content = promptContent, rawData = rawData)
    }

    // Handle cache
    val argsHash =
        computeArgsHash(
            params.queryParam.mode,
            params.query,
            params.queryParam.responseType ?: "",
            params.queryParam.topK.toString(),
            params.queryParam.chunkTopK.toString(),
            params.queryParam.maxEntityTokens.toString(),
            params.queryParam.maxRelationTokens.toString(),
            maxTotalTokens.toString(),
            params.queryParam.userPrompt ?: "",
            params.queryParam.enableRerank.toString(),
        )
    val cachedResult = handleCache(params.hashingKv, argsHash, userQuery, params.queryParam.mode, "query")
    if (cachedResult != null) {
        val (cachedResponse, _) = cachedResult
        logger.info { " == LLM cache == Query cache hit, using cached response as query result" }
        return QueryResult(content = cachedResponse, rawData = rawData)
    }

    // Call LLM
    if (params.queryParam.stream) {
        val streamingModel = model as? StreamingChatModel
        if (streamingModel == null) {
            logger.error { "Streaming is requested but the model does not support it." }
            return QueryResult(content = "Error: Streaming not supported by model.")
        }
        logger.trace { "SysPrompt: $sysPrompt" }
        logger.trace { "UserPrompt: $userQuery" }
        val responseFlow =
            flow {
                val fullResponse = StringBuilder()
                val channel = Channel<String>(Channel.UNLIMITED)

                streamingModel.chat(
                    listOf(SystemMessage(sysPrompt), UserMessage(userQuery)),
                    object : StreamingChatResponseHandler {
                        override fun onPartialResponse(partialResponse: String) {
                            channel.trySend(partialResponse)
                            fullResponse.append(partialResponse)
                        }

                        override fun onCompleteResponse(response: ChatResponse) {
                            channel.close()
                        }

                        override fun onError(error: Throwable) {
                            channel.close(error)
                        }
                    },
                )

                for (token in channel) {
                    emit(token)
                }

                saveQueryCache(params, argsHash, fullResponse.toString(), userQuery, maxTotalTokens)
            }
        return QueryResult(responseIterator = responseFlow, rawData = rawData, isStreaming = true)
    } else {
        var responseContent =
            try {
                logger.trace { "SysPrompt: $sysPrompt" }
                logger.trace { "UserPrompt: $userQuery" }
                val chatResponse = model.chat(listOf(SystemMessage(sysPrompt), UserMessage(userQuery)))
                chatResponse.aiMessage()?.text() ?: ""
            } catch (e: Exception) {
                logger.error(e) { "Error generating response in naiveQuery" }
                "Error generating response."
            }

        if (responseContent.length > sysPrompt.length) {
            responseContent =
                responseContent
                    .replace(sysPrompt, "")
                    .replace("user", "")
                    .replace("model", "")
                    .replace(userQuery, "")
                    .replace("<system>", "")
                    .replace("</system>", "")
                    .trim()
        }

        saveQueryCache(params, argsHash, responseContent, userQuery, maxTotalTokens)

        return QueryResult(content = responseContent, rawData = rawData)
    }
}
