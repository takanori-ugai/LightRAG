package lightrag.operate

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.chat.StreamingChatLanguageModel
import dev.langchain4j.model.output.Response
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import lightrag.core.*
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.utils.JsonUtils
import lightrag.utils.Prompts
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

data class ChunkingResult(
    val tokens: Int,
    val content: String,
    val chunkOrderIndex: Int,
)

data class EntityExtractionResult(
    val entityName: String,
    val entityType: String,
    val description: String,
    val sourceId: String,
)

data class RelationExtractionResult(
    val srcId: String,
    val tgtId: String,
    val description: String,
    val keywords: String,
    val weight: Double,
    val sourceId: String,
)

data class ContextResult(
    val contextStr: String,
    val rawData: Map<String, Any?>?,
)

fun chunkingByTokenSize(
    // Assuming tokenizer returns list of tokens (Int)
    tokenizer: (String) -> List<Int>,
    decoder: (List<Int>) -> String,
    content: String,
    splitByCharacter: String? = null,
    splitByCharacterOnly: Boolean = false,
    chunkOverlapTokenSize: Int = 100,
    chunkTokenSize: Int = 1200,
): List<ChunkingResult> {
    val tokens = tokenizer(content)
    val results = mutableListOf<ChunkingResult>()

    if (splitByCharacter != null) {
        val rawChunks = content.split(splitByCharacter)
        val newChunks = mutableListOf<Pair<Int, String>>()

        if (splitByCharacterOnly) {
            for (chunk in rawChunks) {
                val chunkTokens = tokenizer(chunk)
                if (chunkTokens.size > chunkTokenSize) {
                    logger.warn {
                        "Chunk split_by_character exceeds token limit: " +
                            "len=${chunkTokens.size} limit=$chunkTokenSize"
                    }
                    // In Python code it raises exception, here we can log and maybe truncate or skip?
                    // Python raises ChunkTokenLimitExceededError.
                    // For now, let's just proceed or throw RuntimeException
                    throw RuntimeException("Chunk token limit exceeded: ${chunkTokens.size} > $chunkTokenSize")
                }
                newChunks.add(chunkTokens.size to chunk)
            }
        } else {
            for (chunk in rawChunks) {
                val chunkTokens = tokenizer(chunk)
                if (chunkTokens.size > chunkTokenSize) {
                    var start = 0
                    while (start < chunkTokens.size) {
                        val end = minOf(start + chunkTokenSize, chunkTokens.size)
                        val chunkContent = decoder(chunkTokens.subList(start, end))
                        newChunks.add(
                            minOf(chunkTokenSize, chunkTokens.size - start) to chunkContent,
                        )
                        start += (chunkTokenSize - chunkOverlapTokenSize)
                    }
                } else {
                    newChunks.add(chunkTokens.size to chunk)
                }
            }
        }

        newChunks.forEachIndexed { index, (len, chunk) ->
            results.add(ChunkingResult(len, chunk.trim(), index))
        }
    } else {
        var start = 0
        var index = 0
        while (start < tokens.size) {
            val end = minOf(start + chunkTokenSize, tokens.size)
            val chunkContent = decoder(tokens.subList(start, end))
            results.add(
                ChunkingResult(
                    minOf(chunkTokenSize, tokens.size - start),
                    chunkContent.trim(),
                    index,
                ),
            )
            start += (chunkTokenSize - chunkOverlapTokenSize)
            index++
        }
    }
    return results
}

suspend fun extractEntities(
    chunks: Map<String, Map<String, Any>>,
    globalConfig: Map<String, Any?>,
): Pair<Map<String, List<EntityExtractionResult>>, Map<String, List<RelationExtractionResult>>> {
    val model = globalConfig["llm_model_func"] as? ChatLanguageModel
    if (model == null) {
        logger.error { "No ChatLanguageModel provided for entity extraction" }
        return emptyMap<String, List<EntityExtractionResult>>() to emptyMap()
    }

    val nodes = mutableMapOf<String, MutableList<EntityExtractionResult>>()
    val edges = mutableMapOf<String, MutableList<RelationExtractionResult>>()

    val entityTypes =
        (globalConfig["entity_types"] as? List<*>)?.map { it.toString() }
            ?: listOf("Person", "Organization", "Location")
    val language = globalConfig["language"] as? String ?: "English"

    val examples = Prompts.ENTITY_EXTRACTION_EXAMPLES.joinToString("\n")
    val contextBase =
        mapOf(
            "tuple_delimiter" to Prompts.DEFAULT_TUPLE_DELIMITER,
            "completion_delimiter" to Prompts.DEFAULT_COMPLETION_DELIMITER,
            "entity_types" to entityTypes.joinToString(","),
            "language" to language,
            "examples" to examples,
        )

    val systemPrompt =
        Prompts.ENTITY_EXTRACTION_SYSTEM_PROMPT
            .replace("{tuple_delimiter}", contextBase["tuple_delimiter"]!!)
            .replace("{completion_delimiter}", contextBase["completion_delimiter"]!!)
            .replace("{entity_types}", contextBase["entity_types"]!!)
            .replace("{language}", contextBase["language"]!!)
            .replace("{examples}", contextBase["examples"]!!)

    chunks.forEach { (chunkKey, chunkData) ->
        val content = chunkData["content"] as? String ?: return@forEach

        val userPrompt =
            Prompts.ENTITY_EXTRACTION_USER_PROMPT
                .replace("{language}", language)
                .replace("{entity_types}", entityTypes.joinToString(","))
                .replace("{input_text}", content)
                .replace("{completion_delimiter}", Prompts.DEFAULT_COMPLETION_DELIMITER)
                .replace("{tuple_delimiter}", Prompts.DEFAULT_TUPLE_DELIMITER)

        try {
            val messages =
                listOf(
                    SystemMessage(systemPrompt),
                    UserMessage(userPrompt),
                )
            val response: AiMessage = model.generate(messages).content()
            val responseText = response.text()

            val (chunkNodes, chunkEdges) =
                processExtractionResult(responseText, chunkKey, Prompts.DEFAULT_TUPLE_DELIMITER)

            chunkNodes.forEach { (name, list) ->
                nodes.computeIfAbsent(name) { mutableListOf() }.addAll(list)
            }
            chunkEdges.forEach { (key, list) ->
                edges.computeIfAbsent(key) { mutableListOf() }.addAll(list)
            }
        } catch (e: Exception) {
            logger.error(e) { "Error extracting entities for chunk $chunkKey" }
        }
    }

    return nodes to edges
}

fun processExtractionResult(
    result: String,
    chunkKey: String,
    tupleDelimiter: String,
): Pair<Map<String, List<EntityExtractionResult>>, Map<String, List<RelationExtractionResult>>> {
    val nodes = mutableMapOf<String, MutableList<EntityExtractionResult>>()
    val edges = mutableMapOf<String, MutableList<RelationExtractionResult>>()

    // Basic parsing logic (simplified)
    val lines = result.split("\n")
    for (line in lines) {
        if (line.trim().isEmpty()) continue
        if (line.contains(Prompts.DEFAULT_COMPLETION_DELIMITER)) continue

        // Handle delimiter issues or format variations if needed
        val parts = line.split(tupleDelimiter)

        if (parts.size >= 4 && parts[0].trim() == "entity") {
            // entity<|>name<|>type<|>description
            val name = parts[1].trim()
            val type = parts[2].trim()
            val desc = parts[3].trim()
            nodes.computeIfAbsent(name) { mutableListOf() }.add(
                EntityExtractionResult(name, type, desc, chunkKey),
            )
        } else if (parts.size >= 5 && parts[0].trim() == "relation") {
            // relation<|>src<|>tgt<|>keywords<|>desc
            val src = parts[1].trim()
            val tgt = parts[2].trim()
            val keywords = parts[3].trim()
            val desc = parts[4].trim()
            val weight = 1.0 // Default
            val key = listOf(src, tgt).sorted().joinToString("#")

            edges.computeIfAbsent(key) { mutableListOf() }.add(
                RelationExtractionResult(src, tgt, desc, keywords, weight, chunkKey),
            )
        }
    }
    return nodes to edges
}

suspend fun mergeNodesAndEdges(
    nodes: Map<String, List<EntityExtractionResult>>,
    edges: Map<String, List<RelationExtractionResult>>,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    globalConfig: Map<String, Any?>,
) {
    // 1. Process Nodes
    for ((name, entityList) in nodes) {
        // Simple merge: take the longest description and majority type
        // In real impl, use LLM to summarize descriptions
        val longestDesc = entityList.maxByOrNull { it.description.length }?.description ?: ""
        val typeCounts = entityList.groupingBy { it.entityType }.eachCount()
        val majorityType = typeCounts.maxByOrNull { it.value }?.key ?: "Unknown"
        val sourceIds = entityList.joinToString(Constants.GRAPH_FIELD_SEP) { it.sourceId }

        val nodeData =
            mapOf(
                "entity_id" to name,
                "entity_type" to majorityType,
                "description" to longestDesc,
                "source_id" to sourceIds,
            )

        knowledgeGraphInst.upsertNode(name, nodeData)

        // Update VDB
        val entityContent = "$name\n$longestDesc"
        val vdbData =
            mapOf(
                computeMd5(name) to
                    mapOf(
                        "content" to entityContent,
                        "entity_name" to name,
                    ),
            )
        entitiesVdb.upsert(vdbData)
    }

    // 2. Process Edges
    for ((key, edgeList) in edges) {
        // Simple merge
        val first = edgeList.first()
        val src = first.srcId
        val tgt = first.tgtId
        val longestDesc = edgeList.maxByOrNull { it.description.length }?.description ?: ""
        val allKeywords = edgeList.joinToString(", ") { it.keywords }
        val weight = edgeList.sumOf { it.weight }
        val sourceIds = edgeList.joinToString(Constants.GRAPH_FIELD_SEP) { it.sourceId }

        val edgeData =
            mapOf(
                "weight" to weight.toString(),
                "description" to longestDesc,
                "keywords" to allKeywords,
                "source_id" to sourceIds,
                "src_id" to src,
                "tgt_id" to tgt,
            )

        knowledgeGraphInst.upsertEdge(src, tgt, edgeData)

        // Update VDB
        val relContent = "$allKeywords\t$src\n$tgt\n$longestDesc"
        val vdbData =
            mapOf(
                computeMd5(key) to
                    mapOf(
                        "content" to relContent,
                        "src_id" to src,
                        "tgt_id" to tgt,
                    ),
            )
        relationshipsVdb.upsert(vdbData)
    }
}

suspend fun getContextStrForQuery(
    query: String,
    queryParam: QueryParam,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    chunksVdb: BaseVectorStorage?,
    textChunksDb: BaseKVStorage,
): ContextResult {
    // 1. Keyword extraction (simplified for now - just use query as keyword)
    val keywords = listOf(query)

    // 2. Search (Local/Global/Hybrid) - simplified to Local Search
    // Fetch related entities from entitiesVdb
    val entities = entitiesVdb.query(query, queryParam.topK)

    // Build context
    val contextBuilder = StringBuilder()
    contextBuilder.append(Prompts.KG_QUERY_CONTEXT)

    val entitiesStr =
        entities.joinToString("\n") {
            // Simplified entity string
            "{ \"entity_name\": \"${it["entity_name"]}\", \"content\": \"${
                JsonUtils.escape(it["content"]?.toString() ?: "")
            }\" }"
        }

    // Fetch Relations (Simplified implementation)
    // We can iterate over entities and get connected edges from knowledgeGraphInst
    val entityNames = entities.mapNotNull { it["entity_name"] as? String }
    val edges = mutableListOf<Map<String, String>>()
    val relationsSet = mutableSetOf<String>() // to avoid duplicates

    // Also collect chunk IDs
    val chunkIdsSet = mutableSetOf<String>()

    if (entityNames.isNotEmpty()) {
        val nodesData = knowledgeGraphInst.getNodesBatch(entityNames)

        nodesData.forEach { (nodeId, nodeData) ->
            // Collect chunk IDs from node source_id
            val sourceIds = nodeData["source_id"]?.toString()?.split(Constants.GRAPH_FIELD_SEP)
            if (sourceIds != null) {
                chunkIdsSet.addAll(sourceIds)
            }

            // Get edges for this node
            val nodeEdges = knowledgeGraphInst.getNodeEdges(nodeId)
            nodeEdges?.forEach { (src, tgt) ->
                val edgeKey = if (src < tgt) "$src#$tgt" else "$tgt#$src"
                if (!relationsSet.contains(edgeKey)) {
                    val edgeData = knowledgeGraphInst.getEdge(src, tgt)
                    if (edgeData != null) {
                        relationsSet.add(edgeKey)
                        edges.add(
                            mapOf(
                                "src_id" to src,
                                "tgt_id" to tgt,
                                "description" to (edgeData["description"] ?: ""),
                            ),
                        )
                    }
                }
            }
        }
    }

    val relationsStr =
        edges.take(queryParam.topK).joinToString("\n") {
            "{ \"src_id\": \"${it["src_id"]}\", \"tgt_id\": \"${it["tgt_id"]}\", " +
                "\"content\": \"${JsonUtils.escape(it["description"] ?: "")}\" }"
        }

    // Fetch Text Chunks
    val chunks =
        if (chunkIdsSet.isNotEmpty()) {
            if (chunksVdb != null) {
                chunksVdb.getByIds(chunkIdsSet.toList())
            } else {
                textChunksDb.getByIds(chunkIdsSet.toList())
            }
        } else {
            emptyList()
        }

    val textChunksStr =
        chunks.mapIndexed { index, chunk ->
            // We use index as reference ID in the prompt
            // content from chunksVdb or textChunksDb
            val content = chunk["content"]?.toString() ?: ""
            "{ \"reference_id\": \"${index + 1}\", \"content\": \"${JsonUtils.escape(content)}\" }"
        }.joinToString("\n")

    val referenceListStr =
        chunks.mapIndexed { index, chunk ->
            val filePath = chunk["file_path"] ?: "unknown_source"
            "[${index + 1}] $filePath"
        }.joinToString("\n")

    val contextContent =
        contextBuilder.toString()
            .replace("{entities_str}", entitiesStr)
            .replace("{relations_str}", relationsStr)
            .replace("{text_chunks_str}", textChunksStr)
            .replace("{reference_list_str}", referenceListStr)

    val rawData = mapOf(
        "entities" to entities,
        "relations" to edges,
        "chunks" to chunks,
    )
    return ContextResult(contextStr = contextContent, rawData = rawData)
}

suspend fun kgQuery(
    query: String,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage,
    queryParam: QueryParam,
    globalConfig: Map<String, Any?>,
    hashingKv: BaseKVStorage? = null,
    systemPrompt: String? = null,
    chunksVdb: BaseVectorStorage? = null,
    chatModel: ChatLanguageModel? = null,
): QueryResult? {
    if (query.isBlank()) {
        return QueryResult(content = Prompts.FAIL_RESPONSE)
    }

    val model = chatModel ?: globalConfig["llm_model_func"] as? ChatLanguageModel
    if (model == null) {
        logger.error { "No ChatLanguageModel provided for kgQuery" }
        return null
    }

    val contextResult = getContextStrForQuery(
        query,
        queryParam,
        knowledgeGraphInst,
        entitiesVdb,
        relationshipsVdb,
        chunksVdb,
        textChunksDb
    )
    val contextStr = contextResult.contextStr

    val sysPromptTemplate = systemPrompt ?: Prompts.RAG_RESPONSE
    val userPromptStr = queryParam.userPrompt?.let { "\n\n$it" } ?: "n/a"

    val sysPrompt = sysPromptTemplate
        .replace("{response_type}", queryParam.responseType ?: "Multiple Paragraphs")
        .replace("{user_prompt}", userPromptStr)
        .replace("{context_data}", contextStr)

    if (queryParam.onlyNeedContext) {
        return QueryResult(content = contextStr, rawData = contextResult.rawData)
    }

    if (queryParam.onlyNeedPrompt) {
        return QueryResult(content = "$sysPrompt\n\n---\n\n$query", rawData = contextResult.rawData)
    }

    val hlKeywordsStr = query
    val llKeywordsStr = ""
    val cacheSeed = listOf(
        queryParam.mode,
        query,
        queryParam.responseType ?: "",
        queryParam.topK.toString(),
        queryParam.chunkTopK.toString(),
        queryParam.maxEntityTokens.toString(),
        queryParam.maxRelationTokens.toString(),
        queryParam.maxTotalTokens.toString(),
        hlKeywordsStr,
        llKeywordsStr,
        queryParam.userPrompt ?: "",
        queryParam.enableRerank.toString(),
    ).joinToString("|")
    val argsHash = "kg_query_cache_${computeMd5(cacheSeed)}"

    if (hashingKv != null && globalConfig["enable_llm_cache"] as? Boolean == true) {
        val cached = hashingKv.getById(argsHash)
        val cachedContent = cached?.get("content") as? String
        if (!cachedContent.isNullOrEmpty()) {
            logger.info { " == LLM cache == Query cache hit, using cached response as query result" }
            return QueryResult(content = cachedContent, rawData = contextResult.rawData)
        }
    }

    if (queryParam.stream) {
        val streamingModel = model as? StreamingChatLanguageModel
        if (streamingModel == null) {
            logger.error { "Streaming is requested but the model does not support it." }
            return null
        }

        val responseIterator = flow {
            // This is a simplification. Langchain4j streaming needs a handler.
            // A real implementation would be more complex and might involve a custom handler.
            // For now, let's assume we can iterate over the stream of tokens.
            // The following is a placeholder for actual streaming logic.
            // The 'tokenStream' is not an iterable of tokens.
            // A correct implementation would look like this:
            /*
            val handler = object : StreamingResponseHandler<AiMessage> {
                override fun onNext(token: String) {
                    emit(token) // This is not correct with flow builder scope
                }
                override fun onComplete(response: Response<AiMessage>) {}
                override fun onError(error: Throwable) {}
            }
            streamingModel.generate(messages, handler)
            */
            // The above is complex to set up in a flow. A simpler (but maybe less efficient) way:
            val fullResponse = StringBuilder()
            val blockingQueue = java.util.concurrent.LinkedBlockingQueue<String>()
            val finalResponse = java.util.concurrent.CompletableFuture<Response<AiMessage>>()

            streamingModel.generate(
                listOf(SystemMessage(sysPrompt), UserMessage(query)),
                object : dev.langchain4j.model.StreamingResponseHandler<AiMessage> {
                    override fun onNext(token: String) {
                        blockingQueue.put(token)
                        fullResponse.append(token)
                    }

                    override fun onComplete(response: Response<AiMessage>) {
                        blockingQueue.put("___END___")
                        finalResponse.complete(response)
                    }

                    override fun onError(error: Throwable) {
                        blockingQueue.put("___END___")
                        finalResponse.completeExceptionally(error)
                    }
                }
            )

            while (true) {
                val token = blockingQueue.take()
                if (token == "___END___") break
                emit(token)
            }
            finalResponse.get() // wait for completion

            if (hashingKv != null && globalConfig["enable_llm_cache"] as? Boolean == true) {
                val queryParamDict = QueryParamCache(
                    mode = queryParam.mode,
                    responseType = queryParam.responseType,
                    topK = queryParam.topK,
                    chunkTopK = queryParam.chunkTopK,
                    maxEntityTokens = queryParam.maxEntityTokens,
                    maxRelationTokens = queryParam.maxRelationTokens,
                    maxTotalTokens = queryParam.maxTotalTokens,
                    hlKeywords = hlKeywordsStr,
                    llKeywords = llKeywordsStr,
                    userPrompt = queryParam.userPrompt ?: "",
                    enableRerank = queryParam.enableRerank,
                )
                saveToCache(
                    hashingKv,
                    CacheData(
                        argsHash = argsHash,
                        content = fullResponse.toString(),
                        prompt = query,
                        mode = queryParam.mode,
                        cacheType = "query",
                        queryParam = queryParamDict,
                    ),
                )
            }
        }
        return QueryResult(responseIterator = responseIterator, rawData = contextResult.rawData, isStreaming = true)
    } else {
        val responseText = try {
            model.generate(listOf(SystemMessage(sysPrompt), UserMessage(query))).content().text()
        } catch (e: Exception) {
            logger.error(e) { "Error generating response in kgQuery" }
            "Error generating response."
        }

        if (hashingKv != null && globalConfig["enable_llm_cache"] as? Boolean == true) {
            val queryParamDict = QueryParamCache(
                mode = queryParam.mode,
                responseType = queryParam.responseType,
                topK = queryParam.topK,
                chunkTopK = queryParam.chunkTopK,
                maxEntityTokens = queryParam.maxEntityTokens,
                maxRelationTokens = queryParam.maxRelationTokens,
                maxTotalTokens = queryParam.maxTotalTokens,
                hlKeywords = hlKeywordsStr,
                llKeywords = llKeywordsStr,
                userPrompt = queryParam.userPrompt ?: "",
                enableRerank = queryParam.enableRerank,
            )
            saveToCache(
                hashingKv,
                CacheData(
                    argsHash = argsHash,
                    content = responseText,
                    prompt = query,
                    mode = queryParam.mode,
                    cacheType = "query",
                    queryParam = queryParamDict,
                ),
            )
        }

        var responseContent = responseText
        if (responseContent.length > sysPrompt.length) {
            responseContent = responseContent
                .replace(sysPrompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .trim()
        }
        return QueryResult(content = responseContent, rawData = contextResult.rawData)
    }
}

suspend fun naiveQuery(
    query: String,
    chunksVdb: BaseVectorStorage,
    queryParam: QueryParam,
    globalConfig: Map<String, Any?>,
    hashingKv: BaseKVStorage? = null,
    systemPrompt: String? = null,
    chatModel: ChatLanguageModel? = null,
): String? {
    if (query.isBlank()) {
        return Prompts.FAIL_RESPONSE
    }

    // Basic vector search (naive: direct topK from chunks VDB)
    val searchTopK = queryParam.chunkTopK.coerceAtMost(queryParam.topK)

    val results = chunksVdb.query(query, searchTopK)
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

    val sysPromptTemplate = systemPrompt ?: Prompts.NAIVE_RAG_RESPONSE

    val userPrompt =
        buildString {
            if (!queryParam.userPrompt.isNullOrBlank()) {
                append(queryParam.userPrompt)
            } else {
                append("n/a")
            }
            if (!queryParam.responseType.isNullOrBlank()) {
                append("\n\n")
                append(queryParam.responseType)
            }
        }

    val sysPrompt =
        sysPromptTemplate
            .replace(
                "{response_type}",
                queryParam.responseType ?: "Multiple Paragraphs",
            )
            .replace("{user_prompt}", userPrompt)
            .replace("{content_data}", contextContent)

    if (queryParam.onlyNeedContext) {
        return contextContent
    }

    if (queryParam.onlyNeedPrompt) {
        return listOf(sysPrompt, "---User Query---", query).joinToString("\n")
    }

    // Call LLM
    val model = chatModel ?: globalConfig["llm_model_func"] as? ChatLanguageModel

    if (model == null) {
        logger.error { "No ChatLanguageModel provided for naiveQuery" }
        return "Error: No LLM model configured."
    }

    // Build cache key
    val cacheKeySeed =
        listOf(
            queryParam.mode,
            query,
            queryParam.responseType ?: "",
            queryParam.topK.toString(),
            queryParam.chunkTopK.toString(),
            queryParam.userPrompt ?: "",
            queryParam.enableRerank.toString(),
        ).joinToString("|")
    val cacheKey = "query_cache_${computeMd5(cacheKeySeed)}"

    // Try cache if provided
    if (hashingKv != null) {
        val cached = hashingKv.getById(cacheKey)
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
                UserMessage(query),
            )
        val response: AiMessage = model.generate(messages).content()
        val text = response.text()

        if (hashingKv != null) {
            hashingKv.upsert(mapOf(cacheKey to mapOf("content" to text)))
        }
        text
    } catch (e: Exception) {
        logger.error(e) { "Error generating response in naiveQuery" }
        "Error generating response."
    }
}
