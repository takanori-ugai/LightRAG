package lightrag.operate

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.chat.StreamingChatLanguageModel
import dev.langchain4j.model.output.Response
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import lightrag.core.CacheData
import lightrag.core.Constants
import lightrag.core.QueryParam
import lightrag.core.QueryParamCache
import lightrag.core.QueryResult
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

@Serializable
data class KeywordsExtractionResult(
    val high_level_keywords: List<String>,
    val low_level_keywords: List<String>,
)

data class GetNodeDataResult(
    val nodeDatas: List<Map<String, Any>>,
    val useRelations: List<Map<String, Any>>,
)

data class GetEdgeDataResult(
    val edgeDatas: List<Map<String, Any>>,
    val useEntities: List<Map<String, Any>>,
)

data class PerformKgSearchResult(
    val finalEntities: List<Map<String, Any>>,
    val finalRelations: List<Map<String, Any>>,
    val vectorChunks: List<Map<String, Any>>,
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
    val searchResult =
        performKgSearch(
            query,
            queryParam,
            knowledgeGraphInst,
            entitiesVdb,
            relationshipsVdb,
            chunksVdb,
            textChunksDb,
        )

    val entities = searchResult.finalEntities
    val relations = searchResult.finalRelations

    val entityChunks =
        findRelatedTextUnitFromEntities(
            entities,
            queryParam,
            textChunksDb,
            knowledgeGraphInst,
            query,
            chunksVdb,
        )
    val relationChunks =
        findRelatedTextUnitFromRelations(
            relations,
            queryParam,
            textChunksDb,
            entityChunks,
            query,
            chunksVdb,
        )
    val allChunks = (entityChunks + relationChunks).distinctBy { it["id"] }

    val contextBuilder = StringBuilder()
    contextBuilder.append(Prompts.KG_QUERY_CONTEXT)

    val entitiesStr =
        entities.joinToString("\n") { entity ->
            """{ "entity_name": "${entity["entity_name"]}", "content": "${
                JsonUtils.escape((entity["content"] ?: "").toString())
            }" }"""
        }

    val relationsStr =
        relations.take(queryParam.topK).joinToString("\n") { relation ->
            """{ "src_id": "${relation["src_id"]}", "tgt_id": "${relation["tgt_id"]}", "content": "${
                JsonUtils.escape((relation["description"] ?: "").toString())
            }" }"""
        }

    val textChunksStr =
        allChunks.mapIndexed { index, chunk ->
            val content = chunk["content"]?.toString() ?: ""
            """{ "reference_id": "${index + 1}", "content": "${JsonUtils.escape(content)}" }"""
        }.joinToString("\n")

    val referenceListStr =
        allChunks.mapIndexed { index, chunk ->
            val filePath = chunk["file_path"] ?: "unknown_source"
            "[${index + 1}] $filePath"
        }.joinToString("\n")

    val contextContent =
        contextBuilder.toString()
            .replace("{entities_str}", entitiesStr)
            .replace("{relations_str}", relationsStr)
            .replace("{text_chunks_str}", textChunksStr)
            .replace("{reference_list_str}", referenceListStr)

    val rawData =
        mapOf(
            "entities" to entities,
            "relations" to relations,
            "chunks" to allChunks,
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

    val (hlKeywords, llKeywords) = getKeywordsFromQuery(query, queryParam, globalConfig, hashingKv)
    queryParam.hlKeywords = hlKeywords
    queryParam.llKeywords = llKeywords

    if (llKeywords.isEmpty() && queryParam.mode in listOf("local", "hybrid", "mix")) {
        logger.warn { "low_level_keywords is empty" }
    }
    if (hlKeywords.isEmpty() && queryParam.mode in listOf("global", "hybrid", "mix")) {
        logger.warn { "high_level_keywords is empty" }
    }
    if (hlKeywords.isEmpty() && llKeywords.isEmpty()) {
        if (query.length < 50) {
            logger.warn { "Forced low_level_keywords to origin query: $query" }
            queryParam.llKeywords = listOf(query)
        } else {
            return QueryResult(content = Prompts.FAIL_RESPONSE)
        }
    }

    val contextResult =
        getContextStrForQuery(
            query,
            queryParam,
            knowledgeGraphInst,
            entitiesVdb,
            relationshipsVdb,
            chunksVdb,
            textChunksDb,
        )
    val contextStr = contextResult.contextStr

    val sysPromptTemplate = systemPrompt ?: Prompts.RAG_RESPONSE
    val userPromptStr = queryParam.userPrompt?.let { "\n\n$it" } ?: "n/a"

    val sysPrompt =
        sysPromptTemplate
            .replace("{response_type}", queryParam.responseType ?: "Multiple Paragraphs")
            .replace("{user_prompt}", userPromptStr)
            .replace("{context_data}", contextStr)

    if (queryParam.onlyNeedContext) {
        return QueryResult(content = contextStr, rawData = contextResult.rawData)
    }

    if (queryParam.onlyNeedPrompt) {
        return QueryResult(content = "$sysPrompt\n\n---\n\n$query", rawData = contextResult.rawData)
    }

    val hlKeywordsStr = queryParam.hlKeywords.joinToString(", ")
    val llKeywordsStr = queryParam.llKeywords.joinToString(", ")
    val cacheSeed =
        listOf(
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

        val responseIterator =
            flow {
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
                    },
                )

                while (true) {
                    val token = blockingQueue.take()
                    if (token == "___END___") break
                    emit(token)
                }
                finalResponse.get() // wait for completion

                if (hashingKv != null && globalConfig["enable_llm_cache"] as? Boolean == true) {
                    val queryParamDict =
                        QueryParamCache(
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
        val responseText =
            try {
                model.generate(listOf(SystemMessage(sysPrompt), UserMessage(query))).content().text()
            } catch (e: Exception) {
                logger.error(e) { "Error generating response in kgQuery" }
                "Error generating response."
            }

        if (hashingKv != null && globalConfig["enable_llm_cache"] as? Boolean == true) {
            val queryParamDict =
                QueryParamCache(
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
            responseContent =
                responseContent
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

private suspend fun getKeywordsFromQuery(
    query: String,
    queryParam: QueryParam,
    globalConfig: Map<String, Any?>,
    hashingKv: BaseKVStorage?,
): Pair<List<String>, List<String>> {
    if (queryParam.hlKeywords.isNotEmpty() || queryParam.llKeywords.isNotEmpty()) {
        return queryParam.hlKeywords to queryParam.llKeywords
    }
    return extractKeywordsOnly(query, queryParam, globalConfig, hashingKv)
}

@Suppress("UNUSED_PARAMETER")
private suspend fun extractKeywordsOnly(
    text: String,
    param: QueryParam,
    globalConfig: Map<String, Any?>,
    hashingKv: BaseKVStorage?,
): Pair<List<String>, List<String>> {
    val examples = Prompts.KEYWORDS_EXTRACTION_EXAMPLES.joinToString("\n")
    val language = globalConfig["language"] as? String ?: "English"
    val kwPrompt =
        Prompts.KEYWORDS_EXTRACTION
            .replace("{query}", text)
            .replace("{examples}", examples)
            .replace("{language}", language)

    val model = globalConfig["llm_model_func"] as? ChatLanguageModel
    if (model == null) {
        logger.error { "No ChatLanguageModel provided for keyword extraction" }
        return emptyList<String>() to emptyList()
    }
    val result = model.generate(listOf(UserMessage(kwPrompt))).content().text()
    return try {
        val json =
            Json {
                ignoreUnknownKeys = true
                isLenient = true
            }
        val keywordsResult = json.decodeFromString<KeywordsExtractionResult>(result)
        keywordsResult.high_level_keywords to keywordsResult.low_level_keywords
    } catch (e: Exception) {
        logger.error(e) { "Failed to parse keywords from LLM response" }
        emptyList<String>() to emptyList()
    }
}

@Suppress("UNUSED_PARAMETER")
private suspend fun performKgSearch(
    query: String,
    queryParam: QueryParam,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    chunksVdb: BaseVectorStorage?,
    textChunksDb: BaseKVStorage,
): PerformKgSearchResult =
    coroutineScope {
        var localEntities = emptyList<Map<String, Any>>()
        var localRelations = emptyList<Map<String, Any>>()
        var globalEntities = emptyList<Map<String, Any>>()
        var globalRelations = emptyList<Map<String, Any>>()
        var vectorChunks = emptyList<Map<String, Any>>()

        val localSearch =
            async {
                if (queryParam.mode in listOf("local", "hybrid", "mix") && queryParam.llKeywords.isNotEmpty()) {
                    val (nodeDatas, useRelations) =
                        getNodeData(
                            queryParam.llKeywords.joinToString(", "),
                            knowledgeGraphInst,
                            entitiesVdb,
                            queryParam,
                        )
                    localEntities = nodeDatas
                    localRelations = useRelations
                }
            }

        val globalSearch =
            async {
                if (queryParam.mode in listOf("global", "hybrid", "mix") && queryParam.hlKeywords.isNotEmpty()) {
                    val (edgeDatas, useEntities) =
                        getEdgeData(
                            queryParam.hlKeywords.joinToString(", "),
                            knowledgeGraphInst,
                            relationshipsVdb,
                            queryParam,
                        )
                    globalRelations = edgeDatas
                    globalEntities = useEntities
                }
            }

        val vectorSearch =
            async {
                if (queryParam.mode == "mix" && chunksVdb != null) {
                    vectorChunks = chunksVdb.query(query, queryParam.chunkTopK)
                }
            }

        awaitAll(localSearch, globalSearch, vectorSearch)

        val finalEntities = (localEntities + globalEntities).distinctBy { it["entity_name"] }
        val finalRelations = (localRelations + globalRelations).distinctBy { it["src_id"] to it["tgt_id"] }

        PerformKgSearchResult(finalEntities, finalRelations, vectorChunks)
    }

private suspend fun getNodeData(
    query: String,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    queryParam: QueryParam,
): GetNodeDataResult {
    logger.info {
        "Query nodes: $query (top_k:${queryParam.topK}, cosine:${entitiesVdb.cosineBetterThanThreshold})"
    }
    val results = entitiesVdb.query(query, queryParam.topK)
    if (results.isEmpty()) {
        return GetNodeDataResult(emptyList(), emptyList())
    }
    val nodeIds = results.mapNotNull { it["entity_name"] as? String }
    val nodesDict = knowledgeGraphInst.getNodesBatch(nodeIds)
    val degreesDict = knowledgeGraphInst.nodeDegreesBatch(nodeIds)

    val nodeDatas =
        results.mapNotNull { r ->
            val nodeId = r["entity_name"] as? String
            if (nodeId != null) {
                nodesDict[nodeId]?.let { n ->
                    val mutableN = n.toMutableMap()
                    mutableN["entity_name"] = nodeId
                    mutableN["rank"] = degreesDict[nodeId]?.toString() ?: "0"
                    mutableN["created_at"] = r["created_at"]?.toString() ?: ""
                    mutableN
                }
            } else {
                null
            }
        }

    val useRelations = findMostRelatedEdgesFromEntities(nodeDatas, queryParam, knowledgeGraphInst)
    logger.info { "Local query: ${nodeDatas.size} entities, ${useRelations.size} relations" }
    return GetNodeDataResult(nodeDatas, useRelations)
}

private suspend fun getEdgeData(
    keywords: String,
    knowledgeGraphInst: BaseGraphStorage,
    relationshipsVdb: BaseVectorStorage,
    queryParam: QueryParam,
): GetEdgeDataResult {
    logger.info {
        "Query edges: $keywords (top_k:${queryParam.topK}, cosine:${relationshipsVdb.cosineBetterThanThreshold})"
    }
    val results = relationshipsVdb.query(keywords, queryParam.topK)
    if (results.isEmpty()) {
        return GetEdgeDataResult(emptyList(), emptyList())
    }

    val edgePairs =
        results.mapNotNull { r ->
            val srcId = r["src_id"] as? String
            val tgtId = r["tgt_id"] as? String
            if (srcId != null && tgtId != null) {
                mapOf("src" to srcId, "tgt" to tgtId)
            } else {
                null
            }
        }
    val edgeDataDict = knowledgeGraphInst.getEdgesBatch(edgePairs)

    val edgeDatas =
        results.mapNotNull { k ->
            val srcId = k["src_id"] as? String
            val tgtId = k["tgt_id"] as? String
            if (srcId != null && tgtId != null) {
                val pair = if (srcId < tgtId) srcId to tgtId else tgtId to srcId
                edgeDataDict[pair]?.let { edgeProps ->
                    val mutableEdgeProps = edgeProps.toMutableMap()
                    if (!mutableEdgeProps.containsKey("weight")) {
                        logger.warn { "Edge $pair missing 'weight' attribute, using default value 1.0" }
                        mutableEdgeProps["weight"] = "1.0"
                    }
                    mutableEdgeProps["src_id"] = srcId
                    mutableEdgeProps["tgt_id"] = tgtId
                    mutableEdgeProps["created_at"] = k["created_at"]?.toString() ?: ""
                    mutableEdgeProps
                }
            } else {
                null
            }
        }

    val useEntities = findMostRelatedEntitiesFromRelationships(edgeDatas, queryParam, knowledgeGraphInst)
    logger.info { "Global query: ${useEntities.size} entities, ${edgeDatas.size} relations" }
    return GetEdgeDataResult(edgeDatas, useEntities)
}

@Suppress("UNUSED_PARAMETER")
private suspend fun findMostRelatedEdgesFromEntities(
    nodeDatas: List<Map<String, Any>>,
    queryParam: QueryParam,
    knowledgeGraphInst: BaseGraphStorage,
): List<Map<String, Any>> {
    val nodeNames = nodeDatas.mapNotNull { it["entity_name"] as? String }
    val batchEdgesDict = knowledgeGraphInst.getNodesEdgesBatch(nodeNames)

    val allEdges = mutableSetOf<Pair<String, String>>()
    for (nodeName in nodeNames) {
        val thisEdges = batchEdgesDict[nodeName] ?: emptyList()
        for (edge in thisEdges) {
            val sortedEdge = if (edge.first < edge.second) edge else edge.second to edge.first
            allEdges.add(sortedEdge)
        }
    }

    val edgePairs = allEdges.map { mapOf("src" to it.first, "tgt" to it.second) }
    val edgeDataDict = knowledgeGraphInst.getEdgesBatch(edgePairs)
    val edgeDegreesDict = knowledgeGraphInst.edgeDegreesBatch(allEdges.toList())

    val allEdgesData = mutableListOf<Map<String, Any>>()
    for (pair in allEdges) {
        val edgeProps = edgeDataDict[pair]
        if (edgeProps != null) {
            val weight = (edgeProps["weight"] as? String)?.toDoubleOrNull() ?: 1.0
            val combined = mutableMapOf<String, Any>()
            combined["src_tgt"] = pair
            combined["rank"] = edgeDegreesDict[pair] ?: 0
            combined.putAll(edgeProps)
            combined["weight"] = weight
            allEdgesData.add(combined)
        }
    }

    return allEdgesData.sortedWith(
        compareByDescending<Map<String, Any>> {
            (it["rank"] as? Number)?.toInt() ?: 0
        }.thenByDescending { (it["weight"] as? Number)?.toDouble() ?: 0.0 },
    )
}

@Suppress("UNUSED_PARAMETER")
private suspend fun findMostRelatedEntitiesFromRelationships(
    edgeDatas: List<Map<String, Any>>,
    queryParam: QueryParam,
    knowledgeGraphInst: BaseGraphStorage,
): List<Map<String, Any>> {
    val entityNames = mutableSetOf<String>()
    for (edge in edgeDatas) {
        (edge["src_id"] as? String)?.let { entityNames.add(it) }
        (edge["tgt_id"] as? String)?.let { entityNames.add(it) }
    }

    val nodesDict = knowledgeGraphInst.getNodesBatch(entityNames.toList())

    val nodeDatas = mutableListOf<Map<String, Any>>()
    for (entityName in entityNames) {
        val node = nodesDict[entityName]
        if (node != null) {
            val combined = mutableMapOf<String, Any>()
            combined.putAll(node)
            combined["entity_name"] = entityName
            nodeDatas.add(combined)
        }
    }
    return nodeDatas
}

@Suppress("UNUSED_PARAMETER")
private suspend fun findRelatedTextUnitFromEntities(
    nodeDatas: List<Map<String, Any>>,
    queryParam: QueryParam,
    textChunksDb: BaseKVStorage,
    knowledgeGraphInst: BaseGraphStorage,
    query: String? = null,
    chunksVdb: BaseVectorStorage? = null,
): List<Map<String, Any>> {
    logger.debug { "Finding text chunks from ${nodeDatas.size} entities" }
    if (nodeDatas.isEmpty()) {
        return emptyList()
    }

    val entitiesWithChunks =
        nodeDatas.mapNotNull { entity ->
            (entity["source_id"] as? String)?.let { sourceId ->
                val chunks = sourceId.split(Constants.GRAPH_FIELD_SEP).filter { it.isNotEmpty() }
                if (chunks.isNotEmpty()) {
                    mapOf(
                        "entity_name" to entity["entity_name"],
                        "chunks" to chunks,
                        "entity_data" to entity,
                    )
                } else {
                    null
                }
            }
        }

    if (entitiesWithChunks.isEmpty()) {
        logger.warn { "No entities with text chunks found" }
        return emptyList()
    }

    val allChunkIds = entitiesWithChunks.flatMap { it["chunks"] as List<String> }.distinct()

    return textChunksDb.getByIds(allChunkIds)
}

@Suppress("UNUSED_PARAMETER")
private suspend fun findRelatedTextUnitFromRelations(
    edgeDatas: List<Map<String, Any>>,
    queryParam: QueryParam,
    textChunksDb: BaseKVStorage,
    entityChunks: List<Map<String, Any>> = emptyList(),
    query: String? = null,
    chunksVdb: BaseVectorStorage? = null,
): List<Map<String, Any>> {
    logger.debug { "Finding text chunks from ${edgeDatas.size} relations" }
    if (edgeDatas.isEmpty()) {
        return emptyList()
    }

    val entityChunkIds = entityChunks.mapNotNull { it["id"] as? String }.toSet()

    val relationsWithChunks =
        edgeDatas.mapNotNull { relation ->
            (relation["source_id"] as? String)?.let { sourceId ->
                val chunks =
                    sourceId.split(Constants.GRAPH_FIELD_SEP).filter { it.isNotEmpty() && !entityChunkIds.contains(it) }
                if (chunks.isNotEmpty()) {
                    mapOf(
                        "relation_key" to (relation["src_id"] as String to relation["tgt_id"] as String),
                        "chunks" to chunks,
                        "relation_data" to relation,
                    )
                } else {
                    null
                }
            }
        }

    if (relationsWithChunks.isEmpty()) {
        logger.info { "Find no additional relations-related chunks from ${edgeDatas.size} relations" }
        return emptyList()
    }

    val allChunkIds = relationsWithChunks.flatMap { it["chunks"] as List<String> }.distinct()

    return textChunksDb.getByIds(allChunkIds)
}
