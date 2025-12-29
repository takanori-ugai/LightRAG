package lightrag.operate

import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.service.AiServices
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.CacheData
import lightrag.core.Constants
import lightrag.core.QueryParam
import lightrag.core.QueryParamCache
import lightrag.core.QueryResult
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.llm.KeywordExtractor
import lightrag.utils.JsonUtils
import lightrag.utils.Prompts
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

/**
 * The result of a context retrieval operation.
 * @property contextStr The context string.
 * @property rawData The raw data of the context.
 */
data class ContextResult(
    val contextStr: String,
    val rawData: Map<String, Any?>?,
)

/**
 * The result of a keywords extraction operation.
 * @property highLevelKeywords A list of high-level keywords.
 * @property lowLevelKeywords A list of low-level keywords.
 */
@Serializable
data class KeywordsExtractionResult(
    @SerialName("highLevelKeywords")
    val highLevelKeywords: List<String> = emptyList(),
    @SerialName("lowLevelKeywords")
    val lowLevelKeywords: List<String> = emptyList(),
)

/**
 * An extracted entity.
 * @property name The name of the entity.
 * @property type The type of the entity.
 * @property description A description of the entity.
 */
@Serializable
data class ExtractedEntity(
    val name: String = "",
    val type: String = "",
    val description: String = "",
)

/**
 * An extracted relation.
 * @property source The source entity of the relation.
 * @property target The target entity of the relation.
 * @property keywords Keywords associated with the relation.
 * @property description A description of the relation.
 */
@Serializable
data class ExtractedRelation(
    val source: String = "",
    val target: String = "",
    val keywords: String = "",
    val description: String = "",
)

/**
 * The result of an extraction operation.
 * @property entities A list of extracted entities.
 * @property relations A list of extracted relations.
 */
@Serializable
data class ExtractionResult(
    val entities: List<ExtractedEntity> = emptyList(),
    val relations: List<ExtractedRelation> = emptyList(),
)

/**
 * The result of getting node data.
 * @property nodeDatas A list of node data maps.
 * @property useRelations A list of relation data maps.
 */
data class GetNodeDataResult(
    val nodeDatas: List<Map<String, Any>>,
    val useRelations: List<Map<String, Any>>,
)

/**
 * The result of getting edge data.
 * @property edgeDatas A list of edge data maps.
 * @property useEntities A list of entity data maps.
 */
data class GetEdgeDataResult(
    val edgeDatas: List<Map<String, Any>>,
    val useEntities: List<Map<String, Any>>,
)

/**
 * The result of a knowledge graph search.
 * @property finalEntities A list of final entities.
 * @property finalRelations A list of final relations.
 * @property vectorChunks A list of vector chunks.
 */
data class PerformKgSearchResult(
    val finalEntities: List<Map<String, Any>>,
    val finalRelations: List<Map<String, Any>>,
    val vectorChunks: List<Map<String, Any>>,
)

class QueryProcessor(
    private val knowledgeGraphInst: BaseGraphStorage,
    private val entitiesVdb: BaseVectorStorage,
    private val relationshipsVdb: BaseVectorStorage,
    private val textChunksDb: BaseKVStorage,
    private val chatModel: ChatModel,
    private val hashingKv: BaseKVStorage?,
    private val globalConfig: Map<String, Any?>,
    private val tokenizer: (String) -> List<Int>,
    private val decoder: (List<Int>) -> String,
) {
    suspend fun kgQuery(
        query: String,
        queryParam: QueryParam,
        systemPrompt: String? = null,
        chunksVdb: BaseVectorStorage? = null,
    ): QueryResult? {
        if (query.isBlank()) {
            return QueryResult(content = Prompts.FAIL_RESPONSE)
        }

        // `chatModel` is guaranteed to be non-null by the constructor, so this check is redundant
        // if (chatModel == null) {
        // logger.error { "No ChatModel provided for kgQuery" }
        // return null
        // }

        val (hlKeywords, llKeywords) = getKeywordsFromQuery(query, queryParam)
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
                chunksVdb,
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
            val streamingModel = chatModel as? StreamingChatModel
            if (streamingModel == null) {
                logger.error { "Streaming is requested but the model does not support it." }
                return null
            }

            logger.trace { "SysPrompt :$sysPrompt" }
            logger.trace { "UserQuery :$query" }
            val responseIterator =
                flow {
                    val fullResponse = StringBuilder()
                    val blockingQueue = java.util.concurrent.LinkedBlockingQueue<String>()
                    val finalResponse = java.util.concurrent.CompletableFuture<ChatResponse>()

                    streamingModel.chat(
                        listOf(SystemMessage(sysPrompt), UserMessage(query)),
                        object : StreamingChatResponseHandler {
                            override fun onPartialResponse(partialResponse: String) {
                                blockingQueue.put(partialResponse)
                                fullResponse.append(partialResponse)
                            }

                            override fun onCompleteResponse(response: ChatResponse) {
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
                    logger.trace { "SysPrompt :$sysPrompt" }
                    logger.trace { "UserQuery :$query" }
                    val chatResponse = chatModel.chat(listOf(SystemMessage(sysPrompt), UserMessage(query)))
                    chatResponse.aiMessage()?.text() ?: ""
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
    ): Pair<List<String>, List<String>> {
        if (queryParam.hlKeywords.isNotEmpty() || queryParam.llKeywords.isNotEmpty()) {
            return queryParam.hlKeywords to queryParam.llKeywords
        }
        return extractKeywordsOnly(query, queryParam)
    }

    @Suppress("UNUSED_PARAMETER")
    private suspend fun extractKeywordsOnly(
        text: String,
        param: QueryParam,
    ): Pair<List<String>, List<String>> {
        val model = chatModel
        if (model == null) {
            logger.error { "No ChatModel provided for keyword extraction" }
            return emptyList<String>() to emptyList()
        }

        val language = globalConfig["language"] as? String ?: "English"
        val examples = globalConfig["keyword_examples"] as? String ?: ""

        val keywordExtractor = AiServices.create(KeywordExtractor::class.java, model)

        return try {
            val keywordsResult = keywordExtractor.extract(text, language, examples)
            keywordsResult.highLevelKeywords to keywordsResult.lowLevelKeywords
        } catch (e: Exception) {
            logger.error(e) { "Failed to parse keywords from LLM response" }
            emptyList<String>() to emptyList()
        }
    }

    private suspend fun getContextStrForQuery(
        query: String,
        queryParam: QueryParam,
        chunksVdb: BaseVectorStorage?,
    ): ContextResult {
        val searchResult =
            performKgSearch(
                query,
                queryParam,
                chunksVdb,
            )

        val entities = searchResult.finalEntities
        val relations = searchResult.finalRelations

        val entityChunks =
            findRelatedTextUnitFromEntities(
                entities,
                queryParam,
                query,
                chunksVdb,
            )
        val relationChunks =
            findRelatedTextUnitFromRelations(
                relations,
                queryParam,
                entityChunks,
                query,
                chunksVdb,
            )
        var allChunks = (entityChunks + relationChunks).distinctBy { it["id"] }

        // Fallback: if graph search returns nothing, try direct chunk vector search
        if (allChunks.isEmpty() && chunksVdb != null) {
            logger.info { "No graph matches found; falling back to chunk vector search." }
            val chunkHits =
                chunksVdb.query(query, queryParam.chunkTopK).map {
                    mapOf(
                        "id" to (it["id"] ?: ""),
                        "content" to (it["content"] ?: ""),
                        "file_path" to (it["file_path"] ?: "unknown_source"),
                        "score" to (it["score"] ?: it["distance"] ?: 0.0),
                    )
                }
            allChunks = chunkHits
        }

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
            allChunks
                .mapIndexed {
                    index,
                    chunk,
                    ->
                    val content = chunk["content"]?.toString() ?: ""
                    """{ "reference_id": "${index + 1}", "content": "${JsonUtils.escape(content)}" }"""
                }.joinToString("\n")

        val referenceListStr =
            allChunks
                .mapIndexed {
                    index,
                    chunk,
                    ->
                    val filePath = chunk["file_path"] ?: "unknown_source"
                    "[${index + 1}] $filePath"
                }.joinToString("\n")

        val contextContent =
            contextBuilder
                .toString()
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

    @Suppress("UNUSED_PARAMETER")
    private suspend fun performKgSearch(
        query: String,
        queryParam: QueryParam,
        chunksVdb: BaseVectorStorage?,
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

        val useRelations = findMostRelatedEdgesFromEntities(nodeDatas, queryParam)
        logger.info { "Local query: ${nodeDatas.size} entities, ${useRelations.size} relations" }
        return GetNodeDataResult(nodeDatas, useRelations)
    }

    private suspend fun getEdgeData(
        keywords: String,
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
            results.mapNotNull {
                val srcId = it["src_id"] as? String
                val tgtId = it["tgt_id"] as? String
                if (srcId != null && tgtId != null) {
                    mapOf("src" to srcId, "tgt" to tgtId)
                } else {
                    null
                }
            }
        val edgeDataDict = knowledgeGraphInst.getEdgesBatch(edgePairs)

        val edgeDatas =
            results.mapNotNull {
                val srcId = it["src_id"] as? String
                val tgtId = it["tgt_id"] as? String
                if (srcId != null && tgtId != null) {
                    val pair = if (srcId < tgtId) srcId to tgtId else tgtId to srcId
                    edgeDataDict[pair]?.let {
                        val mutableEdgeProps = it.toMutableMap()
                        if (!mutableEdgeProps.containsKey("weight")) {
                            logger.warn { "Edge $pair missing 'weight' attribute, using default value 1.0" }
                            mutableEdgeProps["weight"] = "1.0"
                        }
                        mutableEdgeProps["src_id"] = srcId
                        mutableEdgeProps["tgt_id"] = tgtId
                        mutableEdgeProps["created_at"] = it["created_at"]?.toString() ?: ""
                        mutableEdgeProps
                    }
                } else {
                    null
                }
            }

        val useEntities = findMostRelatedEntitiesFromRelationships(edgeDatas, queryParam)
        logger.info { "Global query: ${useEntities.size} entities, ${edgeDatas.size} relations" }
        return GetEdgeDataResult(edgeDatas, useEntities)
    }

    @Suppress("UNUSED_PARAMETER")
    private suspend fun findMostRelatedEdgesFromEntities(
        nodeDatas: List<Map<String, Any>>,
        queryParam: QueryParam,
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
            }.thenByDescending {
                (it["weight"] as? Number)?.toDouble() ?: 0.0
            },
        )
    }

    @Suppress("UNUSED_PARAMETER")
    private suspend fun findMostRelatedEntitiesFromRelationships(
        edgeDatas: List<Map<String, Any>>,
        queryParam: QueryParam,
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
        query: String? = null,
        chunksVdb: BaseVectorStorage? = null,
    ): List<Map<String, Any>> {
        logger.debug { "Finding text chunks from ${nodeDatas.size} entities" }
        if (nodeDatas.isEmpty()) {
            return emptyList()
        }

        val entitiesWithChunks =
            nodeDatas.mapNotNull {
                (it["source_id"] as? String)?.let { sourceId ->
                    val chunks = sourceId.split(Constants.GRAPH_FIELD_SEP).filter { it.isNotEmpty() }
                    if (chunks.isNotEmpty()) {
                        mapOf(
                            "entity_name" to it["entity_name"],
                            "chunks" to chunks,
                            "entity_data" to it,
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
            edgeDatas.mapNotNull {
                (it["source_id"] as? String)?.let { sourceId ->
                    val chunks =
                        sourceId.split(Constants.GRAPH_FIELD_SEP).filter { it.isNotEmpty() && !entityChunkIds.contains(it) }
                    if (chunks.isNotEmpty()) {
                        mapOf(
                            "relation_key" to (it["src_id"] as String to it["tgt_id"] as String),
                            "chunks" to chunks,
                            "relation_data" to it,
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
}
