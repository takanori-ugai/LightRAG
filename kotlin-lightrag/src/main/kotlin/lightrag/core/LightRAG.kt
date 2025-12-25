package lightrag.core

import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.embedding.EmbeddingModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.core.types.DocStatus
import lightrag.core.types.DocStatusStorage
import lightrag.kg.json.JsonDocStatusStorage
import lightrag.kg.json.JsonKVStorage
import lightrag.kg.memory.InMemoryGraphStorage
import lightrag.kg.memory.InMemoryVectorStorage
import lightrag.llm.LLMFactory
import lightrag.operate.NaiveQueryParams
import lightrag.operate.chunkingByTokenSize
import lightrag.operate.extractEntities
import lightrag.operate.kgQuery
import lightrag.operate.mergeNodesAndEdges
import lightrag.operate.naiveQuery
import lightrag.utils.computeMd5
import lightrag.utils.generateTrackId
import java.time.Instant

private val logger = KotlinLogging.logger {}

@Serializable
data class QueryParam(
    val mode: String = "global",
    @SerialName("only_need_context")
    val onlyNeedContext: Boolean = false,
    @SerialName("only_need_prompt")
    val onlyNeedPrompt: Boolean = false,
    @SerialName("response_type")
    val responseType: String? = "Multiple Paragraphs",
    val stream: Boolean = false,
    @SerialName("top_k")
    val topK: Int = 40,
    @SerialName("chunk_top_k")
    val chunkTopK: Int = 20,
    @SerialName("max_entity_tokens")
    val maxEntityTokens: Int = 6000,
    @SerialName("max_relation_tokens")
    val maxRelationTokens: Int = 8000,
    @SerialName("max_total_tokens")
    val maxTotalTokens: Int = 30000,
    @SerialName("hl_keywords")
    var hlKeywords: List<String> = emptyList(),
    @SerialName("ll_keywords")
    var llKeywords: List<String> = emptyList(),
    @SerialName("conversation_history")
    val conversationHistory: List<Map<String, String>> = emptyList(),
    @SerialName("user_prompt")
    val userPrompt: String? = null,
    @SerialName("enable_rerank")
    val enableRerank: Boolean = true,
    @SerialName("include_references")
    val includeReferences: Boolean = false,
)

class LightRAG(
    val workingDir: String = "./rag_storage",
    chatModel: ChatLanguageModel? = null,
    embeddingModel: EmbeddingModel? = null,
    val hashingKv: BaseKVStorage? = null,
    llmBinding: String = "ollama",
    llmModelName: String = "llama3",
    embeddingBinding: String = "ollama",
    embeddingModelName: String = "all-minilm",
    graphStorageName: String = "InMemoryGraphStorage",
    addonConfig: AddonConfig = AddonConfig(),
) {
    val chatModel: ChatLanguageModel =
        chatModel ?: LLMFactory.createChatModel(llmBinding, llmModelName)

    private val embedding: EmbeddingModel =
        embeddingModel ?: LLMFactory.createEmbeddingModel(embeddingBinding, embeddingModelName)

    val globalConfig: Map<String, Any?> =
        mapOf(
            "llm_model_func" to chatModel,
            "embedding_func" to embedding,
            "tokenizer" to { text: String -> text.split(Regex("\\s+")).map { it.hashCode() } },
            "chunk_token_size" to 1200,
            "chunk_overlap_token_size" to 100,
            "entity_types" to listOf("Person", "Organization", "Location", "Event", "Concept"),
            "language" to "English",
            "working_dir" to workingDir,
            "enable_llm_cache" to (hashingKv != null),
        ) + addonConfig.toMap()

    val docStatusStorage: DocStatusStorage =
        JsonDocStatusStorage(
            namespace = "doc_status",
            workspace = "default",
            globalConfig = mapOf("working_dir" to workingDir),
        )
    val fullDocs: BaseKVStorage =
        JsonKVStorage(
            namespace = "full_docs",
            workspace = "default",
            globalConfig = mapOf("working_dir" to workingDir),
        )
    val textChunks: BaseKVStorage =
        JsonKVStorage(
            namespace = "text_chunks",
            workspace = "default",
            globalConfig = mapOf("working_dir" to workingDir),
        )

    val fullEntities: BaseKVStorage =
        JsonKVStorage(
            namespace = "full_entities",
            workspace = "default",
            globalConfig = mapOf("working_dir" to workingDir),
        )
    val fullRelations: BaseKVStorage =
        JsonKVStorage(
            namespace = "full_relations",
            workspace = "default",
            globalConfig = mapOf("working_dir" to workingDir),
        )

    val chunksVdb: BaseVectorStorage =
        InMemoryVectorStorage(
            namespace = "chunks_vdb",
            workspace = "default",
            embeddingFunc = embedding,
            globalConfig = globalConfig,
        )
    val entitiesVdb: BaseVectorStorage =
        InMemoryVectorStorage(
            namespace = "entities_vdb",
            workspace = "default",
            embeddingFunc = embedding,
            globalConfig = globalConfig,
        )
    val relationshipsVdb: BaseVectorStorage =
        InMemoryVectorStorage(
            namespace = "relationships_vdb",
            workspace = "default",
            embeddingFunc = embedding,
            globalConfig = globalConfig,
        )

    val chunkEntityRelationGraph: BaseGraphStorage =
        when (graphStorageName) {
            "MongoGraphStorage" -> {
                lightrag.kg.mongo.MongoGraphStorage(
                    namespace = "chunk_entity_relation_graph",
                    globalConfig = globalConfig,
                    embeddingFunc = embedding,
                )
            }
            "Neo4jGraphStorage" -> {
                lightrag.kg.neo4j.Neo4jGraphStorage(
                    namespace = "chunk_entity_relation_graph",
                    globalConfig = globalConfig,
                    embeddingFunc = embedding,
                )
            }
            else ->
                InMemoryGraphStorage(
                    namespace = "chunk_entity_relation_graph",
                    workspace = "default",
                )
        }

    val kvStorage: BaseKVStorage = textChunks
    val vectorStorage: BaseVectorStorage = entitiesVdb
    val graphStorage: BaseGraphStorage = chunkEntityRelationGraph

    suspend fun insert(input: String): String {
        return insert(listOf(input))
    }

    suspend fun insert(input: List<String>): String {
        val trackId = generateTrackId("insert")
        pipelineEnqueueDocuments(input, trackId)
        pipelineProcessEnqueueDocuments()
        return trackId
    }

    private suspend fun pipelineEnqueueDocuments(
        input: List<String>,
        trackId: String,
        filePaths: List<String>? = null,
    ): String {
        val effectiveFilePaths = filePaths ?: List(input.size) { "unknown_source" }

        val uniqueContent = mutableMapOf<String, Pair<String, String>>()

        for (i in input.indices) {
            val content = input[i]
            val path = effectiveFilePaths[i]
            val md5 = computeMd5(content)
            uniqueContent[md5] = content to path
        }

        val newDocs = mutableMapOf<String, Map<String, Any>>()
        val allNewDocIds = uniqueContent.keys
        val missingDocIds = docStatusStorage.filterKeys(allNewDocIds)

        val uniqueNewDocIds = missingDocIds

        uniqueNewDocIds.forEach { docId ->
            val (content, path) = uniqueContent[docId]!!
            newDocs[docId] =
                mapOf(
                    "status" to DocStatus.PENDING.value,
                    "content_summary" to (content.take(100) + "..."),
                    "content_length" to content.length.toString(),
                    "created_at" to Instant.now().toString(),
                    "updated_at" to Instant.now().toString(),
                    "file_path" to path,
                    "track_id" to trackId,
                )
        }

        if (newDocs.isEmpty()) {
            return trackId
        }

        val fullDocsData =
            uniqueNewDocIds.associateWith { docId ->
                val (content, path) = uniqueContent[docId]!!
                mapOf("content" to content, "file_path" to path)
            }
        fullDocs.upsert(fullDocsData)

        docStatusStorage.upsert(newDocs)

        return trackId
    }

    private suspend fun pipelineProcessEnqueueDocuments() {
        val pendingDocs = docStatusStorage.getDocsByStatus(DocStatus.PENDING)
        if (pendingDocs.isEmpty()) return

        val chunkTokenSize = globalConfig["chunk_token_size"] as? Int ?: 1200
        val chunkOverlapTokenSize = globalConfig["chunk_overlap_token_size"] as? Int ?: 100

        pendingDocs.forEach { (docId, status) ->
            try {
                docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSING.value)))

                val docData = fullDocs.getById(docId)
                val content =
                    docData?.get("content") as? String
                        ?: throw IllegalStateException("Doc content missing for $docId")

                val tokenizer: (String) -> List<Int> = { it.map { c -> c.code } }
                val decoder: (List<Int>) -> String = { list -> list.map { it.toChar() }.joinToString("") }

                val chunks =
                    chunkingByTokenSize(
                        tokenizer = tokenizer,
                        decoder = decoder,
                        content = content,
                        chunkTokenSize = chunkTokenSize,
                        chunkOverlapTokenSize = chunkOverlapTokenSize,
                    )

                val chunksData = mutableMapOf<String, Map<String, Any>>()
                chunks.forEach { chunk ->
                    val chunkId = computeMd5(chunk.content)
                    chunksData[chunkId] =
                        mapOf(
                            "content" to chunk.content,
                            "full_doc_id" to docId,
                            "chunk_order_index" to chunk.chunkOrderIndex.toString(),
                            "tokens" to chunk.tokens.toString(),
                        )
                }

                textChunks.upsert(chunksData)
                val chunksVdbData =
                    chunksData.mapValues { (_, v) ->
                        mapOf(
                            "content" to v["content"]!!,
                            "full_doc_id" to v["full_doc_id"]!!,
                            "file_path" to status.filePath,
                        )
                    }
                chunksVdb.upsert(chunksVdbData)

                val (nodes, edges) = extractEntities(chunksData, globalConfig)

                mergeNodesAndEdges(
                    nodes = nodes,
                    edges = edges,
                    knowledgeGraphInst = chunkEntityRelationGraph,
                    entitiesVdb = entitiesVdb,
                    relationshipsVdb = relationshipsVdb,
                    globalConfig = globalConfig,
                )

                docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSED.value)))
            } catch (e: Exception) {
                e.printStackTrace()
                docStatusStorage.upsert(
                    mapOf(
                        docId to
                            mapOf(
                                "status" to DocStatus.FAILED.value,
                                "error_msg" to (e.message ?: "Unknown error"),
                            ),
                    ),
                )
            }
        }
    }

    suspend fun query(
        query: String,
        param: QueryParam,
    ): QueryResult? {
        return when (param.mode) {
            "local", "global", "hybrid", "mix" -> {
                kgQuery(
                    query = query,
                    knowledgeGraphInst = chunkEntityRelationGraph,
                    entitiesVdb = entitiesVdb,
                    relationshipsVdb = relationshipsVdb,
                    textChunksDb = textChunks,
                    queryParam = param,
                    globalConfig = globalConfig,
                    chunksVdb = chunksVdb,
                    chatModel = chatModel,
                    hashingKv = hashingKv,
                )
            }
            "naive" -> {
                val content =
                    naiveQuery(
                        NaiveQueryParams(
                            query = query,
                            chunksVdb = chunksVdb,
                            queryParam = param,
                            globalConfig = globalConfig,
                            chatModel = chatModel,
                            hashingKv = hashingKv,
                        ),
                    )
                QueryResult(content = content)
            }
            "bypass" -> {
                val response = chatModel.generate(UserMessage(query))
                QueryResult(content = response?.content()?.text())
            }
            else -> {
                logger.error { "Unsupported query mode: ${param.mode}" }
                null
            }
        }
    }

    suspend fun getProcessingStatus(): Map<String, Int> {
        return docStatusStorage.getStatusCounts()
    }

    suspend fun deleteByDocId(docId: String): Map<String, String> {
        docStatusStorage.delete(listOf(docId))
        return mapOf("status" to "success", "doc_id" to docId)
    }
}
