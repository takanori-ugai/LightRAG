package lightrag.core

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import dev.langchain4j.model.chat.ChatModel
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
import lightrag.kg.neo4j.Neo4jEmbeddingStoreVectorStorage
import lightrag.kg.neo4j.Neo4jVectorStorage
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

/**
 * Parameters for a query.
 * @property mode The query mode.
 * @property onlyNeedContext Whether to return only the context.
 * @property onlyNeedPrompt Whether to return only the prompt.
 * @property responseType The desired response type.
 * @property stream Whether to stream the response.
 * @property topK The number of top results to return.
 * @property chunkTopK The number of top chunks to return.
 * @property maxEntityTokens The maximum number of tokens for entities.
 * @property maxRelationTokens The maximum number of tokens for relations.
 * @property maxTotalTokens The maximum total number of tokens.
 * @property hlKeywords Keywords to highlight.
 * @property llKeywords Keywords to lowlight.
 * @property conversationHistory The conversation history.
 * @property userPrompt The user prompt.
 * @property enableRerank Whether to enable reranking.
 * @property includeReferences Whether to include references.
 */
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

/**
 * The main class for the LightRAG application.
 * @param workingDir The working directory for the RAG storage.
 * @param chatModel The chat model to use.
 * @param embeddingModel The embedding model to use.
 * @param hashingKv The key-value storage for hashing.
 * @param docStatusStorageOverride The storage for document statuses.
 * @param fullDocsStorageOverride The storage for full documents.
 * @param textChunksStorageOverride The storage for text chunks.
 * @param fullEntitiesStorageOverride The storage for full entities.
 * @param fullRelationsStorageOverride The storage for full relations.
 * @param llmBinding The binding for the language model.
 * @param llmModelName The name of the language model.
 * @param embeddingBinding The binding for the embedding model.
 * @param embeddingModelName The name of the embedding model.
 * @param graphStorageName The name of the graph storage.
 * @param vectorStorageName The name of the vector storage.
 * @param addonConfig The configuration for addons.
 */
class LightRAG(
    val workingDir: String = "./rag_storage",
    chatModel: ChatModel? = null,
    embeddingModel: EmbeddingModel? = null,
    val hashingKv: BaseKVStorage? = null,
    docStatusStorageOverride: DocStatusStorage? = null,
    fullDocsStorageOverride: BaseKVStorage? = null,
    textChunksStorageOverride: BaseKVStorage? = null,
    fullEntitiesStorageOverride: BaseKVStorage? = null,
    fullRelationsStorageOverride: BaseKVStorage? = null,
    llmBinding: String = "ollama",
    llmModelName: String = "llama3",
    embeddingBinding: String = "ollama",
    embeddingModelName: String = "all-minilm",
    val graphStorageName: String = "InMemoryGraphStorage",
    val vectorStorageName: String = "InMemoryVectorStorage",
    val addonConfig: AddonConfig = AddonConfig(),
) {
    companion object {
        private const val DEFAULT_CHUNK_TOKEN_SIZE = 1200
        private const val DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE = 100
        private const val SUMMARY_PREVIEW_LENGTH = 100
    }

    val chatModel: ChatModel =
        chatModel ?: LLMFactory.createChatModel(llmBinding, llmModelName)

    private val embedding: EmbeddingModel =
        embeddingModel ?: LLMFactory.createEmbeddingModel(embeddingBinding, embeddingModelName)

    private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
    private val enc: Encoding = registry.getEncoding(EncodingType.CL100K_BASE)
    /**
     * Tokenizer function that converts a string to a list of integers.
     */
    val tokenizer: (String) -> List<Int> = { text: String ->
        val intArrayList = enc.encode(text)
        val list = mutableListOf<Int>()
        for (i in 0 until intArrayList.size()) {
            list.add(intArrayList.get(i))
        }
        list
    }
    /**
     * Decoder function that converts a list of integers to a string.
     */
    val decoder: (List<Int>) -> String = { list ->
        val intArrayList = IntArrayList()
        list.forEach { intArrayList.add(it) }
        enc.decode(intArrayList)
    }

    /**
     * The global configuration for the LightRAG application.
     */
    val globalConfig: Map<String, Any?> =
        mapOf(
            "llm_model_func" to chatModel,
            "embedding_func" to embedding,
            "chunk_token_size" to DEFAULT_CHUNK_TOKEN_SIZE,
            "chunk_overlap_token_size" to DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE,
            "entity_types" to listOf("Person", "Organization", "Location", "Event", "Concept"),
            "language" to "English",
            "working_dir" to workingDir,
            "enable_llm_cache" to (hashingKv != null),
        ) + addonConfig.toMap()

    /**
     * The storage for document statuses.
     */
    val docStatusStorage: DocStatusStorage =
        docStatusStorageOverride
            ?: JsonDocStatusStorage(
                namespace = "doc_status",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embedding,
            )
    /**
     * The storage for full documents.
     */
    val fullDocs: BaseKVStorage =
        fullDocsStorageOverride
            ?: JsonKVStorage(
                namespace = "full_docs",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embedding,
            )
    /**
     * The storage for text chunks.
     */
    val textChunks: BaseKVStorage =
        textChunksStorageOverride
            ?: JsonKVStorage(
                namespace = "text_chunks",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embedding,
            )

    /**
     * The storage for full entities.
     */
    val fullEntities: BaseKVStorage =
        fullEntitiesStorageOverride
            ?: JsonKVStorage(
                namespace = "full_entities",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embedding,
            )
    /**
     * The storage for full relations.
     */
    val fullRelations: BaseKVStorage =
        fullRelationsStorageOverride
            ?: JsonKVStorage(
                namespace = "full_relations",
                workspace = "default",
                globalConfig = mapOf("working_dir" to workingDir),
                embeddingFunc = embedding,
            )

    private fun createVectorStorage(namespace: String): BaseVectorStorage {
        return when (vectorStorageName) {
            "Neo4jEmbeddingStoreVectorStorage", "Neo4jEmbeddingStore" ->
                Neo4jEmbeddingStoreVectorStorage(
                    namespace = namespace,
                    workspace = "default",
                    globalConfig = globalConfig,
                    embeddingFunc = embedding,
                    cosineThreshold = addonConfig.cosineBetterThreshold,
                )
            "Neo4jVectorStorage" ->
                Neo4jVectorStorage(
                    namespace = namespace,
                    workspace = "default",
                    globalConfig = globalConfig,
                    embeddingFunc = embedding,
                    cosineThreshold = addonConfig.cosineBetterThreshold,
                )
            else ->
                InMemoryVectorStorage(
                    namespace = namespace,
                    workspace = "default",
                    embeddingFunc = embedding,
                    globalConfig = globalConfig,
                    cosineThreshold = addonConfig.cosineBetterThreshold,
                )
        }
    }

    /**
     * The vector storage for chunks.
     */
    val chunksVdb: BaseVectorStorage = createVectorStorage("chunks_vdb")
    /**
     * The vector storage for entities.
     */
    val entitiesVdb: BaseVectorStorage = createVectorStorage("entities_vdb")
    /**
     * The vector storage for relationships.
     */
    val relationshipsVdb: BaseVectorStorage = createVectorStorage("relationships_vdb")

    /**
     * The graph storage for the chunk-entity-relation graph.
     */
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
                    embeddingFunc = embedding,
                )
        }

    /**
     * The key-value storage.
     */
    val kvStorage: BaseKVStorage = textChunks
    /**
     * The vector storage.
     */
    val vectorStorage: BaseVectorStorage = entitiesVdb
    /**
     * The graph storage.
     */
    val graphStorage: BaseGraphStorage = chunkEntityRelationGraph

    /**
     * Inserts a single document.
     * @param input The document to insert.
     * @param fileSource The source of the file.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: String,
        fileSource: String? = null,
    ): String {
        val fileSources = fileSource?.let { listOf(it) }
        return insert(listOf(input), fileSources)
    }

    /**
     * Inserts multiple documents.
     * @param input The documents to insert.
     * @param fileSources The sources of the files.
     * @return A track ID for the insertion.
     */
    suspend fun insert(
        input: List<String>,
        fileSources: List<String>? = null,
    ): String {
        val trackId = generateTrackId("insert")
        pipelineEnqueueDocuments(input, trackId, fileSources)
        pipelineProcessEnqueueDocuments()
        return trackId
    }

    private suspend fun pipelineEnqueueDocuments(
        input: List<String>,
        trackId: String,
        filePaths: List<String>? = null,
    ): String {
        val effectiveFilePaths =
            input.mapIndexed { index, _ ->
                filePaths?.getOrNull(index)
                    ?: filePaths?.lastOrNull()
                    ?: "unknown_source"
            }

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

        // If the content already exists but a new file path is provided, update the stored path
        val existingDocIds = allNewDocIds - uniqueNewDocIds
        if (filePaths != null && existingDocIds.isNotEmpty()) {
            val updates = mutableMapOf<String, Map<String, Any>>()
            existingDocIds.forEach { docId ->
                val (_, path) = uniqueContent[docId] ?: return@forEach
                updates[docId] =
                    mapOf(
                        "status" to DocStatus.PENDING.value,
                        "track_id" to trackId,
                        "file_path" to path,
                        "updated_at" to Instant.now().toString(),
                    )
            }
            if (updates.isNotEmpty()) {
                docStatusStorage.upsert(updates)
                fullDocs.upsert(
                    updates.mapValues { (docId, _) ->
                        val (_, path) = uniqueContent[docId]!!
                        mapOf(
                            "file_path" to path,
                        )
                    },
                )
            }
        }

        uniqueNewDocIds.forEach { docId ->
            val (content, path) = uniqueContent[docId]!!
            newDocs[docId] =
                mapOf(
                    "status" to DocStatus.PENDING.value,
                    "content_summary" to (content.take(SUMMARY_PREVIEW_LENGTH) + "..."),
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

        val chunkTokenSize = globalConfig["chunk_token_size"] as? Int ?: DEFAULT_CHUNK_TOKEN_SIZE
        val chunkOverlapTokenSize =
            globalConfig["chunk_overlap_token_size"] as? Int ?: DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE

        pendingDocs.forEach { (docId, status) ->
            try {
                docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSING.value)))

                val docData = fullDocs.getById(docId)
                val content =
                    docData?.get("content") as? String
                        ?: run {
                            val msg = "Doc content missing for $docId"
                            logger.warn { msg }
                            docStatusStorage.upsert(
                                mapOf(
                                    docId to
                                        mapOf(
                                            "status" to DocStatus.FAILED.value,
                                            "error_msg" to msg,
                                        ),
                                ),
                            )
                            // Clean up the orphaned status so we don't retry endlessly.
                            docStatusStorage.delete(listOf(docId))
                            fullDocs.delete(listOf(docId))
                            return@forEach
                        }

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
                            "file_path" to (status.filePath.ifBlank { "unknown_source" }),
                        )
                }

                textChunks.upsert(chunksData)
                val chunksVdbData =
                    chunksData.mapValues { (_, v) ->
                        mapOf(
                            "content" to v["content"]!!,
                            "full_doc_id" to v["full_doc_id"]!!,
                            "file_path" to status.filePath.ifBlank { "unknown_source" },
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
                    fullEntities = fullEntities,
                    fullRelations = fullRelations,
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

    /**
     * Rebuilds the derived storage if it is empty.
     */
    suspend fun rebuildDerivedStorageIfEmpty() {
        val processedDocs = docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
        val graphEmpty = chunkEntityRelationGraph.getAllNodes().isEmpty()
        val chunksEmpty = textChunks.isEmpty()

        if (processedDocs.isEmpty() || (!graphEmpty && !chunksEmpty)) {
            return
        }

        logger.warn { "Derived stores empty but processed docs exist. Rebuilding graph/vector/kvs from persisted full_docs." }

        // Clear derived stores
        chunkEntityRelationGraph.drop()
        chunksVdb.drop()
        entitiesVdb.drop()
        relationshipsVdb.drop()
        fullEntities.drop()
        fullRelations.drop()
        textChunks.drop()

        // Mark processed docs back to pending and re-run pipeline
        val resetStatuses =
            processedDocs.keys.associateWith {
                mapOf(
                    "status" to DocStatus.PENDING.value,
                    "updated_at" to Instant.now().toString(),
                )
            }
        docStatusStorage.upsert(resetStatuses)
        pipelineProcessEnqueueDocuments()
    }

    /**
     * Queries the LightRAG system.
     * @param query The query to execute.
     * @param param The query parameters.
     * @return The query result.
     */
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
                naiveQuery(
                    NaiveQueryParams(
                        query = query,
                        chunksVdb = chunksVdb,
                        queryParam = param,
                        globalConfig = globalConfig,
                        chatModel = chatModel,
                        hashingKv = hashingKv,
                        tokenizer = tokenizer,
                        decoder = decoder,
                    ),
                )
            }
            "bypass" -> {
                val response = chatModel.chat(query)
                QueryResult(content = response)
            }
            else -> {
                logger.error { "Unsupported query mode: ${param.mode}" }
                null
            }
        }
    }

    /**
     * Gets the processing status of the documents.
     * @return A map of the status counts.
     */
    suspend fun getProcessingStatus(): Map<String, Int> {
        return docStatusStorage.getStatusCounts()
    }

    /**
     * Deletes a document by its ID.
     * @param docId The ID of the document to delete.
     * @return A map of the status.
     */
    suspend fun deleteByDocId(docId: String): Map<String, String> {
        docStatusStorage.delete(listOf(docId))
        return mapOf("status" to "success", "doc_id" to docId)
    }
}
