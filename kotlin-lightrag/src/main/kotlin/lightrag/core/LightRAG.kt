package lightrag.core

import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.embedding.EmbeddingModel
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
import lightrag.operate.chunkingByTokenSize
import lightrag.operate.extractEntities
import lightrag.operate.kgQuery
import lightrag.operate.mergeNodesAndEdges
import lightrag.operate.naiveQuery
import lightrag.utils.computeMd5
import lightrag.utils.generateTrackId
import java.time.Instant

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
    val hlKeywords: List<String> = emptyList(),
    @SerialName("ll_keywords")
    val llKeywords: List<String> = emptyList(),
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
    // Allow injecting custom models, or configuring via binding strings
    chatModel: ChatLanguageModel? = null,
    embeddingModel: EmbeddingModel? = null,
    llmBinding: String = "ollama",
    llmModelName: String = "llama3",
    embeddingBinding: String = "ollama",
    embeddingModelName: String = "all-minilm",
    graphStorageName: String = "InMemoryGraphStorage",
) {
    // Initialize LLM and Embedding models
    private val model: ChatLanguageModel =
        chatModel ?: LLMFactory.createChatModel(llmBinding, llmModelName)

    private val embedding: EmbeddingModel =
        embeddingModel ?: LLMFactory.createEmbeddingModel(embeddingBinding, embeddingModelName)

    // Global config equivalent
    val globalConfig: Map<String, Any> =
        mapOf(
            "llm_model_func" to model,
            "embedding_func" to embedding,
            "tokenizer" to { text: String -> text.split(Regex("\\s+")).map { it.hashCode() } },
            // Placeholder tokenizer
            "chunk_token_size" to 1200,
            "chunk_overlap_token_size" to 100,
            "entity_types" to listOf("Person", "Organization", "Location", "Event", "Concept"),
            "language" to "English",
            "working_dir" to workingDir,
        )

    // Initialize Storages
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

    // Additional storages to match Python implementation
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

    // Vector Storages
    val chunksVdb: BaseVectorStorage =
        InMemoryVectorStorage(
            namespace = "chunks_vdb",
            workspace = "default",
            embeddingFunc = embedding,
        )
    val entitiesVdb: BaseVectorStorage =
        InMemoryVectorStorage(
            namespace = "entities_vdb",
            workspace = "default",
            embeddingFunc = embedding,
        )
    val relationshipsVdb: BaseVectorStorage =
        InMemoryVectorStorage(
            namespace = "relationships_vdb",
            workspace = "default",
            embeddingFunc = embedding,
        )

    // Graph Storage
    val chunkEntityRelationGraph: BaseGraphStorage =
        when (graphStorageName) {
            "MongoGraphStorage" -> {
                // Use reflection or hardcode instantiation for now as imports might not be available here if modularized,
                // but since it's the same module, we can instantiate directly.
                // We need to import MongoGraphStorage.
                // Since I cannot change imports easily with merge_diff without top context, I'll rely on fully qualified name if possible
                // or just add import. Wait, I should add import.
                // For now, I'll assume I can't import easily and just try to instantiate if I can add import in another block.
                // Actually, let's use a factory approach or hardcode for now.
                // I will add the import in a separate block.
                lightrag.kg.mongo.MongoGraphStorage(
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

    // Alias for backward compatibility if needed, but pointing to specific ones is better
    val kvStorage: BaseKVStorage = textChunks // Default kvStorage points to textChunks
    val vectorStorage: BaseVectorStorage = entitiesVdb // Default to entities?
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
        // 1. Validate and deduplicate
        val effectiveFilePaths = filePaths ?: List(input.size) { "unknown_source" }

        val uniqueContent = mutableMapOf<String, Pair<String, String>>() // md5 -> (content, filePath)

        for (i in input.indices) {
            val content = input[i]
            val path = effectiveFilePaths[i]
            val md5 = computeMd5(content)
            uniqueContent[md5] = content to path
        }

        // 2. Filter out already processed documents
        val newDocs = mutableMapOf<String, Map<String, Any>>()
        val allNewDocIds = uniqueContent.keys
        val missingDocIds = docStatusStorage.filterKeys(allNewDocIds)

        // filterKeys returns keys that are NOT in storage, so we should process them.
        val uniqueNewDocIds = missingDocIds

        uniqueNewDocIds.forEach { docId ->
            val (content, path) = uniqueContent[docId]!!
            // Use String values where possible to simplify serialization, cast Any if needed by storage
            newDocs[docId] =
                mapOf(
                    "status" to DocStatus.PENDING.toString(),
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

        // 3. Store full doc content
        val fullDocsData =
            uniqueNewDocIds.associateWith { docId ->
                val (content, path) = uniqueContent[docId]!!
                mapOf("content" to content, "file_path" to path)
            }
        fullDocs.upsert(fullDocsData)

        // 4. Store doc status
        docStatusStorage.upsert(newDocs)

        return trackId
    }

    private suspend fun pipelineProcessEnqueueDocuments() {
        // 1. Get pending documents
        val pendingDocs = docStatusStorage.getDocsByStatus(DocStatus.PENDING)
        if (pendingDocs.isEmpty()) return

        val chunkTokenSize = globalConfig["chunk_token_size"] as? Int ?: 1200
        val chunkOverlapTokenSize = globalConfig["chunk_overlap_token_size"] as? Int ?: 100

        // 2. Process each document
        pendingDocs.forEach { (docId, status) ->
            try {
                // Update status to PROCESSING
                docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSING.toString())))

                // Get content
                val docData = fullDocs.getById(docId)
                val content =
                    docData?.get("content") as? String
                        ?: throw IllegalStateException("Doc content missing for $docId")

                // Chunking
                // We use a reversible character-based "tokenizer" logic here for simplicity and robustness
                // or just split by token size but keep strings.
                // The chunkingByTokenSize function signature requires (String) -> List<Int> and (List<Int>) -> String
                // We can implement a simple character-level tokenizer which is 100% safe.
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

                // Prepare chunks for storage and extraction
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

                // Upsert chunks to KV and VectorDB
                textChunks.upsert(chunksData)
                // In Python, chunks_vdb stores vectors. Here we need to embed.
                // InMemoryVectorStorage handles embedding in upsert if embeddingFunc is present.
                // But data needs to be structured correctly.
                val chunksVdbData =
                    chunksData.mapValues { (_, v) ->
                        mapOf(
                            "content" to v["content"]!!,
                            "full_doc_id" to v["full_doc_id"]!!,
                            "file_path" to status.filePath,
                        )
                    }
                chunksVdb.upsert(chunksVdbData)

                // Extract Entities and Relations
                // Passing chunksData directly
                val (nodes, edges) = extractEntities(chunksData, globalConfig)

                // Merge and Upsert Graph
                mergeNodesAndEdges(
                    nodes = nodes,
                    edges = edges,
                    knowledgeGraphInst = chunkEntityRelationGraph,
                    entitiesVdb = entitiesVdb,
                    relationshipsVdb = relationshipsVdb,
                    globalConfig = globalConfig,
                )

                // Update Status to PROCESSED
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
    ): String {
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
                ) ?: "No result generated."
            }
            "naive" -> {
                naiveQuery(
                    query = query,
                    chunksVdb = chunksVdb,
                    queryParam = param,
                    globalConfig = globalConfig,
                ) ?: "No result generated."
            }
            "bypass" -> {
                // Direct LLM call
                try {
                    val messages = mutableListOf<dev.langchain4j.data.message.ChatMessage>()
                    // If we have conversation history, we could add it here
                    // For now, just user query
                    messages.add(dev.langchain4j.data.message.UserMessage(query))
                    model.generate(messages).content().text()
                } catch (e: Exception) {
                    "Error generating response: ${e.message}"
                }
            }
            else -> throw IllegalArgumentException("Unknown mode: ${param.mode}")
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
