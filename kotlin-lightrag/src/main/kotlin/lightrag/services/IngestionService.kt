package lightrag.services

import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.types.DocStatus
import lightrag.operate.chunkingByTokenSize
import lightrag.operate.extractEntities
import lightrag.operate.mergeNodesAndEdges
import lightrag.utils.computeMd5
import lightrag.utils.generateTrackId
import java.time.Instant

private val logger = KotlinLogging.logger {}

class IngestionService(
    private val storageManager: StorageManager,
    private val globalConfig: Map<String, Any?>,
    private val tokenizer: (String) -> List<Int>,
    private val decoder: (List<Int>) -> String,
) {
    companion object {
        private const val DEFAULT_CHUNK_TOKEN_SIZE = 1200
        private const val DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE = 100
        private const val SUMMARY_PREVIEW_LENGTH = 100
    }

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
        val missingDocIds = storageManager.docStatusStorage.filterKeys(allNewDocIds)

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
                storageManager.docStatusStorage.upsert(updates)
                storageManager.fullDocs.upsert(
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
        storageManager.fullDocs.upsert(fullDocsData)

        storageManager.docStatusStorage.upsert(newDocs)

        return trackId
    }

    private suspend fun pipelineProcessEnqueueDocuments() {
        val pendingDocs = storageManager.docStatusStorage.getDocsByStatus(DocStatus.PENDING)
        if (pendingDocs.isEmpty()) return

        val chunkTokenSize = globalConfig["chunk_token_size"] as? Int ?: DEFAULT_CHUNK_TOKEN_SIZE
        val chunkOverlapTokenSize =
            globalConfig["chunk_overlap_token_size"] as? Int ?: DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE

        pendingDocs.forEach { (docId, status) ->
            try {
                storageManager.docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSING.value)))

                val docData = storageManager.fullDocs.getById(docId)
                val content =
                    docData?.get("content") as? String
                        ?: run {
                            val msg = "Doc content missing for $docId"
                            logger.warn { msg }
                            storageManager.docStatusStorage.upsert(
                                mapOf(
                                    docId to
                                        mapOf(
                                            "status" to DocStatus.FAILED.value,
                                            "error_msg" to msg,
                                        ),
                                ),
                            )
                            // Clean up the orphaned status so we don't retry endlessly.
                            storageManager.docStatusStorage.delete(listOf(docId))
                            storageManager.fullDocs.delete(listOf(docId))
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

                storageManager.textChunks.upsert(chunksData)
                val chunksVdbData =
                    chunksData.mapValues { (_, v) ->
                        mapOf(
                            "content" to v["content"]!!,
                            "full_doc_id" to v["full_doc_id"]!!,
                            "file_path" to status.filePath.ifBlank { "unknown_source" },
                        )
                    }
                storageManager.chunksVdb.upsert(chunksVdbData)

                val (nodes, edges) = extractEntities(chunksData, globalConfig)

                mergeNodesAndEdges(
                    nodes = nodes,
                    edges = edges,
                    knowledgeGraphInst = storageManager.chunkEntityRelationGraph,
                    entitiesVdb = storageManager.entitiesVdb,
                    relationshipsVdb = storageManager.relationshipsVdb,
                    fullEntities = storageManager.fullEntities,
                    fullRelations = storageManager.fullRelations,
                )

                storageManager.docStatusStorage.upsert(mapOf(docId to mapOf("status" to DocStatus.PROCESSED.value)))
            } catch (e: Exception) {
                e.printStackTrace()
                storageManager.docStatusStorage.upsert(
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
        val processedDocs = storageManager.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
        val graphEmpty = storageManager.chunkEntityRelationGraph.getAllNodes().isEmpty()
        val chunksEmpty = storageManager.textChunks.isEmpty()

        if (processedDocs.isEmpty() || (!graphEmpty && !chunksEmpty)) {
            return
        }

        logger.warn { "Derived stores empty but processed docs exist. Rebuilding graph/vector/kvs from persisted full_docs." }

        // Clear derived stores
        storageManager.chunkEntityRelationGraph.drop()
        storageManager.chunksVdb.drop()
        storageManager.entitiesVdb.drop()
        storageManager.relationshipsVdb.drop()
        storageManager.fullEntities.drop()
        storageManager.fullRelations.drop()
        storageManager.textChunks.drop()

        // Mark processed docs back to pending and re-run pipeline
        val resetStatuses =
            processedDocs.keys.associateWith {
                mapOf(
                    "status" to DocStatus.PENDING.value,
                    "updated_at" to Instant.now().toString(),
                )
            }
        storageManager.docStatusStorage.upsert(resetStatuses)
        pipelineProcessEnqueueDocuments()
    }

    /**
     * Gets the processing status of the documents.
     * @return A map of the status counts.
     */
    suspend fun getProcessingStatus(): Map<String, Int> {
        return storageManager.docStatusStorage.getStatusCounts()
    }

    /**
     * Deletes a document by its ID.
     * @param docId The ID of the document to delete.
     * @return A map of the status.
     */
    suspend fun deleteByDocId(docId: String): Map<String, String> {
        storageManager.docStatusStorage.delete(listOf(docId))
        return mapOf("status" to "success", "doc_id" to docId)
    }
}
