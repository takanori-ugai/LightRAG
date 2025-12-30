package lightrag.services

import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.types.DocStatus
import lightrag.operate.chunkingByTokenSize
import lightrag.operate.extractEntities
import lightrag.operate.mergeNodesAndEdges
import lightrag.utils.computeMd5

private val logger = KotlinLogging.logger {}

class DocumentProcessor(
    private val storageManager: StorageManager,
    private val globalConfig: Map<String, Any?>,
    private val tokenizer: (String) -> List<Int>,
    private val decoder: (List<Int>) -> String,
) {
    companion object {
        private const val DEFAULT_CHUNK_TOKEN_SIZE = 1200
        private const val DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE = 100
    }

    suspend fun pipelineProcessEnqueueDocuments() {
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
            } catch (e: IllegalStateException) {
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
            } catch (e: IllegalArgumentException) {
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
}
