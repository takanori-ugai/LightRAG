package lightrag.services

import dev.langchain4j.model.chat.ChatModel
import io.github.oshai.kotlinlogging.KotlinLogging
import lightrag.core.QueryParam
import lightrag.core.QueryResult
import lightrag.core.types.BaseKVStorage
import lightrag.operate.NaiveQueryParams
import lightrag.operate.kgQuery
import lightrag.operate.naiveQuery

private val logger = KotlinLogging.logger {}

class QueryService(
    private val storageManager: StorageManager,
    private val chatModel: ChatModel,
    private val hashingKv: BaseKVStorage?,
    private val globalConfig: Map<String, Any?>,
    private val tokenizer: (String) -> List<Int>,
    private val decoder: (List<Int>) -> String,
) {
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
                    knowledgeGraphInst = storageManager.chunkEntityRelationGraph,
                    entitiesVdb = storageManager.entitiesVdb,
                    relationshipsVdb = storageManager.relationshipsVdb,
                    textChunksDb = storageManager.textChunks,
                    queryParam = param,
                    globalConfig = globalConfig,
                    chunksVdb = storageManager.chunksVdb,
                    chatModel = chatModel,
                    hashingKv = hashingKv,
                )
            }
            "naive" -> {
                naiveQuery(
                    NaiveQueryParams(
                        query = query,
                        chunksVdb = storageManager.chunksVdb,
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
}
