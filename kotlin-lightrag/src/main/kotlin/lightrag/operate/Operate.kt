package lightrag.operate

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatLanguageModel
import lightrag.core.QueryParam
import lightrag.core.types.BaseGraphStorage
import lightrag.core.types.BaseKVStorage
import lightrag.core.types.BaseVectorStorage
import lightrag.utils.Prompts
import org.slf4j.LoggerFactory

private val logger = LoggerFactory.getLogger("lightrag.operate")

data class ChunkingResult(
    val tokens: Int,
    val content: String,
    val chunkOrderIndex: Int,
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
                    logger.warn(
                        "Chunk split_by_character exceeds token limit: len=${chunkTokens.size} limit=$chunkTokenSize",
                    )
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
                        newChunks.add(minOf(chunkTokenSize, chunkTokens.size - start) to chunkContent)
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

suspend fun kgQuery(
    query: String,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage,
    queryParam: QueryParam,
    // Changed from dict[str, str] to Map<String, Any> to be more flexible
    globalConfig: Map<String, Any>,
    hashingKv: BaseKVStorage? = null,
    systemPrompt: String? = null,
    chunksVdb: BaseVectorStorage? = null,
    chatModel: ChatLanguageModel? = null,
): String? {
    if (query.isBlank()) {
        return Prompts.FAIL_RESPONSE
    }

    val model = chatModel ?: globalConfig["llm_model_func"] as? ChatLanguageModel
    if (model == null) {
        logger.error("No ChatLanguageModel provided for kgQuery")
        return null
    }

    // 1. Keyword extraction (simplified for now - just use query as keyword)
    val keywords = listOf(query)

    // 2. Search (Local/Global/Hybrid) - simplified to Local Search
    // Fetch related entities from entitiesVdb
    val entities = entitiesVdb.query(query, queryParam.top_k)

    // Build context
    val contextBuilder = StringBuilder()
    contextBuilder.append(Prompts.KG_QUERY_CONTEXT)

    val entitiesStr =
        entities.joinToString("\n") {
            // Simplified entity string
            "{ \"entity_name\": \"${it["entity_name"]}\", \"content\": \"${it["content"]?.toString()?.replace("\n", " ")}\" }"
        }

    val relationsStr = "" // Placeholder for relations

    val textChunksStr = "" // Placeholder for chunks

    val referenceListStr = "" // Placeholder for references

    val contextContent =
        contextBuilder.toString()
            .replace("{entities_str}", entitiesStr)
            .replace("{relations_str}", relationsStr)
            .replace("{text_chunks_str}", textChunksStr)
            .replace("{reference_list_str}", referenceListStr)

    // 3. LLM call
    val sysPromptTemplate = systemPrompt ?: Prompts.RAG_RESPONSE
    val userPrompt = if (queryParam.response_type != null) "\n\n${queryParam.response_type}" else "n/a"

    val sysPrompt =
        sysPromptTemplate
            .replace("{response_type}", queryParam.response_type ?: "Multiple Paragraphs")
            .replace("{user_prompt}", userPrompt)
            .replace("{context_data}", contextContent)

    if (queryParam.only_need_context) {
        return contextContent
    }

    try {
        val messages =
            listOf(
                SystemMessage(sysPrompt),
                UserMessage(query),
            )
        val response: AiMessage = model.generate(messages).content()
        return response.text()
    } catch (e: Exception) {
        logger.error("Error generating response in kgQuery", e)
        return "Error generating response."
    }
}

suspend fun naiveQuery(
    query: String,
    chunksVdb: BaseVectorStorage,
    queryParam: QueryParam,
    globalConfig: Map<String, Any>,
    hashingKv: BaseKVStorage? = null,
    systemPrompt: String? = null,
    chatModel: ChatLanguageModel? = null,
): String? {
    if (query.isBlank()) {
        return Prompts.FAIL_RESPONSE
    }

    // Basic vector search
    // In Python: _get_vector_context -> process_chunks_unified -> generate_reference_list_from_chunks -> build context -> LLM

    val searchTopK = queryParam.top_k
    // queryParam.chunk_top_k is not in QueryParam class yet, assuming top_k

    val results = chunksVdb.query(query, searchTopK)
    if (results.isEmpty()) {
        return null
    }

    val contextBuilder = StringBuilder()
    contextBuilder.append(Prompts.NAIVE_QUERY_CONTEXT)
    // We need to fill in text_chunks_str and reference_list_str

    val docChunks =
        results.mapIndexed { index, res ->
            mapOf(
                "reference_id" to "${index + 1}",
                "content" to (res["content"] ?: ""),
            )
        }

    // For now, simple context building (simplification of Python logic)
    val textChunksStr =
        docChunks.joinToString("\n") { chunk ->
            "{\"reference_id\": \"${chunk["reference_id"]}\", \"content\": \"${chunk["content"]}\"}"
        }

    val referenceListStr =
        results.mapIndexed { index, res ->
            "[${index + 1}] ${res["file_path"] ?: "unknown_source"}"
        }.joinToString("\n")

    val contextContent =
        contextBuilder.toString()
            .replace("{text_chunks_str}", textChunksStr)
            .replace("{reference_list_str}", referenceListStr)

    val sysPromptTemplate = systemPrompt ?: Prompts.NAIVE_RAG_RESPONSE

    val userPrompt = if (queryParam.response_type != null) "\n\n${queryParam.response_type}" else "n/a" // Assuming user_prompt logic

    val sysPrompt =
        sysPromptTemplate
            .replace("{response_type}", queryParam.response_type ?: "Multiple Paragraphs")
            // Wait, python code has user_prompt in QueryParam separate from user query
            .replace("{user_prompt}", userPrompt)
            .replace("{content_data}", contextContent)

    if (queryParam.only_need_context) {
        return contextContent
    }

    // Call LLM
    val model = chatModel ?: globalConfig["llm_model_func"] as? ChatLanguageModel

    if (model == null) {
        logger.error("No ChatLanguageModel provided for naiveQuery")
        return "Error: No LLM model configured."
    }

    try {
        val messages =
            listOf(
                SystemMessage(sysPrompt),
                UserMessage(query),
            )
        val response: AiMessage = model.generate(messages).content()
        return response.text()
    } catch (e: Exception) {
        logger.error("Error generating response in naiveQuery", e)
        return "Error generating response."
    }
}
