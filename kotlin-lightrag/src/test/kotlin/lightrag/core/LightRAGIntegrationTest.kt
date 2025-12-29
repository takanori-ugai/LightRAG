package lightrag.core

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.runBlocking
import lightrag.buildTestLightRag
import org.junit.Test
import java.io.File
import kotlin.test.assertTrue

class LightRAGIntegrationTest {
    @Test
    fun testInsertAndQueryNaiveMode() {
        runBlocking {
            val tempDir = File("build/tmp/test_rag_it_naive_${System.currentTimeMillis()}")
            tempDir.mkdirs()

            try {
                // Setup
                val mockChatModel = mockk<ChatModel>()
                val mockEmbeddingModel = mockk<EmbeddingModel>()

                // 1. Mock Embedding: Return constant vector so cosine similarity is always 1.0
                every { mockEmbeddingModel.embed(any<String>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embed(any<TextSegment>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embedAll(any<List<TextSegment>>()) } answers {
                    val input = firstArg<List<TextSegment>>()
                    Response.from(input.map { Embedding(FloatArray(384) { 0.1f }) })
                }

                // 2. Mock Chat Model
                val messagesSlot = slot<List<ChatMessage>>()
                val responseProvider: (List<ChatMessage>) -> ChatResponse = { messages ->
                    val lastMessage = messages.last()
                    val lastText = if (lastMessage is UserMessage) lastMessage.singleText() ?: "" else ""
                    val extractionJson =
                        """
                        {
                          "entities": [
                            {"name": "France", "type": "Location", "description": "A country in Europe"},
                            {"name": "Paris", "type": "Location", "description": "The capital of France"}
                          ],
                          "relations": [
                            {"source": "France", "target": "Paris", "keywords": "capital", "description": "Paris is the capital of France"}
                          ]
                        }
                        """.trimIndent()

                    // Check if it's Entity Extraction prompt (Insert phase)
                    if (lastText.contains("Entity_types", ignoreCase = true)) {
                        ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                    } else {
                        // Query Phase
                        ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                    }
                }
                every { mockChatModel.chat(capture(messagesSlot)) } answers {
                    responseProvider(messagesSlot.captured)
                }
                every { mockChatModel.chat(any<ChatRequest>()) } answers {
                    val request = invocation.args[0] as ChatRequest
                    responseProvider(request.messages())
                }
                every { mockChatModel.supportedCapabilities() } returns emptySet()

                val rag = buildTestLightRag(mockChatModel, mockEmbeddingModel, tempDir)

                // 3. Insert
                val content = "The capital of France is Paris."
                rag.insert(content)

                // 4. Query (Naive Mode)
                val param = QueryParam(mode = "naive")
                rag.query("What is the capital of France?", param)

                // 5. Verify that the query context contained the inserted content
                // We look at the captured messages. The last one might be the user query,
                // but the SystemMessage before it should contain the context.
                val capturedMessages = messagesSlot.captured
                // We expect [SystemMessage(with context), UserMessage(query)]

                // Note: The slot captures the *last* call.
                // If query() calls generate() multiple times (e.g. for keywords), we need to be careful.
                // Naive query usually calls generate once.

                val systemMessage = capturedMessages.find { it is SystemMessage } as? SystemMessage
                val systemText = systemMessage?.text() ?: ""

                assertTrue(
                    systemText.contains("The capital of France is Paris") ||
                        systemText.contains("Paris"),
                    "Context should contain inserted text. Actual system prompt: $systemText",
                )
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }

    @Test
    fun testInsertAndQueryLocalMode() {
        runBlocking {
            val tempDir = File("build/tmp/test_rag_it_local_${System.currentTimeMillis()}")
            tempDir.mkdirs()

            try {
                val mockChatModel = mockk<ChatModel>()
                val mockEmbeddingModel = mockk<EmbeddingModel>()

                // Mock Embedding
                every { mockEmbeddingModel.embed(any<String>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embed(any<TextSegment>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embedAll(any<List<TextSegment>>()) } answers {
                    val input = firstArg<List<TextSegment>>()
                    Response.from(input.map { Embedding(FloatArray(384) { 0.1f }) })
                }

                val messagesSlot = slot<List<ChatMessage>>()
                val responseProvider: (List<ChatMessage>) -> ChatResponse = { messages ->
                    val lastMessage = messages.last()
                    val lastText = if (lastMessage is UserMessage) lastMessage.singleText() ?: "" else ""
                    val extractionJson =
                        """
                        {
                          "entities": [
                            {"name": "France", "type": "Location", "description": "A country in Europe"},
                            {"name": "Paris", "type": "Location", "description": "The capital of France"}
                          ],
                          "relations": [
                            {"source": "France", "target": "Paris", "keywords": "capital", "description": "Paris is the capital of France"}
                          ]
                        }
                        """.trimIndent()

                    if (lastText.contains("Entity_types", ignoreCase = true)) {
                        ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                    } else {
                        // KG Query
                        ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                    }
                }
                every { mockChatModel.chat(capture(messagesSlot)) } answers {
                    responseProvider(messagesSlot.captured)
                }
                every { mockChatModel.chat(any<ChatRequest>()) } answers {
                    val request = invocation.args[0] as ChatRequest
                    responseProvider(request.messages())
                }
                every { mockChatModel.supportedCapabilities() } returns emptySet()

                val rag = buildTestLightRag(mockChatModel, mockEmbeddingModel, tempDir)

                rag.insert("The capital of France is Paris.")

                val param = QueryParam(mode = "local")
                rag.query("What is the capital of France?", param)

                val capturedMessages = messagesSlot.captured
                val systemMessage = capturedMessages.find { it is SystemMessage } as? SystemMessage
                val systemText = systemMessage?.text() ?: ""

                // In local mode, context is built from entities.
                // We mocked entity extraction to return "France" and "Paris".
                // We expect "France" or "Paris" to appear in the context data (JSON-like structure).

                assertTrue(
                    systemText.contains("France") || systemText.contains("Paris"),
                    "Context should contain extracted entities. Actual system prompt: $systemText",
                )
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }
}
