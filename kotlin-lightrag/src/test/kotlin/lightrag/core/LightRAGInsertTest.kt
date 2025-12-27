package lightrag.core

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
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
import lightrag.core.types.DocProcessingStatus
import lightrag.core.types.DocStatus
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class LightRAGInsertTest {
    @Test
    fun testInsertSingleDocument() {
        runBlocking {
            try {
                // Setup
                val mockChatModel = mockk<ChatModel>()
                val mockEmbeddingModel = mockk<EmbeddingModel>()

                // Mock Embedding behavior
                every { mockEmbeddingModel.embed(any<String>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embed(any<TextSegment>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embedAll(any<List<TextSegment>>()) } answers {
                    val input = firstArg<List<TextSegment>>()
                    Response.from(input.map { Embedding(FloatArray(384) { 0.1f }) })
                }

                // Mock Chat Model behavior
                val messagesSlot = slot<List<ChatMessage>>()
                val responseProvider: (List<ChatMessage>) -> ChatResponse = { messages ->
                    val lastMessage = messages.last()
                    val text = if (lastMessage is UserMessage) lastMessage.singleText() ?: "" else ""
                    val extractionJson =
                        """
                        {
                          "entities": [
                            {"name": "Apple", "type": "Organization", "description": "A tech company"},
                            {"name": "iPhone", "type": "Product", "description": "A mobile phone"}
                          ],
                          "relations": [
                            {"source": "Apple", "target": "iPhone", "keywords": "manufactures", "description": "Apple manufactures iPhone"}
                          ]
                        }
                        """.trimIndent()

                    if (text.contains("Entity_types", ignoreCase = true)) {
                        ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                    } else {
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

                // Use a temporary directory for storage
                val tempDir = File("build/tmp/test_rag_storage_mockk_${System.currentTimeMillis()}")
                tempDir.mkdirs()

                val rag =
                    LightRAG(
                        workingDir = tempDir.absolutePath,
                        chatModel = mockChatModel,
                        embeddingModel = mockEmbeddingModel,
                    )

                // Test Input
                val content = "Apple released the new iPhone yesterday."

                // Action
                val trackId = rag.insert(content)

                // Verification
                assertNotNull(trackId)

                // Check DocStatus
                val processedDocs: Map<String, DocProcessingStatus> =
                    rag.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
                assertEquals(1, processedDocs.size)

                val docId = processedDocs.keys.first()
                val docData: DocProcessingStatus? = processedDocs[docId]

                assertEquals(DocStatus.PROCESSED, docData?.status)

                // Check Graph
                val graph = rag.chunkEntityRelationGraph
                val node = graph.getNode("Apple")
                assertNotNull(node, "Node 'Apple' should exist in graph")
                assertEquals("Organization", node["entity_type"])

                val edge = graph.getEdge("Apple", "iPhone")
                assertNotNull(edge, "Edge between 'Apple' and 'iPhone' should exist")

                // Cleanup
                tempDir.deleteRecursively()
            } catch (e: Throwable) {
                e.printStackTrace()
                throw e
            }
        }
    }

    @Test
    fun testDuplicateDocumentInsertion() {
        runBlocking {
            try {
                // Setup
                val mockChatModel = mockk<ChatModel>(relaxed = true)
                val mockEmbeddingModel = mockk<EmbeddingModel>(relaxed = true)

                every { mockEmbeddingModel.embed(any<String>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embedAll(any<List<TextSegment>>()) } answers {
                    val input = firstArg<List<TextSegment>>()
                    Response.from(input.map { Embedding(FloatArray(384) { 0.1f }) })
                }
                val extractionJson =
                    """
                    {
                      "entities": [
                        {"name": "Doc", "type": "Concept", "description": "Test entity"}
                      ],
                      "relations": []
                    }
                    """.trimIndent()

                every { mockChatModel.chat(any<List<ChatMessage>>()) } returns
                    ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                every { mockChatModel.chat(any<ChatRequest>()) } returns
                    ChatResponse.builder().aiMessage(AiMessage.from(extractionJson)).build()
                every { mockChatModel.supportedCapabilities() } returns emptySet()

                val tempDir = File("build/tmp/test_rag_storage_dup_mockk_${System.currentTimeMillis()}")
                tempDir.mkdirs()

                val rag =
                    LightRAG(
                        workingDir = tempDir.absolutePath,
                        chatModel = mockChatModel,
                        embeddingModel = mockEmbeddingModel,
                    )

                val content = "This is a test document."

                // First Insert
                rag.insert(content)

                var processedDocs: Map<String, DocProcessingStatus> =
                    rag.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
                assertEquals(1, processedDocs.size)

                // Second Insert (Same content)
                rag.insert(content)

                processedDocs = rag.docStatusStorage.getDocsByStatus(DocStatus.PROCESSED)
                assertEquals(1, processedDocs.size, "Should still be 1 document after duplicate insertion")

                val pendingDocs = rag.docStatusStorage.getDocsByStatus(DocStatus.PENDING)
                assertEquals(0, pendingDocs.size)

                // Cleanup
                tempDir.deleteRecursively()
            } catch (e: Throwable) {
                e.printStackTrace()
                throw e
            }
        }
    }
}
