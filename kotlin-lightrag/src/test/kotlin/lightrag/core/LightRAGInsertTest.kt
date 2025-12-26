package lightrag.core

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatLanguageModel
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
                val mockChatModel = mockk<ChatLanguageModel>()
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
                every { mockChatModel.generate(capture(messagesSlot)) } answers {
                    val messages = messagesSlot.captured
                    val lastMessage = messages.last()
                    val text = if (lastMessage is UserMessage) lastMessage.singleText() ?: "" else ""

                    if (text.contains("Entity_types", ignoreCase = true)) {
                        // Use <|#|> delimiter matching Constants/Prompts
                        val responseText =
                            """
                            entity<|#|>Apple<|#|>Organization<|#|>A tech company
                            entity<|#|>iPhone<|#|>Product<|#|>A mobile phone
                            relation<|#|>Apple<|#|>iPhone<|#|>manufactures<|#|>Apple manufactures iPhone
                            """.trimIndent()
                        Response.from(AiMessage(responseText))
                    } else {
                        Response.from(AiMessage("Mock response"))
                    }
                }

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
                val mockChatModel = mockk<ChatLanguageModel>(relaxed = true)
                val mockEmbeddingModel = mockk<EmbeddingModel>(relaxed = true)

                every { mockEmbeddingModel.embed(any<String>()) } returns
                    Response.from(Embedding(FloatArray(384) { 0.1f }))
                every { mockEmbeddingModel.embedAll(any<List<TextSegment>>()) } answers {
                    val input = firstArg<List<TextSegment>>()
                    Response.from(input.map { Embedding(FloatArray(384) { 0.1f }) })
                }
                every { mockChatModel.generate(any<List<ChatMessage>>()) } returns
                    Response.from(AiMessage("Mock response"))

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
