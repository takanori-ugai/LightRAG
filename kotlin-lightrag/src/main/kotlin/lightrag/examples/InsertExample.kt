package lightrag.examples

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG

/**
 * A mock chat model for testing purposes.
 */
class MockChatModel : ChatModel {
    /**
     * Returns a mock response to a user message.
     */
    override fun chat(userMessage: String): String {
        return "entity<|#|>Entity1<|#|>Type1<|#|>Desc1\n" +
            "relation<|#|>Entity1<|#|>Entity2<|#|>Keywords<|#|>Desc2"
    }

    /**
     * Returns a mock response to a list of chat messages.
     */
    override fun chat(messages: List<ChatMessage>): ChatResponse {
        return ChatResponse
            .builder()
            .aiMessage(
                AiMessage.from(
                    "entity<|#|>Entity1<|#|>Type1<|#|>Desc1\n" +
                        "relation<|#|>Entity1<|#|>Entity2<|#|>Keywords<|#|>Desc2",
                ),
            ).build()
    }
}

/**
 * A mock embedding model for testing purposes.
 */
class MockEmbeddingModel : EmbeddingModel {
    /**
     * Returns a mock embedding for a string.
     */
    override fun embed(text: String): Response<Embedding> {
        return Response.from(Embedding(FloatArray(384) { 0.0f }))
    }

    /**
     * Returns a mock embedding for a text segment.
     */
    override fun embed(textSegment: TextSegment): Response<Embedding> {
        return Response.from(Embedding(FloatArray(384) { 0.0f }))
    }

    /**
     * Returns a mock embedding for a list of text segments.
     */
    override fun embedAll(textSegments: List<TextSegment>): Response<List<Embedding>> {
        return Response.from(textSegments.map { Embedding(FloatArray(384) { 0.0f }) })
    }
}

/**
 * The main function for the insert example.
 */
fun main() =
    runBlocking {
        val rag =
            LightRAG(
                chatModel = MockChatModel(),
                embeddingModel = MockEmbeddingModel(),
            )

        val trackId = rag.insert("This is a test document content about Entity1 and Entity2.")
        println("Insert started with trackId: $trackId")

        // In a real app we would poll status, but here we just wait a bit or assume it's done if sync
        // The insert implementation calls pipelineProcessEnqueueDocuments() which runs in the same
        // coroutine scope in my implementation

        val status = rag.getProcessingStatus()
        println("Status: $status")
    }
