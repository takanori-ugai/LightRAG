package lightrag.examples

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG

class MockChatModel : ChatLanguageModel {
    override fun generate(messages: List<dev.langchain4j.data.message.ChatMessage>): Response<AiMessage> {
        return Response.from(
            AiMessage.from("entity<|#|>Entity1<|#|>Type1<|#|>Desc1\nrelation<|#|>Entity1<|#|>Entity2<|#|>Keywords<|#|>Desc2"),
        )
    }
}

class MockEmbeddingModel : EmbeddingModel {
    override fun embed(text: String): Response<Embedding> {
        return Response.from(Embedding(FloatArray(384) { 0.0f }))
    }

    override fun embed(textSegment: TextSegment): Response<Embedding> {
        return Response.from(Embedding(FloatArray(384) { 0.0f }))
    }

    override fun embedAll(textSegments: List<TextSegment>): Response<List<Embedding>> {
        return Response.from(textSegments.map { Embedding(FloatArray(384) { 0.0f }) })
    }
}

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
        // The insert implementation calls pipelineProcessEnqueueDocuments() which runs in the same coroutine scope in my implementation

        val status = rag.getProcessingStatus()
        println("Status: $status")
    }
