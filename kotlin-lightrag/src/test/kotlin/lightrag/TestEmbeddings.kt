package lightrag

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import io.mockk.every
import io.mockk.mockk

object TestEmbeddings {
    fun mockEmbeddingModel(
        vectors: Map<String, List<Float>> = emptyMap(),
        default: List<Float> = listOf(0.0f, 0.0f),
    ): EmbeddingModel {
        val model = mockk<EmbeddingModel>()

        every { model.embed(any<String>()) } answers {
            val text = this.args.first() as String
            val vector = vectors[text] ?: default
            Response.from(Embedding(vector.toFloatArray()))
        }
        return model
    }
}
