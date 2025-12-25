package lightrag.examples

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import java.io.File
import java.time.Duration

fun main() {
    val workingDir = "./mongodb_test_dir"
    val dir = File(workingDir)
    if (!dir.exists()) {
        dir.mkdirs()
    }

    val apiKey = System.getenv("OPENAI_API_KEY") ?: "sk-"
    // Ensure Mongo env vars are set or default to localhost

    // Configure OpenAI Models
    val chatModel =
        OpenAiChatModel.builder()
            .apiKey(apiKey)
            .modelName("gpt-4o-mini") // gpt_4o_mini_complete equivalent
            .timeout(Duration.ofSeconds(60))
            .build()

    val embeddingModel =
        OpenAiEmbeddingModel.builder()
            .apiKey(apiKey)
            .modelName("text-embedding-3-large")
            .dimensions(3072) // Assuming default for text-embedding-3-large
            .build()

    // Python script calculates dimension. We can do that too if needed, but for now assuming standard.
    // text-embedding-3-large default is 3072 unless reduced.

    val rag =
        LightRAG(
            workingDir = workingDir,
            chatModel = chatModel,
            embeddingModel = embeddingModel,
            graphStorageName = "MongoGraphStorage",
        )

    runBlocking {
        // Initialize storages
        rag.chunkEntityRelationGraph.initialize()

        // Also kvStorage and others might need init if they were persistent
        // (JsonKVStorage loads from file in init block usually or constructor).

        val bookFile = File("book.txt")
        if (bookFile.exists()) {
            val content = bookFile.readText()
            rag.insert(content)
        } else {
            println("book.txt not found. Please place 'book.txt' in the working directory.")
            return@runBlocking
        }

        // Perform naive search
        println("Naive Search:")
        println(
            rag.query(
                "What are the top themes in this story?",
                QueryParam(mode = "naive"),
            )?.content,
        )

        // Perform local search
        println("\nLocal Search:")
        println(
            rag.query(
                "What are the top themes in this story?",
                QueryParam(mode = "local"),
            )?.content,
        )

        // Perform global search
        println("\nGlobal Search:")
        println(
            rag.query(
                "What are the top themes in this story?",
                QueryParam(mode = "global"),
            )?.content,
        )

        // Perform hybrid search
        println("\nHybrid Search:")
        println(
            rag.query(
                "What are the top themes in this story?",
                QueryParam(mode = "hybrid"),
            )?.content,
        )

        // Finalize (Close connection)
        rag.chunkEntityRelationGraph.finalize()
    }
}
