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
    // Python script sets them programmatically but also reads them.
    // In Kotlin we assume they are set in environment or we can set properties here if needed for testing,
    // but usually System.getenv is read-only. We can rely on defaults in MongoGraphStorage or user setting them.

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
            .dimensions(3072) // Assuming default for text-embedding-3-large, or we can check. Python example calculates it.
            // Python example: "embedding_func" gets dimensions dynamically.
            // LangChain4j usually handles this.
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
        // In Kotlin LightRAG, storages are initialized in constructor or lazily?
        // BaseGraphStorage has initialize(). LightRAG constructor initializes the properties, but
        // it doesn't call initialize() on them.
        // We should probably call initialize() on rag if such method existed, or on storages directly?
        // Current LightRAG.kt doesn't have an explicit initialize() method that calls storage.initialize().
        // But checking StorageInterfaces.kt, initialize() default implementation is empty.
        // MongoGraphStorage overrides it.
        // So we need to call it.
        // However, `rag.chunkEntityRelationGraph` is accessible.
        rag.chunkEntityRelationGraph.initialize()

        // Also kvStorage and others might need init if they were persistent (JsonKVStorage loads from file in init block usually or constructor).
        // Let's check JsonKVStorage. It loads in init block.
        // MongoGraphStorage has `initialize` method.

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
            ),
        )

        // Perform local search
        println("\nLocal Search:")
        println(
            rag.query(
                "What are the top themes in this story?",
                QueryParam(mode = "local"),
            ),
        )

        // Perform global search
        println("\nGlobal Search:")
        println(
            rag.query(
                "What are the top themes in this story?",
                QueryParam(mode = "global"),
            ),
        )

        // Perform hybrid search
        println("\nHybrid Search:")
        println(
            rag.query(
                "What are the top themes in this story?",
                QueryParam(mode = "hybrid"),
            ),
        )

        // Finalize (Close connection)
        rag.chunkEntityRelationGraph.finalize()
    }
}
