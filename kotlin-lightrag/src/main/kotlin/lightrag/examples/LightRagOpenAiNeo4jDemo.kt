package lightrag.examples

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import java.io.File
import java.time.Duration

fun main() = runBlocking {
    // Check OpenAI environment variable
    val apiKey = System.getenv("OPENAI_API_KEY")
    if (apiKey.isNullOrBlank()) {
        println("Error: OPENAI_API_KEY environment variable is not set.")
        return@runBlocking
    }

    // Check Neo4j environment variables
    val neo4jUri = System.getenv("NEO4J_URI")
    val neo4jUser = System.getenv("NEO4J_USERNAME")
    val neo4jPass = System.getenv("NEO4J_PASSWORD")

    if (neo4jUri.isNullOrBlank() || neo4jUser.isNullOrBlank() || neo4jPass.isNullOrBlank()) {
        println("Error: NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables are required.")
        return@runBlocking
    }

    val workingDir = "./neo4j_test_dir"
    val dir = File(workingDir)
    if (!dir.exists()) {
        dir.mkdirs()
    }

    // Configure OpenAI Models
    val chatModel = OpenAiChatModel.builder()
        .apiKey(apiKey)
        .modelName("gpt-4o-mini")
        .timeout(Duration.ofSeconds(60))
        .build()

    val embeddingModel = OpenAiEmbeddingModel.builder()
        .apiKey(apiKey)
        .modelName("text-embedding-3-large")
        .dimensions(3072)
        .build()

    val rag = LightRAG(
        workingDir = workingDir,
        chatModel = chatModel,
        embeddingModel = embeddingModel,
        graphStorageName = "Neo4jGraphStorage"
    )

    // Initialize Neo4j Storage (Create indexes etc.)
    println("Initializing Neo4j Graph Storage...")
    rag.chunkEntityRelationGraph.initialize()

    // Prepare content
    val bookFile = File("book.txt")
    val content = if (bookFile.exists()) {
        bookFile.readText()
    } else {
        println("Warning: book.txt not found. Using dummy content.")
        """
        Neo4j is a graph database management system developed by Neo4j, Inc.
        It is an ACID-compliant transactional database with native graph storage and processing.
        LightRAG is a retrieval-augmented generation system that can use Neo4j as a backend.
        Kotlin is a cross-platform, statically typed, general-purpose programming language with type inference.
        """.trimIndent()
    }

    // Insert
    println("Inserting content...")
    rag.insert(content)

    // Query
    val modes = listOf("naive", "local", "global", "hybrid")
    val queryText = "What is Neo4j and how is it related to LightRAG?"

    modes.forEach { mode ->
        println("\n=====================")
        println("Query mode: $mode")
        println("=====================")
        try {
            val result = rag.query(queryText, QueryParam(mode = mode))
            println(result)
        } catch (e: Exception) {
            println("Error querying mode $mode: ${e.message}")
        }
    }

    // Finalize
    rag.chunkEntityRelationGraph.finalize()
    println("\nDone!")
}
