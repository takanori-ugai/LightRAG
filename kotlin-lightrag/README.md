# LightRAG Kotlin (Ktor backend)

LightRAG is a Retrieval-Augmented Generation (RAG) toolkit. This Kotlin module mirrors the Python LightRAG logic with Ktor-based services and LangChain4j models.

## Environment Variables
- `OPENAI_API_KEY` – required for OpenAI models in examples.
- `NEO4J_URI` (default `bolt://localhost:7687`)
- `NEO4J_USERNAME` (default `neo4j`)
- `NEO4J_PASSWORD` (default `neo4j`)
- `NEO4J_DATABASE` (optional, defaults to Neo4j’s default database)
- `NEO4J_WORKSPACE` (optional, sets workspace label for graph/vector nodes)
- `NEO4J_MAX_CONNECTION_POOL_SIZE`, `NEO4J_CONNECTION_TIMEOUT`, `NEO4J_MAX_CONNECTION_LIFETIME` (optional tuning)

## Using `LightRAG`
```kotlin
val rag = LightRAG(
    workingDir = "./rag_storage",
    chatModel = /* ChatModel instance, e.g., OpenAiChatModel */,
    embeddingModel = /* EmbeddingModel instance */,
    graphStorageName = "InMemoryGraphStorage",      // or "Neo4jGraphStorage"
    vectorStorageName = "InMemoryVectorStorage",    // or "Neo4jVectorStorage"
    addonConfig = AddonConfig(
        neo4j = Neo4jConfig(
            uri = System.getenv("NEO4J_URI") ?: "bolt://localhost:7687",
            username = System.getenv("NEO4J_USERNAME") ?: "neo4j",
            password = System.getenv("NEO4J_PASSWORD") ?: "neo4j",
            database = System.getenv("NEO4J_DATABASE")
        ),
        overrides = LightRagOverrides(
            chunkTokenSize = 256,
            chunkOverlapTokenSize = 16,
            cosineBetterThreshold = 0.2,
            entityTypes = listOf("Person", "Organization", "Location", "Event", "Concept"),
            language = "English"
        )
    )
)

// Insert content (returns trackId)
val trackId = rag.insert("Some document text")

// Query
val result = rag.query(
    "What does the document say?",
    QueryParam(
        mode = "naive",              // naive | local | global | hybrid | mix
        includeReferences = true,
        topK = 3,
        chunkTopK = 3
    )
)
println(result?.content)
```

## Running Examples
- Neo4j graph + Neo4j vector store:  
  `./gradlew execute -PmainClass=lightrag.examples.LightRagOpenAiNeo4jVectorDemoKt`
- Neo4j graph (vector in-memory):  
  `./gradlew execute -PmainClass=lightrag.examples.LightRagOpenAiNeo4jDemoKt`
- Simple insert/query (in-memory):  
  `./gradlew execute -PmainClass=lightrag.examples.InsertExampleKt`

Ensure Neo4j is running and the environment variables above are set for the Neo4j demos.
