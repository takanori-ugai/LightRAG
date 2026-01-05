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
- `MONGO_URI` (default `mongodb://0.0.0.0:27017/?directConnection=true`), `MONGO_DATABASE` (default `LightRAG`), `MONGO_KG_COLLECTION` (default `MDB_KG`)

## Storage Engines
- Configure storages via `graphStorageName` and `vectorStorageName` when creating `LightRAG` (see example below). Defaults use in-memory variants for quick starts.
- Graph storage
  - `InMemoryGraphStorage` – default, no external services, keeps the KG in memory.
  - `Neo4jGraphStorage` – persists the KG to Neo4j; uses `NEO4J_*` settings (workspace label respects `NEO4J_WORKSPACE`, `NEO4J_DATABASE`, pool/timeout tunables).
  - `MongoGraphStorage` – stores nodes/edges in MongoDB; configure with `MONGO_URI`, `MONGO_DATABASE`, `MONGO_KG_COLLECTION`.
- Key-value / doc-status storage (used for caching, chunk/entity payloads, and doc status)
  - `JsonKVStorage` + `JsonDocStatusStorage` – default JSON files under `workingDir` (e.g., `./rag_storage/kv_store_text_chunks.json`), no external services required.
  - `Neo4jKVStorage` + `Neo4jDocStatusStorage` – persist KV/doc status in Neo4j; configure via `NEO4J_*` env vars or pass `globalConfig["neo4j"]` when constructing storages; handy for sharing caches across runs or nodes.
- Vector storage
  - `InMemoryVectorStorage` – default, in-memory vectors.
  - `Neo4jVectorStorage` – writes vectors/metadata as Neo4j nodes; uses `NEO4J_*` connection settings.
  - `Neo4jEmbeddingStoreVectorStorage` (alias `Neo4jEmbeddingStore`) – uses LangChain4j’s Neo4jEmbeddingStore with native vector indexes; honors `NEO4J_METADATA_PREFIX`, `NEO4J_AWAIT_INDEX_TIMEOUT`, `NEO4J_EMBEDDING_DIMENSION` plus the standard `NEO4J_*` vars.

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

### Configuring KV storage
- By default, KV/doc-status stores use JSON files under `workingDir` (`JsonKVStorage`/`JsonDocStatusStorage`). The same `workingDir` passed to `LightRAG` is used for these files.
- To persist KV/doc-status in Neo4j (shared caches across runs/nodes), construct the storages and pass them into `LightRAG`:
```kotlin
val embeddingModel = /* EmbeddingModel instance */
val neo4jConfig =
    mapOf(
        "neo4j" to
            mapOf(
                "uri" to (System.getenv("NEO4J_URI") ?: "bolt://localhost:7687"),
                "username" to (System.getenv("NEO4J_USERNAME") ?: "neo4j"),
                "password" to (System.getenv("NEO4J_PASSWORD") ?: "neo4j"),
            ),
    )

val hashingKv = Neo4jKVStorage("hash_cache", "default", neo4jConfig, embeddingModel)
val docStatus = Neo4jDocStatusStorage("doc_status", "default", neo4jConfig, embeddingModel)
val fullDocs = Neo4jKVStorage("full_docs", "default", neo4jConfig, embeddingModel)
// Optional: use Neo4j for chunks/entities/relations too
val textChunks = Neo4jKVStorage("text_chunks", "default", neo4jConfig, embeddingModel)

runBlocking {
    listOf(hashingKv, docStatus, fullDocs, textChunks).forEach { it.initialize() }
}

val rag =
    LightRAG(
        chatModel = /* ChatModel instance */,
        embeddingModel = embeddingModel,
        hashingKv = hashingKv,                        // enables LLM hashing cache
        docStatusStorageOverride = docStatus,
        fullDocsStorageOverride = fullDocs,
        textChunksStorageOverride = textChunks,       // omit to keep JSON-backed chunks
        // fullEntitiesStorageOverride / fullRelationsStorageOverride can also be set
    )
```

## Running Examples
- Neo4j graph + Neo4j vector store:  
  `./gradlew execute -PmainClass=lightrag.examples.LightRagOpenAiNeo4jVectorDemoKt`
- Neo4j graph (vector in-memory):  
  `./gradlew execute -PmainClass=lightrag.examples.LightRagOpenAiNeo4jDemoKt`
- JSON-backed KV/doc status/vector stores (persists under `./json_demo_storage`):  
  `./gradlew execute -PmainClass=lightrag.examples.LightRagJsonStorageDemoKt`
- Neo4j KV/doc status persistence (graph/vector in-memory):  
  `./gradlew execute -PmainClass=lightrag.examples.LightRagNeo4jKVExampleKt`
- Simple insert/query (in-memory):  
  `./gradlew execute -PmainClass=lightrag.examples.InsertExampleKt`

Ensure Neo4j is running and the environment variables above are set for the Neo4j demos.
