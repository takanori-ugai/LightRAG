package lightrag

import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationStopping
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticResources
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.swagger.swaggerUI
import io.ktor.server.response.respond
import io.ktor.server.response.respondRedirect
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import lightrag.api.routers.configureDocumentRoutes
import lightrag.api.routers.configureGraphRoutes
import lightrag.api.routers.configureOllamaRoutes
import lightrag.api.routers.configureQueryRoutes
import lightrag.core.LightRAG
import lightrag.llm.LLMFactory

private val logger = KotlinLogging.logger {}

fun main() {
    embeddedServer(Netty, port = 8080, host = "0.0.0.0", module = Application::module)
        .start(wait = true)
}

fun Application.module() {
    install(ContentNegotiation) {
        json()
    }
    install(CORS) {
        anyHost()
        allowHeader(io.ktor.http.HttpHeaders.ContentType)
    }

    val apiKey =
        System.getenv("OPENAI_API_KEY")
            ?: error("OPENAI_API_KEY environment variable is required to use OpenAI models.")

    val chatModel =
        LLMFactory.createChatModel(
            binding = "openai",
            modelName = "gpt-5.2",
            apiKey = apiKey,
        )

    val embeddingModel =
        LLMFactory.createEmbeddingModel(
            binding = "openai",
            modelName = "text-embedding-3-small",
            apiKey = apiKey,
        )

    logger.info { "Starting LightRAG with OpenAI models: chat=gpt-4o-mini, embedding=text-embedding-3-small" }

    val rag =
        LightRAG(
            chatModel = chatModel,
            embeddingModel = embeddingModel,
        )

    val resetStorage = System.getenv("LIGHTRAG_RESET_STORAGE")?.equals("true", ignoreCase = true) == true
    if (resetStorage) {
        logger.warn { "LIGHTRAG_RESET_STORAGE=true detected. Dropping all persisted stores before initialization." }
        runBlocking { rag.dropStorages() }
    }

    runBlocking { rag.initializeStorages() }
    runBlocking { rag.rebuildDerivedStorageIfEmpty() }
    environment.monitor.subscribe(ApplicationStopping) {
        runBlocking { rag.persistStorages() }
    }

    routing {
        get("/") { call.respondRedirect("/swagger") }
        swaggerUI(path = "swagger", swaggerFile = "openapi/documentation.yaml")
        staticResources("/ui", "static")
        get("/app") { call.respondRedirect("/ui/index.html") }
        post("/admin/drop") {
            val result = rag.dropStorages()
            call.respond(result)
        }
    }

    configureDocumentRoutes(rag)
    configureQueryRoutes(rag)
    configureGraphRoutes(rag)
    configureOllamaRoutes(rag)
}

private suspend fun LightRAG.initializeStorages() {
    hashingKv?.initialize()
    docStatusStorage.initialize()
    fullDocs.initialize()
    textChunks.initialize()
    fullEntities.initialize()
    fullRelations.initialize()
    chunkEntityRelationGraph.initialize()
    chunksVdb.initialize()
    entitiesVdb.initialize()
    relationshipsVdb.initialize()
    logger.info { "Persistent stores initialized under working dir: $workingDir" }
}

private suspend fun LightRAG.persistStorages() {
    hashingKv?.indexDoneCallback()
    docStatusStorage.indexDoneCallback()
    fullDocs.indexDoneCallback()
    textChunks.indexDoneCallback()
    fullEntities.indexDoneCallback()
    fullRelations.indexDoneCallback()
    chunkEntityRelationGraph.indexDoneCallback()
    chunksVdb.indexDoneCallback()
    entitiesVdb.indexDoneCallback()
    relationshipsVdb.indexDoneCallback()
    logger.info { "Persistent stores flushed under working dir: $workingDir" }
}

@Serializable
data class DropResponse(
    val status: String,
    val details: Map<String, Map<String, String>>,
)

private suspend fun LightRAG.dropStorages(): DropResponse {
    val results: Map<String, Map<String, String>> =
        mapOf(
            "hashing_kv" to (hashingKv?.drop() ?: mapOf("status" to "skipped", "message" to "hashingKv not configured")),
            "doc_status" to docStatusStorage.drop(),
            "full_docs" to fullDocs.drop(),
            "text_chunks" to textChunks.drop(),
            "full_entities" to fullEntities.drop(),
            "full_relations" to fullRelations.drop(),
            "graph" to chunkEntityRelationGraph.drop(),
            "chunks_vdb" to chunksVdb.drop(),
            "entities_vdb" to entitiesVdb.drop(),
            "relationships_vdb" to relationshipsVdb.drop(),
        )
    logger.info { "Persistent stores dropped under working dir: $workingDir" }
    return DropResponse(status = "success", details = results)
}
