package lightrag

import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.swagger.swaggerUI
import io.ktor.server.response.respondRedirect
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
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

    routing {
        get("/") { call.respondRedirect("/swagger") }
        swaggerUI(path = "swagger", swaggerFile = "openapi/documentation.yaml")
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

    configureDocumentRoutes(rag)
    configureQueryRoutes(rag)
    configureGraphRoutes(rag)
    configureOllamaRoutes(rag)
}
