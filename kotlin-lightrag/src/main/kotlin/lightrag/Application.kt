package lightrag

import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import lightrag.api.routers.configureDocumentRoutes
import lightrag.api.routers.configureGraphRoutes
import lightrag.api.routers.configureOllamaRoutes
import lightrag.api.routers.configureQueryRoutes
import lightrag.core.LightRAG

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

    val rag = LightRAG()

    configureDocumentRoutes(rag)
    configureQueryRoutes(rag)
    configureGraphRoutes(rag)
    configureOllamaRoutes(rag)
}
