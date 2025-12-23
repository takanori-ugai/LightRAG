package lightrag

import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.routing.*
import lightrag.api.routers.configureDocumentRoutes
import lightrag.api.routers.configureQueryRoutes
import lightrag.api.routers.configureGraphRoutes
import lightrag.api.routers.configureOllamaRoutes
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
