package lightrag.api.routers

import io.ktor.server.application.*
import io.ktor.server.routing.*
import io.ktor.server.response.*
import io.ktor.server.request.*
import io.ktor.http.*
import lightrag.core.LightRAG
import kotlinx.serialization.Serializable

@Serializable
data class OllamaVersionResponse(val version: String)

@Serializable
data class OllamaTagResponse(val models: List<OllamaModel>)

@Serializable
data class OllamaModel(
    val name: String,
    val model: String,
    val modified_at: String,
    val size: Long,
    val digest: String,
    val details: OllamaModelDetails
)

@Serializable
data class OllamaModelDetails(
    val parent_model: String,
    val format: String,
    val family: String,
    val families: List<String>,
    val parameter_size: String,
    val quantization_level: String
)

@Serializable
data class OllamaGenerateRequest(
    val model: String,
    val prompt: String,
    val stream: Boolean = false
)

@Serializable
data class OllamaChatRequest(
    val model: String,
    val messages: List<OllamaMessage>,
    val stream: Boolean = false
)

@Serializable
data class OllamaMessage(
    val role: String,
    val content: String
)


fun Application.configureOllamaRoutes(rag: LightRAG) {
    routing {
        route("/api") { // Ollama API is prefixed with /api in python code as well when included in main app
             get("/version") {
                 call.respond(OllamaVersionResponse("0.1.0"))
             }

             get("/tags") {
                 // Mock
                 call.respond(OllamaTagResponse(listOf()))
             }

             post("/generate") {
                 val request = call.receive<OllamaGenerateRequest>()
                 call.respondText("Generate not implemented", status = HttpStatusCode.NotImplemented)
             }

             post("/chat") {
                 val request = call.receive<OllamaChatRequest>()
                 call.respondText("Chat not implemented", status = HttpStatusCode.NotImplemented)
             }
        }
    }
}
