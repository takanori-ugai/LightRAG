package lightrag.api.routers

import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG

@Serializable
data class OllamaVersionResponse(val version: String)

@Serializable
data class OllamaTagResponse(val models: List<OllamaModel>)

@Serializable
data class OllamaModel(
    val name: String,
    val model: String,
    @SerialName("modified_at")
    val modifiedAt: String,
    val size: Long,
    val digest: String,
    val details: OllamaModelDetails,
)

@Serializable
data class OllamaModelDetails(
    @SerialName("parent_model")
    val parentModel: String,
    val format: String,
    val family: String,
    val families: List<String>,
    @SerialName("parameter_size")
    val parameterSize: String,
    @SerialName("quantization_level")
    val quantizationLevel: String,
)

@Serializable
data class OllamaGenerateRequest(
    val model: String,
    val prompt: String,
    val stream: Boolean = false,
)

@Serializable
data class OllamaChatRequest(
    val model: String,
    val messages: List<OllamaMessage>,
    val stream: Boolean = false,
)

@Serializable
data class OllamaMessage(
    val role: String,
    val content: String,
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
                call.respondText("Generate not implemented: ${request.model}", status = HttpStatusCode.NotImplemented)
            }

            post("/chat") {
                val request = call.receive<OllamaChatRequest>()
                call.respondText("Chat not implemented: ${request.model}", status = HttpStatusCode.NotImplemented)
            }
        }
    }
}
