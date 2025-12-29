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

/**
 * Represents the response for the Ollama version endpoint.
 * @property version The version of the Ollama API.
 */
@Serializable
data class OllamaVersionResponse(
    val version: String,
)

/**
 * Represents the response for the Ollama tags endpoint.
 * @property models A list of Ollama models.
 */
@Serializable
data class OllamaTagResponse(
    val models: List<OllamaModel>,
)

/**
 * Represents an Ollama model.
 * @property name The name of the model.
 * @property model The model identifier.
 * @property modifiedAt The last modification date of the model.
 * @property size The size of the model in bytes.
 * @property digest The digest of the model.
 * @property details The details of the model.
 */
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

/**
 * Represents the details of an Ollama model.
 * @property parentModel The parent model.
 * @property format The format of the model.
 * @property family The family of the model.
 * @property families The families of the model.
 * @property parameterSize The parameter size of the model.
 * @property quantizationLevel The quantization level of the model.
 */
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

/**
 * Represents the request for the Ollama generate endpoint.
 * @property model The model to use for generation.
 * @property prompt The prompt to generate from.
 * @property stream Whether to stream the response.
 */
@Serializable
data class OllamaGenerateRequest(
    val model: String,
    val prompt: String,
    val stream: Boolean = false,
)

/**
 * Represents the request for the Ollama chat endpoint.
 * @property model The model to use for chat.
 * @property messages The list of messages in the chat.
 * @property stream Whether to stream the response.
 */
@Serializable
data class OllamaChatRequest(
    val model: String,
    val messages: List<OllamaMessage>,
    val stream: Boolean = false,
)

/**
 * Represents a message in an Ollama chat.
 * @property role The role of the message sender (e.g., "user", "assistant").
 * @property content The content of the message.
 */
@Serializable
data class OllamaMessage(
    val role: String,
    val content: String,
)

/**
 * Configures the Ollama-compatible API routes for the Ktor application.
 *
 * This function sets up mock endpoints that mimic the Ollama API for version, tags, generation, and chat.
 *
 * @param rag The LightRAG instance (unused in the current mock implementation).
 */
@Suppress("UnusedParameter")
fun Application.configureOllamaRoutes(rag: LightRAG) {
    routing {
        route("/api") {
            // Ollama API is prefixed with /api in python code as well when included in main app
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
