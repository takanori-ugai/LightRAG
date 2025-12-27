package lightrag.api.routers

import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.delete
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG

@Serializable
data class InsertTextRequest(
    val text: String,
    @SerialName("file_source") val fileSource: String? = null,
)

@Serializable
data class InsertTextsRequest(
    val texts: List<String>,
    @SerialName("file_sources") val fileSources: List<String>? = null,
)

@Serializable
data class InsertResponse(
    val status: String,
    val message: String,
    @SerialName("track_id") val trackId: String,
)

@Serializable
data class DeleteDocRequest(
    @SerialName("doc_ids") val docIds: List<String>,
)

fun Application.configureDocumentRoutes(rag: LightRAG) {
    routing {
        route("/documents") {
            post("/upload") {
                // Simplified upload handling
                call.respond(InsertResponse("success", "File uploaded (mock)", "mock_track_id"))
            }

            post("/text") {
                val request = call.receive<InsertTextRequest>()
                val trackId = rag.insert(request.text, request.fileSource)
                call.respond(InsertResponse("success", "Text received", trackId))
            }

            post("/texts") {
                val request = call.receive<InsertTextsRequest>()
                val trackId = rag.insert(request.texts, request.fileSources)
                call.respond(InsertResponse("success", "Texts received", trackId))
            }

            get("/status_counts") {
                call.respond(rag.getProcessingStatus())
            }

            delete("/delete_document") {
                val request = call.receive<DeleteDocRequest>()
                // Simplified deletion
                val result = request.docIds.map { rag.deleteByDocId(it) }
                call.respond(result)
            }
        }
    }
}
