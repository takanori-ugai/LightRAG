package lightrag.api.routers

import io.ktor.server.application.*
import io.ktor.server.routing.*
import io.ktor.server.response.*
import io.ktor.server.request.*
import io.ktor.http.*
import lightrag.core.LightRAG
import kotlinx.serialization.Serializable

@Serializable
data class InsertTextRequest(val text: String, val file_source: String? = null)

@Serializable
data class InsertTextsRequest(val texts: List<String>, val file_sources: List<String>? = null)

@Serializable
data class InsertResponse(val status: String, val message: String, val track_id: String)

@Serializable
data class DeleteDocRequest(val doc_ids: List<String>)

fun Application.configureDocumentRoutes(rag: LightRAG) {
    routing {
        route("/documents") {
            post("/upload") {
                // Simplified upload handling
                 call.respond(InsertResponse("success", "File uploaded (mock)", "mock_track_id"))
            }

            post("/text") {
                val request = call.receive<InsertTextRequest>()
                val trackId = rag.insert(request.text)
                call.respond(InsertResponse("success", "Text received", trackId))
            }

            post("/texts") {
                 val request = call.receive<InsertTextsRequest>()
                 val trackId = rag.insert(request.texts)
                 call.respond(InsertResponse("success", "Texts received", trackId))
            }

            get("/status_counts") {
                 call.respond(rag.getProcessingStatus())
            }

            delete("/delete_document") {
                 val request = call.receive<DeleteDocRequest>()
                 // Simplified deletion
                 val result = request.doc_ids.map { rag.deleteByDocId(it) }
                 call.respond(result)
            }
        }
    }
}
