package lightrag.api.routers

import io.ktor.server.application.*
import io.ktor.server.routing.*
import io.ktor.server.response.*
import io.ktor.server.request.*
import io.ktor.http.*
import lightrag.core.LightRAG
import kotlinx.serialization.Serializable

@Serializable
data class GraphLabelsResponse(val labels: List<String>)

@Serializable
data class KnowledgeGraphResponse(val nodes: List<String>, val edges: List<String>) // Simplified

@Serializable
data class EntityUpdateRequest(
    val entity_name: String,
    val updated_data: Map<String, String>, // Simplified to String map
    val allow_rename: Boolean = false,
    val allow_merge: Boolean = false
)

@Serializable
data class EntityCreateRequest(
    val entity_name: String,
    val entity_data: Map<String, String> // Simplified
)

@Serializable
data class RelationCreateRequest(
    val source_entity: String,
    val target_entity: String,
    val relation_data: Map<String, String> // Simplified
)

@Serializable
data class EntityMergeRequest(
    val entities_to_change: List<String>,
    val entity_to_change_into: String
)


fun Application.configureGraphRoutes(rag: LightRAG) {
    routing {
        route("/graph") {
            get("/label/list") {
                // Mock implementation
                call.respond(GraphLabelsResponse(listOf("Person", "Organization", "Location")))
            }

            get("/graphs") {
                val label = call.request.queryParameters["label"]
                if (label == null) {
                     call.respond(HttpStatusCode.BadRequest, "Missing label parameter")
                     return@get
                }
                call.respond(KnowledgeGraphResponse(listOf("Node1", "Node2"), listOf("Edge1")))
            }

            post("/entity/edit") {
                val request = call.receive<EntityUpdateRequest>()
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Entity updated"))
            }

            post("/entity/create") {
                val request = call.receive<EntityCreateRequest>()
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Entity created"))
            }

            post("/relation/create") {
                val request = call.receive<RelationCreateRequest>()
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Relation created"))
            }

            post("/entities/merge") {
                val request = call.receive<EntityMergeRequest>()
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Entities merged"))
            }
        }
    }
}
