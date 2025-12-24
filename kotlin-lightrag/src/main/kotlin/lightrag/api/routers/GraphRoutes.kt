package lightrag.api.routers

import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG

@Serializable
data class GraphLabelsResponse(val labels: List<String>)

@Serializable
// Simplified
data class KnowledgeGraphResponse(val nodes: List<String>, val edges: List<String>)

@Serializable
data class EntityUpdateRequest(
    val entity_name: String,
    // Simplified to String map
    val updated_data: Map<String, String>,
    val allow_rename: Boolean = false,
    val allow_merge: Boolean = false,
)

@Serializable
data class EntityCreateRequest(
    val entity_name: String,
    // Simplified
    val entity_data: Map<String, String>,
)

@Serializable
data class RelationCreateRequest(
    val source_entity: String,
    val target_entity: String,
    // Simplified
    val relation_data: Map<String, String>,
)

@Serializable
data class EntityMergeRequest(
    val entities_to_change: List<String>,
    val entity_to_change_into: String,
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
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Entity updated"))
            }

            post("/entity/create") {
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Entity created"))
            }

            post("/relation/create") {
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Relation created"))
            }

            post("/entities/merge") {
                // Mock
                call.respond(mapOf("status" to "success", "message" to "Entities merged"))
            }
        }
    }
}
