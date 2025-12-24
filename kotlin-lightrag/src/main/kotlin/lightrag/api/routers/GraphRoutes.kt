package lightrag.api.routers

import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG

@Serializable
data class GraphLabelsResponse(val labels: List<String>)

@Serializable
// Simplified
data class KnowledgeGraphResponse(val nodes: List<String>, val edges: List<String>)

@Serializable
data class EntityUpdateRequest(
    @SerialName("entity_name")
    val entityName: String,
    // Simplified to String map
    @SerialName("updated_data")
    val updatedData: Map<String, String>,
    @SerialName("allow_rename")
    val allowRename: Boolean = false,
    @SerialName("allow_merge")
    val allowMerge: Boolean = false,
)

@Serializable
data class EntityCreateRequest(
    @SerialName("entity_name")
    val entityName: String,
    // Simplified
    @SerialName("entity_data")
    val entityData: Map<String, String>,
)

@Serializable
data class RelationCreateRequest(
    @SerialName("source_entity")
    val sourceEntity: String,
    @SerialName("target_entity")
    val targetEntity: String,
    // Simplified
    @SerialName("relation_data")
    val relationData: Map<String, String>,
)

@Serializable
data class EntityMergeRequest(
    @SerialName("entities_to_change")
    val entitiesToChange: List<String>,
    @SerialName("entity_to_change_into")
    val entityToChangeInto: String,
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
