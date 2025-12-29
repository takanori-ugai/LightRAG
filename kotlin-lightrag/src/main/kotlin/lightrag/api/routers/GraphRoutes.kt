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

/**
 * Represents the response containing a list of graph labels.
 * @property labels The list of labels.
 */
@Serializable
data class GraphLabelsResponse(
    val labels: List<String>,
)

/**
 * Represents a simplified knowledge graph response.
 * @property nodes A list of nodes in the graph.
 * @property edges A list of edges in the graph.
 */
@Serializable
// Simplified
data class KnowledgeGraphResponse(
    val nodes: List<String>,
    val edges: List<String>,
)

/**
 * Represents the request body for updating an entity.
 * @property entityName The name of the entity to update.
 * @property updatedData A map of the data to update.
 * @property allowRename Whether to allow renaming the entity.
 * @property allowMerge Whether to allow merging the entity.
 */
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

/**
 * Represents the request body for creating an entity.
 * @property entityName The name of the entity to create.
 * @property entityData A map of the entity's data.
 */
@Serializable
data class EntityCreateRequest(
    @SerialName("entity_name")
    val entityName: String,
    // Simplified
    @SerialName("entity_data")
    val entityData: Map<String, String>,
)

/**
 * Represents the request body for creating a relation.
 * @property sourceEntity The source entity of the relation.
 * @property targetEntity The target entity of the relation.
 * @property relationData A map of the relation's data.
 */
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

/**
 * Represents the request body for merging entities.
 * @property entitiesToChange The list of entities to merge.
 * @property entityToChangeInto The entity to merge into.
 */
@Serializable
data class EntityMergeRequest(
    @SerialName("entities_to_change")
    val entitiesToChange: List<String>,
    @SerialName("entity_to_change_into")
    val entityToChangeInto: String,
)

/**
 * Configures the graph-related routes for the Ktor application.
 *
 * This function sets up endpoints for interacting with the knowledge graph,
 * including listing labels, retrieving graphs, and creating/editing entities and relations.
 *
 * @param rag The LightRAG instance to be used for graph operations.
 */
@Suppress("UnusedParameter")
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
