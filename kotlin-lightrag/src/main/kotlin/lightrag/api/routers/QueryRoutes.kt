package lightrag.api.routers

import io.ktor.http.HttpStatusCode
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG
import lightrag.core.QueryParam

@Serializable
data class QueryRequest(
    val query: String,
    val mode: String = "mix",
    val only_need_context: Boolean? = null,
    val response_type: String? = null,
    val top_k: Int? = null,
    val max_token_for_text_unit: Int? = null,
)

@Serializable
data class QueryResponse(val response: String)

fun Application.configureQueryRoutes(rag: LightRAG) {
    routing {
        route("/query") {
            post {
                val request = call.receive<QueryRequest>()

                val param =
                    QueryParam(
                        mode = request.mode,
                        only_need_context = request.only_need_context ?: false,
                        response_type = request.response_type,
                        top_k = request.top_k ?: 10,
                    )

                val result = rag.query(request.query, param)
                call.respond(QueryResponse(result))
            }

            post("/stream") {
                call.respondText("Streaming not implemented", status = HttpStatusCode.NotImplemented)
            }
        }
    }
}
