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
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import lightrag.core.LightRAG
import lightrag.core.QueryParam

@Serializable
data class QueryRequest(
    val query: String,
    val mode: String = "mix",
    @SerialName("only_need_context")
    val onlyNeedContext: Boolean? = null,
    @SerialName("response_type")
    val responseType: String? = null,
    @SerialName("top_k")
    val topK: Int? = null,
    @SerialName("max_token_for_text_unit")
    val maxTokenForTextUnit: Int? = null,
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
                        onlyNeedContext = request.onlyNeedContext ?: false,
                        responseType = request.responseType,
                        topK = request.topK ?: 10,
                    )

                val result = rag.query(request.query, param)
                call.respond(QueryResponse(result?.content ?: "No result generated."))
            }

            post("/stream") {
                call.respondText("Streaming not implemented", status = HttpStatusCode.NotImplemented)
            }
        }
    }
}
