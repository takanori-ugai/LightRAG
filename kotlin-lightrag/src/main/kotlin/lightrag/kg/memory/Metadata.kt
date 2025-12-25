package lightrag.kg.memory

data class Metadata(
    val content: String? = null,
    val vector: List<Float>? = null,
    val entityName: String? = null,
    val srcId: String? = null,
    val tgtId: String? = null,
    val raw: Map<String, Any> = emptyMap(),
)
