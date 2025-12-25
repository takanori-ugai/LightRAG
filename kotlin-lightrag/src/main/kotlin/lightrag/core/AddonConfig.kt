package lightrag.core

data class AddonConfig(
    val neo4j: Neo4jConfig? = null,
    val overrides: LightRagOverrides = LightRagOverrides(),
    val extras: Map<String, Any> = emptyMap(),
) {
    fun toMap(): Map<String, Any> {
        val base = mutableMapOf<String, Any>()
        neo4j?.let { base["neo4j"] = it }
        base.putAll(overrides.toMap())
        base.putAll(extras)
        return base
    }
}

data class LightRagOverrides(
    val chunkTokenSize: Int? = null,
    val chunkOverlapTokenSize: Int? = null,
    val entityTypes: List<String>? = null,
    val language: String? = null,
) {
    fun toMap(): Map<String, Any> =
        buildMap {
            chunkTokenSize?.let { put("chunk_token_size", it) }
            chunkOverlapTokenSize?.let { put("chunk_overlap_token_size", it) }
            entityTypes?.let { put("entity_types", it) }
            language?.let { put("language", it) }
        }
}
