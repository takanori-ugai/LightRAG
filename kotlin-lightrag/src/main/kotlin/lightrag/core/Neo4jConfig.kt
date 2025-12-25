package lightrag.core

data class Neo4jConfig(
    val uri: String? = null,
    val username: String? = null,
    val password: String? = null,
    val database: String? = null,
) {
    fun toMap(): Map<String, Any?> =
        mapOf(
            "uri" to uri,
            "username" to username,
            "password" to password,
            "database" to database,
        )
}
