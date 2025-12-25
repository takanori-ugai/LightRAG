package lightrag.kg.json

data class KVEntry(
    val value: KVValue,
)

data class KVValue(
    val data: Map<String, Any>,
)
