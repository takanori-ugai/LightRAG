package lightrag.kg.json

/**
 * Represents an entry in the key-value store.
 * @property value The value of the entry.
 */
data class KVEntry(
    val value: KVValue,
)

/**
 * Represents the value of a key-value entry.
 * @property data The data of the value.
 */
data class KVValue(
    val data: Map<String, Any>,
)
