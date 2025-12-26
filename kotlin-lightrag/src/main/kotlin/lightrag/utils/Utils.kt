package lightrag.utils

import java.security.MessageDigest
import java.util.UUID

fun computeMd5(input: String): String {
    val md = MessageDigest.getInstance("MD5")
    val digest = md.digest(input.toByteArray())
    return digest.joinToString("") { "%02x".format(it) }
}

fun generateTrackId(prefix: String): String {
    return "$prefix-${UUID.randomUUID()}"
}

fun <T> List<T>.chunked(
    size: Int,
    overlap: Int,
): List<List<T>> {
    require(size > 0) { "Size must be greater than 0" }
    require(overlap >= 0) { "Overlap must be non-negative" }
    require(overlap < size) { "Overlap must be less than size" }

    val result = mutableListOf<List<T>>()
    var index = 0
    while (index < this.size) {
        val end = minOf(index + size, this.size)
        result.add(this.subList(index, end))
        index += (size - overlap)
        if (end == this.size) break
    }
    return result
}
