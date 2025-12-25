package lightrag.utils

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

object JsonUtils {
    fun escape(input: String): String {
        return input.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace("\b", "\\b")
            .replace("\u000C", "\\f")
    }

    inline fun <reified T> convertObjectToJson(obj: T): String {
        return Json.encodeToString(obj)
    }

    inline fun <reified T> convertJsonToObject(json: String): T {
        return Json.decodeFromString<T>(json)
    }
}
