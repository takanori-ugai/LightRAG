package lightrag.utils

import java.nio.file.Files
import java.nio.file.Path

data class CausalCsvRow(
    val cause: String,
    val result: String,
    val text: String,
    val causalRelationship: Boolean,
    val category: String,
)

fun loadCausalCsv(path: Path): List<CausalCsvRow> =
    Files.newBufferedReader(path).useLines { lines ->
        lines.filter { it.isNotBlank() }.map { parseCausalCsvLine(it) }.toList()
    }

fun parseCausalCsvLine(line: String): CausalCsvRow {
    val fields = parseCsvLine(line)
    require(fields.size == 5) { "Expected 5 columns, got ${fields.size}: $line" }
    val causalRelationship = fields[3].trim().lowercase()
    val isCausal =
        when (causalRelationship) {
            "true" -> true
            "false" -> false
            else -> error("Invalid causal-relationship value: ${fields[3]}")
        }
    return CausalCsvRow(
        cause = fields[0],
        result = fields[1],
        text = stripEntityTags(fields[2]),
        causalRelationship = isCausal,
        category = fields[4],
    )
}

private fun parseCsvLine(line: String): List<String> {
    val fields = mutableListOf<String>()
    val current = StringBuilder()
    var inQuotes = false
    var index = 0
    while (index < line.length) {
        val char = line[index]
        when {
            char == '"' -> {
                if (inQuotes && index + 1 < line.length && line[index + 1] == '"') {
                    current.append('"')
                    index += 1
                } else {
                    inQuotes = !inQuotes
                }
            }

            char == ',' && !inQuotes -> {
                fields.add(current.toString())
                current.setLength(0)
            }

            else -> {
                current.append(char)
            }
        }
        index += 1
    }
    fields.add(current.toString())
    return fields
}

private fun stripEntityTags(text: String): String =
    text
        .replace("<e1>", "")
        .replace("</e1>", "")
        .replace("<e2>", "")
        .replace("</e2>", "")
