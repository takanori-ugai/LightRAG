package lightrag.eval

import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVParser
import java.io.StringReader

object RagasContextExtractor {
    fun extractContexts(context: String): List<String> {
        val csvBlocks = extractCsvBlocks(context)
        if (csvBlocks.isEmpty()) return emptyList()
        val collected = mutableListOf<String>()
        for (csv in csvBlocks) {
            val rows = parseCsv(csv)
            if (rows.isEmpty()) continue
            val header = rows.first().map { it.trim().lowercase() }
            val colIdx = header.indexOf("content").takeIf { it != -1 } ?: header.indexOf("context")
            collectContexts(rows, colIdx, collected)
        }
        return collected.distinct()
    }

    private fun extractCsvBlocks(context: String): List<String> {
        val regex =
            Regex(
                "```csv\\s*([\\s\\S]*?)\\s*```",
                RegexOption.MULTILINE,
            )
        return regex.findAll(context).mapNotNull { it.groupValues.getOrNull(1)?.trim() }.toList()
    }

    private fun parseCsv(csv: String): List<List<String>> =
        StringReader(csv).use { reader ->
            val format =
                CSVFormat.DEFAULT
                    .builder()
                    .setIgnoreEmptyLines(true)
                    .build()
            CSVParser.parse(reader, format).use { parser ->
                parser.records.map { record -> record.toList() }
            }
        }

    private fun collectContexts(
        rows: List<List<String>>,
        colIdx: Int,
        collected: MutableList<String>,
    ) {
        rows
            .drop(1)
            .mapNotNull { row ->
                if (colIdx != -1) {
                    row.getOrNull(colIdx)
                } else {
                    row.joinToString(" | ") { it.trim() }
                }
            }.map { it.trim() }
            .filter { it.isNotBlank() }
            .forEach { collected.add(it) }
    }
}
