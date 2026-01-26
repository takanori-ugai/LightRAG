package lightrag.examples

import lightrag.utils.loadCausalCsv
import java.nio.file.Path

fun main() {
    val rows = loadCausalCsv(Path.of("data.csv"))
    val first = rows.firstOrNull()
    if (first == null) {
        println("No rows found in data.csv.")
        return
    }
    println("First row:")
    println(first)
}
