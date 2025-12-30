package lightrag.examples

import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.coroutines.runBlocking
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import java.io.File

fun prepareWorkingDir(
    path: String,
    filesToDelete: List<String> = emptyList(),
): File {
    val workingDirFile = File(path)
    if (!workingDirFile.exists()) {
        workingDirFile.mkdirs()
    }
    if (filesToDelete.isNotEmpty()) {
        cleanOldFiles(workingDirFile, filesToDelete)
    }
    return workingDirFile
}

fun cleanOldFiles(
    workingDirFile: File,
    filesToDelete: List<String>,
) {
    filesToDelete.forEach { fileName ->
        val file = File(workingDirFile, fileName)
        if (file.exists()) {
            file.delete()
            println("Deleting old file: ${file.absolutePath}")
        }
    }
}

fun testEmbeddingModel(
    embeddingModel: EmbeddingModel,
    testText: String,
) {
    try {
        val embeddingResponse = embeddingModel.embed(testText)
        val embedding = embeddingResponse.content()
        val embeddingDim = embedding.dimension()
        println("\n=======================")
        println("Test embedding function")
        println("========================")
        println("Test text: $testText")
        println("Detected embedding dimension: $embeddingDim\n\n")
    } catch (e: IllegalStateException) {
        println("Error testing embedding: ${e.message}")
    } catch (e: IllegalArgumentException) {
        println("Error testing embedding: ${e.message}")
    }
}

fun loadBookContent(): String {
    val bookFile = File("./book.txt")
    return if (bookFile.exists()) {
        bookFile.readText()
    } else {
        println("Warning: ./book.txt not found. Using dummy content.")
        "This is a story about a developer converting Python code to Kotlin. " +
            "It was a long and arduous journey, " +
            "but eventually, the code compiled and ran successfully. " +
            "The themes involve persistence, programming languages, and AI assistants."
    }
}

fun runDemoQueries(
    rag: LightRAG,
    queryText: String,
    modes: List<String> = listOf("naive", "local", "global", "hybrid"),
    paramBuilder: (String) -> QueryParam =
        { mode ->
            QueryParam(
                mode = mode,
                topK = 5,
                chunkTopK = 2,
            )
        },
) = runBlocking {
    modes.forEach { mode ->
        println("\n=====================")
        println("Query mode: $mode")
        println("=====================")
        try {
            val result =
                rag.query(
                    queryText,
                    paramBuilder(mode),
                )
            println(result?.content)
        } catch (e: IllegalStateException) {
            println("Error querying mode $mode: ${e.message}")
        } catch (e: IllegalArgumentException) {
            println("Error querying mode $mode: ${e.message}")
        }
    }
}

fun addonConfigFrom(cfg: LightRagConfig) =
    AddonConfig(
        overrides =
            LightRagOverrides(
                chunkTokenSize = cfg.addonConfig.chunkTokenSize,
                chunkOverlapTokenSize = cfg.addonConfig.chunkOverlapTokenSize,
                entityTypes = cfg.addonConfig.entityTypes,
                language = cfg.addonConfig.language,
                cosineBetterThreshold = cfg.addonConfig.cosineBetterThreshold,
            ),
        cosineBetterThreshold = cfg.addonConfig.cosineBetterThreshold,
    )

fun globalConfigFrom(appConfig: AppConfig): Map<String, Any?> {
    val overrides = appConfig.addonConfig.overrides
    val chunkTokenSize = overrides.chunkTokenSize ?: 1200
    val chunkOverlapTokenSize = overrides.chunkOverlapTokenSize ?: 100
    val entityTypes = overrides.entityTypes ?: listOf("Person", "Organization", "Location", "Event", "Concept")
    val language = overrides.language ?: "English"
    return mapOf(
        "llm_model_func" to appConfig.chatModel,
        "embedding_func" to appConfig.embeddingModel,
        "chunk_token_size" to chunkTokenSize,
        "chunk_overlap_token_size" to chunkOverlapTokenSize,
        "entity_types" to entityTypes,
        "language" to language,
        "working_dir" to appConfig.workingDir,
        "enable_llm_cache" to (appConfig.hashingKv != null),
    ) + appConfig.addonConfig.toMap()
}
