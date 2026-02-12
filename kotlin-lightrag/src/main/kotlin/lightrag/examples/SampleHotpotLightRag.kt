package lightrag.examples

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import lightrag.core.LightRAG
import lightrag.core.QueryParam
import lightrag.di.AppConfig
import lightrag.di.LightRagConfig
import lightrag.di.appModule
import lightrag.eval.HotpotSample
import lightrag.eval.RagasContextExtractor
import lightrag.eval.RagasMetrics
import lightrag.eval.RagasSample
import lightrag.eval.RagasScores
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.java.KoinJavaComponent.get
import org.koin.java.KoinJavaComponent.getKoin
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Executors

data class SampleResult(
    val index: Int,
    val question: String,
    val expectedAnswer: String,
    val queryAnswer: String,
    val context: String,
    val matches: Boolean,
    val scores: RagasScores,
)

private fun formatScore(value: Double?): String = value?.let { String.format(java.util.Locale.ROOT, "%.3f", it) } ?: "n/a"

private fun formatFixed(value: Double): String = String.format(java.util.Locale.ROOT, "%.3f", value)

private fun RagasScores.toLogString(): String =
    "relevancy=${formatFixed(answerRelevancy)}, " +
        "ctx_recall=${formatFixed(contextRecall)}, " +
        "ctx_precision=${formatFixed(contextPrecision)}, " +
        "faithfulness=${formatFixed(faithfulness)}, " +
        "answer_correctness=${formatScore(answerCorrectness)}, " +
        "answer_precision=${formatScore(answerPrecision)}, " +
        "answer_recall=${formatScore(answerRecall)}, " +
        "answer_f1=${formatScore(answerF1)}"

fun main(args: Array<String>) =
    runBlocking {
        val inputPath: Path =
            Path.of(
                System.getenv("HOTPOT_INPUT_PATH")
                    ?: "data/data/musique_ans_v1.0_train-200.jsonl",
            )
        val json = Json { ignoreUnknownKeys = true }

        val graphStorage = System.getenv("GRAPH_STORAGE") ?: "InMemoryGraphStorage"
        val vectorStorage = System.getenv("VECTOR_STORAGE") ?: "InMemoryVectorStorage"
        val workingDir = System.getenv("WORKING_DIR") ?: "./sample_cache"
        val configuredParallelism =
            args.firstOrNull()?.toIntOrNull()
                ?: System.getenv("LIGHTRAG_PARALLELISM")?.toIntOrNull()
                ?: System.getenv("PATHRAG_PARALLELISM")?.toIntOrNull()
                ?: System.getenv("PARALLELISM")?.toIntOrNull()
                ?: 10
        val parallelism = configuredParallelism.coerceAtLeast(1)
        val configuredQueryParallelism =
            args.getOrNull(1)?.toIntOrNull()
                ?: System.getenv("LIGHTRAG_QUERY_PARALLELISM")?.toIntOrNull()
                ?: System.getenv("QUERY_PARALLELISM")?.toIntOrNull()
                ?: 10
        val queryParallelism = configuredQueryParallelism.coerceAtLeast(1)
        val queryDispatcher =
            Executors.newFixedThreadPool(queryParallelism).asCoroutineDispatcher()

        try {
            startKoin {
                allowOverride(true)
                modules(appModule)
            }

            val cfg: LightRagConfig = get(LightRagConfig::class.java)
            val chatModel: ChatModel = get(ChatModel::class.java)
            val embeddingModel: EmbeddingModel = get(EmbeddingModel::class.java)
            val tokenizer: (String) -> List<Int> = getKoin().get(named("tokenizer"))
            val decoder: (List<Int>) -> String = getKoin().get(named("decoder"))
            val metrics = RagasMetrics()

            var total = 0
            var correct = 0
            var sumAnswerRelevancy = 0.0
            var sumContextRecall = 0.0
            var sumContextPrecision = 0.0
            var sumFaithfulness = 0.0
            var sumAnswerCorrectness = 0.0
            var answerCorrectnessCount = 0
            var sumAnswerPrecision = 0.0
            var sumAnswerRecall = 0.0
            var sumAnswerF1 = 0.0
            var answerMetricCount = 0
            val results =
                Files.newBufferedReader(inputPath).useLines { lines ->
                    val semaphore = Semaphore(parallelism)
                    val deferred =
                        lines
                            .filter { it.isNotBlank() }
                            .mapIndexed { index, line ->
                                async(Dispatchers.IO) {
                                    semaphore.withPermit {
                                        processSample(
                                            index = index,
                                            line = line,
                                            json = json,
                                            cfg = cfg,
                                            chatModel = chatModel,
                                            embeddingModel = embeddingModel,
                                            tokenizer = tokenizer,
                                            decoder = decoder,
                                            workingDir = workingDir,
                                            graphStorage = graphStorage,
                                            vectorStorage = vectorStorage,
                                            queryDispatcher = queryDispatcher,
                                            metrics = metrics,
                                        )
                                    }
                                }
                            }.toList()
                    deferred.awaitAll().filterNotNull()
                }

            results.sortedBy { it.index }.forEach { result ->
                total += 1
                if (result.matches) {
                    correct += 1
                }

                sumAnswerRelevancy += result.scores.answerRelevancy
                sumContextRecall += result.scores.contextRecall
                sumContextPrecision += result.scores.contextPrecision
                sumFaithfulness += result.scores.faithfulness
                if (result.scores.answerCorrectness != null) {
                    sumAnswerCorrectness += result.scores.answerCorrectness
                    answerCorrectnessCount += 1
                }
                if (result.scores.answerPrecision != null &&
                    result.scores.answerRecall != null &&
                    result.scores.answerF1 != null
                ) {
                    sumAnswerPrecision += result.scores.answerPrecision
                    sumAnswerRecall += result.scores.answerRecall
                    sumAnswerF1 += result.scores.answerF1
                    answerMetricCount += 1
                }

                println("Sample: ${result.index + 1}")
                println("Context: ${result.context}")
                println("Question: ${result.question}")
                println("Expected: ${result.expectedAnswer}")
                println("LightRAG: ${result.queryAnswer}")
                println("Match: ${result.matches}")
                println("Accuracy: $correct/$total")
                println("RAGAS: ${result.scores.toLogString()}")
            }
            if (total > 0) {
                val accuracy = correct.toDouble() / total.toDouble()
                println(
                    "Summary: $correct/$total (" +
                        "${String.format(java.util.Locale.ROOT, "%.2f", accuracy * 100)}%)",
                )
                val avgAnswerCorrectness =
                    if (answerCorrectnessCount > 0) {
                        sumAnswerCorrectness / answerCorrectnessCount
                    } else {
                        null
                    }
                val avgAnswerPrecision =
                    if (answerMetricCount > 0) {
                        sumAnswerPrecision / answerMetricCount
                    } else {
                        null
                    }
                val avgAnswerRecall =
                    if (answerMetricCount > 0) {
                        sumAnswerRecall / answerMetricCount
                    } else {
                        null
                    }
                val avgAnswerF1 =
                    if (answerMetricCount > 0) {
                        sumAnswerF1 / answerMetricCount
                    } else {
                        null
                    }
                println(
                    "RAGAS Summary: " +
                        "relevancy=${formatFixed(sumAnswerRelevancy / total)}, " +
                        "ctx_recall=${formatFixed(sumContextRecall / total)}, " +
                        "ctx_precision=${formatFixed(sumContextPrecision / total)}, " +
                        "faithfulness=${formatFixed(sumFaithfulness / total)}, " +
                        "answer_correctness=${formatScore(avgAnswerCorrectness)}, " +
                        "answer_precision=${formatScore(avgAnswerPrecision)}, " +
                        "answer_recall=${formatScore(avgAnswerRecall)}, " +
                        "answer_f1=${formatScore(avgAnswerF1)}",
                )
            } else {
                println("Summary: 0/0 (no samples processed)")
            }
        } finally {
            queryDispatcher.close()
        }
    }

private suspend fun processSample(
    index: Int,
    line: String,
    json: Json,
    cfg: LightRagConfig,
    chatModel: ChatModel,
    embeddingModel: EmbeddingModel,
    tokenizer: (String) -> List<Int>,
    decoder: (List<Int>) -> String,
    workingDir: String,
    graphStorage: String,
    vectorStorage: String,
    queryDispatcher: CoroutineDispatcher,
    metrics: RagasMetrics,
): SampleResult? {
    println("Processing sample: ${index + 1}")
    val sample: HotpotSample = json.decodeFromString(line)
    val sampleWorkingDir = Path.of(workingDir, "sample_${index + 1}").toString()
    val (rag, storageManager) =
        buildSampleRag(
            cfg = cfg,
            chatModel = chatModel,
            embeddingModel = embeddingModel,
            tokenizer = tokenizer,
            decoder = decoder,
            workingDir = sampleWorkingDir,
            graphStorage = graphStorage,
            vectorStorage = vectorStorage,
        )

    return try {
        try {
            storageManager.initialize()
            val paragraphs = sample.paragraphs.map { it.paragraphText }
            val paragraphTitles = sample.paragraphs.map { it.title }
            rag.insert(paragraphs, fileSources = paragraphTitles)
            rag.rebuildDerivedStorageIfEmpty()

            val (queryAnswer, context) =
                coroutineScope {
                    val queryDeferred =
                        async(queryDispatcher) {
                            rag
                                .query(
                                    "Answer in one or few words, no extra information: ${sample.question}",
                                    param = QueryParam(mode = "hybrid", topK = 10),
                                )?.content
                                .orEmpty()
                        }
                    val contextDeferred =
                        async(queryDispatcher) {
                            rag
                                .query(
                                    "Answer in one or few words, no extra information: ${sample.question}",
                                    param = QueryParam(mode = "hybrid", onlyNeedContext = true, topK = 10),
                                )?.content
                                .orEmpty()
                        }
                    listOf(queryDeferred, contextDeferred).awaitAll()
                }
            val expectedAnswer = sample.answer
            val matches = queryAnswer.trim().equals(expectedAnswer.trim(), ignoreCase = true)

            val ragasSample =
                RagasSample(
                    question = sample.question,
                    answer = queryAnswer,
                    contexts = RagasContextExtractor.extractContexts(context),
                    groundTruths = listOf(expectedAnswer),
                )
            val scores = metrics.score(ragasSample)

            SampleResult(
                index = index,
                question = sample.question,
                expectedAnswer = expectedAnswer,
                queryAnswer = queryAnswer,
                context = context,
                matches = matches,
                scores = scores,
            )
        } finally {
            storageManager.drop()
        }
    } catch (e: Exception) {
        val reason = e.message ?: e.javaClass.simpleName
        println("Skipping sample ${index + 1} due to error: $reason")
        null
    }
}

private fun buildSampleRag(
    cfg: LightRagConfig,
    chatModel: ChatModel,
    embeddingModel: EmbeddingModel,
    tokenizer: (String) -> List<Int>,
    decoder: (List<Int>) -> String,
    workingDir: String,
    graphStorage: String,
    vectorStorage: String,
): Pair<LightRAG, StorageManager> {
    val baseAddonConfig = addonConfigFrom(cfg)
    val addonConfig =
        baseAddonConfig.copy(
            overrides = baseAddonConfig.overrides.copy(chunkTokenSize = 1200),
        )
    val appConfig =
        AppConfig(
            workingDir = workingDir,
            graphStorageName = graphStorage,
            vectorStorageName = vectorStorage,
            addonConfig = addonConfig,
            chatModel = chatModel,
            embeddingModel = embeddingModel,
        )
    val globalConfig = globalConfigFrom(appConfig)
    val storageManager =
        StorageManager(
            workingDir = appConfig.workingDir,
            embeddingModel = appConfig.embeddingModel,
            graphStorageName = appConfig.graphStorageName,
            vectorStorageName = appConfig.vectorStorageName,
            addonConfig = appConfig.addonConfig,
            globalConfig = globalConfig,
            docStatusStorageOverride = appConfig.docStatusStorageOverride,
            fullDocsStorageOverride = appConfig.fullDocsStorageOverride,
            textChunksStorageOverride = appConfig.textChunksStorageOverride,
            fullEntitiesStorageOverride = appConfig.fullEntitiesStorageOverride,
            fullRelationsStorageOverride = appConfig.fullRelationsStorageOverride,
        )
    val ingestionService =
        IngestionService(
            storageManager = storageManager,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
        )
    val queryService =
        QueryService(
            storageManager = storageManager,
            chatModel = chatModel,
            hashingKv = appConfig.hashingKv,
            globalConfig = globalConfig,
            tokenizer = tokenizer,
            decoder = decoder,
        )
    return LightRAG(ingestionService, queryService, storageManager) to storageManager
}
