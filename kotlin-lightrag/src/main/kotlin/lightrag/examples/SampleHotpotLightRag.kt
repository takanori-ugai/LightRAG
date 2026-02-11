package lightrag.examples

import kotlinx.coroutines.runBlocking
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
import lightrag.services.StorageManager
import org.koin.core.context.startKoin
import org.koin.core.qualifier.named
import org.koin.dsl.module
import org.koin.java.KoinJavaComponent.get
import java.nio.file.Files
import java.nio.file.Path

data class SampleResult(
    val index: Int,
    val question: String,
    val expectedAnswer: String,
    val queryAnswer: String,
    val context: String,
    val matches: Boolean,
    val scores: RagasScores,
)

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
                ?: 1
        val parallelism = configuredParallelism.coerceAtLeast(1)

        startKoin {
            allowOverride(true)
            modules(appModule, hotpotOverrideModule(workingDir, graphStorage, vectorStorage))
        }

        val rag: LightRAG = get(LightRAG::class.java)
        val storageManager: StorageManager = get(StorageManager::class.java)

        storageManager.initialize()
        storageManager.drop()

        if (parallelism > 1) {
            println("Note: LightRAG evaluation runs sequentially; parallelism=$parallelism will be ignored.")
        }

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
                lines.filter { it.isNotBlank() }.toList()
            }

        results.forEachIndexed { index, line ->
            val result =
                processSample(
                    index = index,
                    line = line,
                    json = json,
                    rag = rag,
                    storageManager = storageManager,
                )
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
            println(
                "RAGAS: relevancy=${String.format("%.3f", result.scores.answerRelevancy)}, " +
                    "ctx_recall=${String.format("%.3f", result.scores.contextRecall)}, " +
                    "ctx_precision=${String.format("%.3f", result.scores.contextPrecision)}, " +
                    "faithfulness=${String.format("%.3f", result.scores.faithfulness)}, " +
                    "answer_correctness=${result.scores.answerCorrectness?.let { String.format("%.3f", it) } ?: "n/a"}, " +
                    "answer_precision=${result.scores.answerPrecision?.let { String.format("%.3f", it) } ?: "n/a"}, " +
                    "answer_recall=${result.scores.answerRecall?.let { String.format("%.3f", it) } ?: "n/a"}, " +
                    "answer_f1=${result.scores.answerF1?.let { String.format("%.3f", it) } ?: "n/a"}",
            )
        }
        if (total > 0) {
            val accuracy = correct.toDouble() / total.toDouble()
            println("Summary: $correct/$total (${String.format("%.2f", accuracy * 100)}%)")
            println(
                "RAGAS Summary: " +
                    "relevancy=${String.format("%.3f", sumAnswerRelevancy / total)}, " +
                    "ctx_recall=${String.format("%.3f", sumContextRecall / total)}, " +
                    "ctx_precision=${String.format("%.3f", sumContextPrecision / total)}, " +
                    "faithfulness=${String.format("%.3f", sumFaithfulness / total)}, " +
                    "answer_correctness=${
                        if (answerCorrectnessCount > 0) {
                            String.format("%.3f", sumAnswerCorrectness / answerCorrectnessCount)
                        } else {
                            "n/a"
                        }
                    }, " +
                    "answer_precision=${
                        if (answerMetricCount > 0) {
                            String.format("%.3f", sumAnswerPrecision / answerMetricCount)
                        } else {
                            "n/a"
                        }
                    }, " +
                    "answer_recall=${
                        if (answerMetricCount > 0) {
                            String.format("%.3f", sumAnswerRecall / answerMetricCount)
                        } else {
                            "n/a"
                        }
                    }, " +
                    "answer_f1=${
                        if (answerMetricCount > 0) {
                            String.format("%.3f", sumAnswerF1 / answerMetricCount)
                        } else {
                            "n/a"
                        }
                    }",
            )
        } else {
            println("Summary: 0/0 (no samples processed)")
        }
    }

private suspend fun processSample(
    index: Int,
    line: String,
    json: Json,
    rag: LightRAG,
    storageManager: StorageManager,
): SampleResult {
    println("Processing sample: ${index + 1}")
    val sample: HotpotSample = json.decodeFromString(line)
    val metrics = RagasMetrics()

    try {
        val paragraphs = sample.paragraphs.map { it.paragraphText }
        rag.insert(paragraphs)
        rag.rebuildDerivedStorageIfEmpty()

        val queryAnswer =
            rag.query(
                "Answer in one or few words, no extra information: ${sample.question}",
                param = QueryParam(mode = "hybrid"),
            )?.content.orEmpty()
        val context =
            rag.query(
                "Answer in one or few words, no extra information: ${sample.question}",
                param = QueryParam(mode = "hybrid", onlyNeedContext = true),
            )?.content.orEmpty()
        val expectedAnswer = sample.answer
        val matches = queryAnswer.trim().equals(expectedAnswer.trim(), ignoreCase = true)

        val ragasSample =
            RagasSample(
                question = sample.question,
                answer = queryAnswer,
                contexts = RagasContextExtractor.extractContexts(context),
                ground_truths = listOf(expectedAnswer),
            )
        val scores = metrics.score(ragasSample)

        return SampleResult(
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
}

private fun hotpotOverrideModule(
    workingDir: String,
    graphStorage: String,
    vectorStorage: String,
) = module {
    single<AppConfig> {
        val cfg = get<LightRagConfig>()
        val baseAddonConfig = addonConfigFrom(cfg)
        val addonConfig =
            baseAddonConfig.copy(
                overrides = baseAddonConfig.overrides.copy(chunkTokenSize = 1200),
            )
        AppConfig(
            workingDir = workingDir,
            graphStorageName = graphStorage,
            vectorStorageName = vectorStorage,
            addonConfig = addonConfig,
            chatModel = get(),
            embeddingModel = get(),
        )
    }

    single<Map<String, Any?>>(named("globalConfig")) {
        val appConfig = get<AppConfig>()
        globalConfigFrom(appConfig)
    }
}
