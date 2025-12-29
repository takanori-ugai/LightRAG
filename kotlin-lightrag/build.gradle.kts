import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.3.0"
    application
    id("com.gradleup.shadow") version "9.2.0"
    kotlin("plugin.serialization") version "2.3.0"
    id("org.jlleitschuh.gradle.ktlint") version "14.0.1"
    id("io.gitlab.arturbosch.detekt") version "1.23.6"
}

detekt {
    buildUponDefaultConfig = true
    config.setFrom(files("config/detekt.yml"))
}

group = "com.lightrag"
version = "0.0.1"

application {
    mainClass.set("lightrag.ApplicationKt")
}

repositories {
    mavenCentral()
    gradlePluginPortal()
}

// sourceSets {
//    main {
//        kotlin {
//            exclude("lightrag/examples/**")
//        }
//    }
// }

dependencies {
    implementation("io.ktor:ktor-server-core:2.3.12")
    implementation("io.ktor:ktor-server-netty:2.3.12")
    implementation("io.ktor:ktor-server-content-negotiation:2.3.12")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.12")
    implementation("io.ktor:ktor-server-swagger:2.3.12")
    implementation("io.ktor:ktor-server-cors:2.3.12")
    implementation("ch.qos.logback:logback-classic:1.5.13")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
    implementation("io.github.oshai:kotlin-logging-jvm:5.1.0")

    // Koin for Ktor
    implementation("io.insert-koin:koin-ktor:4.1.1")
    implementation("io.insert-koin:koin-core:4.1.1")
    implementation("io.insert-koin:koin-logger-slf4j:4.1.1")

    // LangChain4j dependencies
    implementation("dev.langchain4j:langchain4j:1.9.1")
    implementation("dev.langchain4j:langchain4j-open-ai:1.9.1")
    implementation("dev.langchain4j:langchain4j-ollama:1.9.1")
    implementation("dev.langchain4j:langchain4j-community-neo4j:1.9.1-beta17")

    // JTokkit
    implementation("com.knuddels:jtokkit:1.1.0")

    // MongoDB
    implementation("org.mongodb:mongodb-driver-kotlin-coroutine:5.1.0")
    implementation("org.mongodb:bson-kotlinx:5.1.0")
    implementation("org.neo4j.driver:neo4j-java-driver:5.20.0")

    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.3.0")
    testImplementation("io.mockk:mockk:1.13.9")
}

tasks {
    withType<Test> {
        jvmArgs("-XX:+EnableDynamicAgentLoading")
        testLogging {
            events("passed", "skipped", "failed", "standardOut", "standardError")
            showStandardStreams = true
        }
    }

    val execute by registering(JavaExec::class) {
        group = "application"
        mainClass.set(
            if (project.hasProperty("mainClass")) {
                project.property("mainClass") as String
            } else {
                application.mainClass.get()
            },
        )
        classpath = sourceSets.main.get().runtimeClasspath
    }
}

ktlint {
    version.set("1.8.0")
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}
