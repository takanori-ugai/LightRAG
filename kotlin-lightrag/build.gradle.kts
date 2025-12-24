plugins {
    kotlin("jvm") version "1.9.22"
    id("io.ktor.plugin") version "2.3.9"
    kotlin("plugin.serialization") version "1.9.22"
    id("org.jlleitschuh.gradle.ktlint") version "12.1.0"
}

group = "com.lightrag"
version = "0.0.1"

application {
    mainClass.set("lightrag.ApplicationKt")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.ktor:ktor-server-core:2.3.9")
    implementation("io.ktor:ktor-server-netty:2.3.9")
    implementation("io.ktor:ktor-server-content-negotiation:2.3.9")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.9")
    implementation("io.ktor:ktor-server-swagger:2.3.9")
    implementation("io.ktor:ktor-server-cors:2.3.9")
    implementation("ch.qos.logback:logback-classic:1.4.14")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
    implementation("io.github.oshai:kotlin-logging-jvm:5.1.0")

    // LangChain4j dependencies
    implementation("dev.langchain4j:langchain4j-core:0.31.0")
    implementation("dev.langchain4j:langchain4j-open-ai:0.31.0")
    implementation("dev.langchain4j:langchain4j-ollama:0.31.0")

    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:1.9.22")
}
