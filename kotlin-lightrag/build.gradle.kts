plugins {
    kotlin("jvm") version "1.9.22"
    id("io.ktor.plugin") version "2.3.9"
    kotlin("plugin.serialization") version "1.9.22"
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
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:1.9.22")
}
