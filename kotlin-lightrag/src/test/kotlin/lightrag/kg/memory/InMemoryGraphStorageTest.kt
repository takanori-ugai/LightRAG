package lightrag.kg.memory

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class InMemoryGraphStorageTest {
    @Test
    fun `upsert edge stores undirected relation and deduplicates getAllEdges`() {
        runBlocking {
            val storage = InMemoryGraphStorage(namespace = "ns", workspace = "ws")
            storage.upsertNode("A", mapOf("label" to "A"))
            storage.upsertNode("B", mapOf("label" to "B"))

            val edgeData = mapOf("type" to "knows")
            storage.upsertEdge("A", "B", edgeData)

            assertTrue(storage.hasEdge("A", "B"))
            assertTrue(storage.hasEdge("B", "A"))

            val edges = storage.getAllEdges()
            assertEquals(1, edges.size, "Edges should be deduped for undirected storage")
            val stored = edges.first()
            assertEquals("A", stored["source"])
            assertEquals("B", stored["target"])
            assertEquals("knows", stored["type"])

            val nodeEdges = storage.getNodeEdges("A")
            assertNotNull(nodeEdges)
            assertEquals(listOf("A" to "B"), nodeEdges)
            assertNull(storage.getNodeEdges("missing"))
        }
    }

    @Test
    fun `delete node removes all connected edges`() {
        runBlocking {
            val storage = InMemoryGraphStorage(namespace = "ns", workspace = "ws")
            storage.upsertNode("A", mapOf())
            storage.upsertNode("B", mapOf())
            storage.upsertNode("C", mapOf())
            storage.upsertEdge("A", "B", mapOf())
            storage.upsertEdge("B", "C", mapOf())

            storage.deleteNode("B")

            assertFalse(storage.hasNode("B"))
            assertFalse(storage.hasEdge("A", "B"))
            assertFalse(storage.hasEdge("B", "A"))
            assertFalse(storage.hasEdge("B", "C"))
            assertFalse(storage.hasEdge("C", "B"))
            assertEquals(emptyList(), storage.getNodeEdges("A"))
        }
    }

    @Test
    fun `removeEdges clears both directions without deleting nodes`() {
        runBlocking {
            val storage = InMemoryGraphStorage(namespace = "ns", workspace = "ws")
            storage.upsertNode("A", mapOf())
            storage.upsertNode("B", mapOf())
            storage.upsertEdge("A", "B", mapOf("weight" to "1"))

            storage.removeEdges(listOf("A" to "B"))

            assertTrue(storage.hasNode("A"))
            assertTrue(storage.hasNode("B"))
            assertFalse(storage.hasEdge("A", "B"))
            assertFalse(storage.hasEdge("B", "A"))
            assertEquals(emptyList(), storage.getAllEdges())
        }
    }

    @Test
    fun `popular and search labels reflect degrees and query matching`() {
        runBlocking {
            val storage = InMemoryGraphStorage(namespace = "ns", workspace = "ws")
            storage.upsertNode("alpha", mapOf())
            storage.upsertNode("beta", mapOf())
            storage.upsertNode("gamma", mapOf())
            storage.upsertEdge("alpha", "beta", mapOf())
            storage.upsertEdge("alpha", "gamma", mapOf())

            val popular = storage.getPopularLabels(limit = 2)
            assertEquals("alpha", popular.first(), "Node with highest degree should be first")
            assertEquals(2, popular.size)

            val matches = storage.searchLabels("a", limit = 3)
            assertTrue(matches.containsAll(listOf("alpha", "gamma")))
        }
    }

    @Test
    fun `knowledge graph placeholder returns empty graph`() {
        runBlocking {
            val storage = InMemoryGraphStorage(namespace = "ns", workspace = "ws")
            val kg = storage.getKnowledgeGraph(nodeLabel = "any", maxDepth = 3, maxNodes = 10)
            assertTrue(kg.nodes.isEmpty())
            assertTrue(kg.edges.isEmpty())
            assertFalse(kg.isTruncated)
        }
    }
}
