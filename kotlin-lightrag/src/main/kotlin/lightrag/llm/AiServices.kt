package lightrag.llm

import dev.langchain4j.service.SystemMessage
import dev.langchain4j.service.UserMessage
import dev.langchain4j.service.V
import lightrag.operate.ExtractionResult
import lightrag.operate.KeywordsExtractionResult
import lightrag.utils.Prompts

/**
 * An AI service for extracting entities and relationships from text.
 */
interface EntityExtractor {
    /**
     * Extracts entities and relationships from the given text.
     * @param text The text to extract from.
     * @param entityTypes The types of entities to extract.
     * @param language The language of the text.
     * @return An [ExtractionResult] containing the extracted entities and relationships.
     */
    @SystemMessage(
        Prompts.ENTITY_EXTRACTION_SYSTEM_PROMPT,
    )
    fun extract(
        @V("input_text")
        @UserMessage(
            Prompts.ENTITY_EXTRACTION_USER_PROMPT,
        )
        text: String,
        @V("entity_types") entityTypes: String,
        @V("language") language: String,
    ): ExtractionResult
}

/**
 * An AI service for extracting keywords from text.
 */
interface KeywordExtractor {
    /**
     * Extracts keywords from the given text.
     * @param text The text to extract from.
     * @param language The language of the text.
     * @param examples Examples of keywords to extract.
     * @return A [KeywordsExtractionResult] containing the extracted keywords.
     */
    @SystemMessage(
        Prompts.KEYWORDS_EXTRACTION,
    )
    fun extract(
        @V("query")
        @UserMessage(
            """User Query: {{query}}""",
        )
        text: String,
        @V("language") language: String,
        @V("examples") examples: String,
    ): KeywordsExtractionResult
}
