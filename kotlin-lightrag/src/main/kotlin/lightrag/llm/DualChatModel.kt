package lightrag.llm

import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.model.ModelProvider
import dev.langchain4j.model.chat.Capability
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.listener.ChatModelListener
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.request.ChatRequestParameters
import dev.langchain4j.model.chat.response.ChatResponse

/**
 * Adapter that exposes the non-streaming [ChatModel] implementation.
 *
 * LangChain4j's streaming interface now uses publisher return types that collide
 * with the synchronous method signatures, so the two interfaces need separate adapters.
 */
class DualChatModel(
    private val chatModel: ChatModel,
) : ChatModel {
    // ChatModel delegate
    override fun chat(request: ChatRequest): ChatResponse = chatModel.chat(request)

    override fun doChat(request: ChatRequest): ChatResponse = chatModel.doChat(request)

    override fun defaultRequestParameters(): ChatRequestParameters = chatModel.defaultRequestParameters()

    override fun listeners(): List<ChatModelListener> = chatModel.listeners()

    override fun provider(): ModelProvider = chatModel.provider()

    override fun chat(message: String): String = chatModel.chat(message)

    override fun chat(vararg messages: ChatMessage): ChatResponse = chatModel.chat(*messages)

    override fun chat(messages: List<ChatMessage>): ChatResponse = chatModel.chat(messages)

    override fun supportedCapabilities(): Set<Capability> = chatModel.supportedCapabilities()
}
