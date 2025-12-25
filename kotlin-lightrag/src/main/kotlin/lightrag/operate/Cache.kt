package lightrag.operate

import lightrag.core.CacheData
import lightrag.core.types.BaseKVStorage
import lightrag.utils.JsonUtils

suspend fun saveToCache(
    storage: BaseKVStorage,
    data: CacheData,
) {
    val cacheValue =
        mapOf(
            "content" to data.content,
            "prompt" to data.prompt,
            "mode" to data.mode,
            "cache_type" to data.cacheType,
            "queryparam" to JsonUtils.convertObjectToJson(data.queryParam),
        )
    storage.upsert(mapOf(data.argsHash to cacheValue))
}
