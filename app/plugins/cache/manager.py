from typing import Optional

from app.plugins.cache.base import CacheProvider
from app.plugins.cache.providers.redis import RedisCache

# 单例缓存客户端
_cache_client: Optional[CacheProvider] = None


def get_cache_client() -> CacheProvider:
    """
    获取缓存客户端实例
    """
    global _cache_client
    if _cache_client is None:
        _cache_client = RedisCache()
    return _cache_client
