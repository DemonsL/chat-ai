import json
from typing import Any, Optional

import redis.asyncio as redis
from loguru import logger

from app.core.config import settings
from app.plugins.cache.base import CacheProvider


class RedisCache(CacheProvider):
    def __init__(self):
        self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        if settings.REDIS_PASSWORD:
            self.redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """连接到Redis服务器"""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    async def disconnect(self) -> None:
        """断开与Redis服务器的连接"""
        if self.client:
            await self.client.close()
            logger.info("Redis连接已关闭")

    async def get(self, key: str) -> Any:
        """获取缓存项"""
        if not self.client:
            await self.connect()
        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存项"""
        if not self.client:
            await self.connect()

        if not isinstance(value, (str, bytes, int, float)):
            value = json.dumps(value)

        if expire:
            return await self.client.setex(key, expire, value)
        return await self.client.set(key, value)

    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        if not self.client:
            await self.connect()
        return bool(await self.client.delete(key))

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.client:
            await self.connect()
        return bool(await self.client.exists(key))

    async def clear(self) -> bool:
        """清除所有缓存"""
        if not self.client:
            await self.connect()
        return bool(await self.client.flushdb())
