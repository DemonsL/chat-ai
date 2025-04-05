from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class CacheProvider(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """连接到缓存服务器"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开与缓存服务器的连接"""
        pass

    @abstractmethod
    async def get(self, key: str) -> Any:
        """获取缓存项"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存项"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """清除所有缓存"""
        pass
