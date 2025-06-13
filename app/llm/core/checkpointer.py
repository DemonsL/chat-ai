"""
LangGraph Checkpointer 配置模块
提供 AsyncPostgresSaver 和 PostgresSaver 的配置和管理功能
"""

import logging
import asyncio
import sys
from typing import Optional, Union
from uuid import UUID

# Windows兼容性修复：设置正确的事件循环策略
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logger = logging.getLogger(__name__)
        logger.debug("已设置Windows兼容的事件循环策略")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"设置Windows事件循环策略失败: {e}")

# 将PostgresSaver导入变为可选
try:
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    ASYNC_POSTGRES_AVAILABLE = True
except ImportError:
    AsyncPostgresSaver = None
    AsyncConnectionPool = None
    ASYNC_POSTGRES_AVAILABLE = False

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    SYNC_POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None
    SYNC_POSTGRES_AVAILABLE = False

POSTGRES_AVAILABLE = ASYNC_POSTGRES_AVAILABLE or SYNC_POSTGRES_AVAILABLE

if not POSTGRES_AVAILABLE:
    import warnings
    warnings.warn("PostgresSaver 不可用. 将使用 MemorySaver 作为备选方案。")

from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings

logger = logging.getLogger(__name__)


class CheckpointerManager:
    """Checkpointer 管理器"""
    
    def __init__(self):
        self._memory_saver: Optional[MemorySaver] = None
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._use_postgres: bool = getattr(settings, 'USE_POSTGRES_CHECKPOINTER', True) and POSTGRES_AVAILABLE
        self._prefer_async: bool = getattr(settings, 'PREFER_ASYNC_POSTGRES', True) and ASYNC_POSTGRES_AVAILABLE
        
        if not POSTGRES_AVAILABLE and getattr(settings, 'USE_POSTGRES_CHECKPOINTER', True):
            logger.warning("PostgreSQL checkpointer 依赖不可用，自动使用 MemorySaver")
    
    async def get_checkpointer(self, conversation_id: Optional[UUID] = None) -> Union[AsyncPostgresSaver, PostgresSaver, MemorySaver]:
        """获取 checkpointer 实例（异步版本）"""
        if self._use_postgres and POSTGRES_AVAILABLE:
            if self._prefer_async and ASYNC_POSTGRES_AVAILABLE:
                try:
                    return await self._create_async_postgres_saver()
                except Exception as e:
                    logger.warning(f"AsyncPostgresSaver 创建失败，尝试同步版本: {e}")
                    if SYNC_POSTGRES_AVAILABLE:
                        return self._create_sync_postgres_saver()
                    else:
                        logger.info("同步PostgresSaver不可用，使用内存checkpointer")
                        return self._get_memory_saver()
            elif SYNC_POSTGRES_AVAILABLE:
                return self._create_sync_postgres_saver()
            else:
                logger.info("PostgreSQL checkpointer 不可用，使用内存 checkpointer")
                return self._get_memory_saver()
        else:
            logger.info("未配置 PostgreSQL 或已禁用，使用内存 checkpointer")
            return self._get_memory_saver()
    
    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """获取PostgreSQL连接池"""
        if self._connection_pool is None:
            try:
                # Windows兼容性修复：在创建连接池前设置事件循环策略
                if sys.platform == "win32":
                    try:
                        current_policy = asyncio.get_event_loop_policy()
                        if not isinstance(current_policy, asyncio.WindowsSelectorEventLoopPolicy):
                            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                            logger.debug("已设置Windows兼容的事件循环策略")
                    except Exception as e:
                        logger.warning(f"设置Windows事件循环策略失败: {e}")
                
                # 获取数据库连接URL
                database_url = getattr(settings, 'POSTGRES_DATABASE_URL', None)
                if not database_url:
                    raise ValueError("POSTGRES_DATABASE_URL 未配置")
                
                # 配置连接池大小
                max_size = getattr(settings, 'POSTGRES_POOL_SIZE', 5)

                self._connection_pool = AsyncConnectionPool(
                    database_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 10,  # 增加超时时间
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info(f"异步连接池创建成功，最大连接数: {max_size}")
            except Exception as e:
                logger.error(f"异步连接池创建失败: {e}")
                raise e
        return self._connection_pool

    async def _create_async_postgres_saver(self):
        """创建新的 AsyncPostgresSaver 实例"""
        if not ASYNC_POSTGRES_AVAILABLE:
            raise RuntimeError("AsyncPostgresSaver 依赖不可用")
            
        try:
            connection_pool = await self._get_connection_pool()
            checkpointer = AsyncPostgresSaver(connection_pool)
            
            # 确保数据库表已创建（只在第一次需要）
            try:
                await checkpointer.setup()
            except Exception as setup_error:
                # 如果setup失败，表可能已经存在，继续使用
                logger.debug(f"AsyncPostgresSaver setup 警告: {setup_error}")
            
            logger.debug("创建了新的 AsyncPostgresSaver 实例")
            return checkpointer
            
        except Exception as e:
            logger.error(f"AsyncPostgresSaver 创建失败: {e}")
            raise
    
    def _create_sync_postgres_saver(self):
        """创建同步 PostgresSaver 实例（Windows兼容）"""
        if not SYNC_POSTGRES_AVAILABLE:
            raise RuntimeError("PostgresSaver 依赖不可用")
            
        try:
            # 获取数据库连接URL
            database_url = getattr(settings, 'POSTGRES_DATABASE_URL', None)
            if not database_url:
                raise ValueError("POSTGRES_DATABASE_URL 未配置")
            
            # 创建同步PostgresSaver
            postgres_saver_context = PostgresSaver.from_conn_string(database_url)
            postgres_saver = postgres_saver_context.__enter__()
            
            # 确保数据库表已创建
            try:
                postgres_saver.setup()
            except Exception as setup_error:
                logger.debug(f"PostgresSaver setup 警告: {setup_error}")
            
            logger.debug("创建了新的 PostgresSaver 实例（同步版本）")
            return postgres_saver
            
        except Exception as e:
            logger.error(f"PostgresSaver 创建失败: {e}")
            raise
    
    def _get_memory_saver(self) -> MemorySaver:
        """获取或创建 MemorySaver 实例"""
        if self._memory_saver is None:
            self._memory_saver = MemorySaver()
            logger.info("MemorySaver 初始化成功")
        
        return self._memory_saver
    
    async def clear_checkpoint(self, conversation_id: UUID):
        """清除指定对话的检查点（异步版本）"""
        try:
            checkpointer = await self.get_checkpointer()
            if hasattr(checkpointer, 'adelete'):
                # AsyncPostgresSaver 支持异步删除特定对话的检查点
                await checkpointer.adelete({"configurable": {"thread_id": str(conversation_id)}})
                logger.info(f"已清除对话 {conversation_id} 的检查点")
            elif hasattr(checkpointer, 'delete'):
                # 同步删除方法
                checkpointer.delete({"configurable": {"thread_id": str(conversation_id)}})
                logger.info(f"已清除对话 {conversation_id} 的检查点")
            else:
                logger.warning("当前 checkpointer 不支持删除操作")
        except Exception as e:
            logger.error(f"清除检查点失败: {e}")
    
    def get_conversation_config(self, conversation_id: UUID) -> dict:
        """获取对话配置"""
        return {
            "configurable": {
                "thread_id": str(conversation_id),
                "checkpoint_ns": "chat-ai"  # 命名空间，避免与其他应用冲突
            }
        }
    
    async def close(self):
        """关闭连接池"""
        if self._connection_pool:
            try:
                await self._connection_pool.close()
                logger.info("PostgreSQL 连接池已关闭")
            except Exception as e:
                logger.error(f"关闭连接池失败: {e}")
        
        self._connection_pool = None
        self._memory_saver = None
    
    def is_postgres_available(self) -> bool:
        """检查 PostgreSQL checkpointer 是否可用"""
        return POSTGRES_AVAILABLE and self._use_postgres


# 全局 checkpointer 管理器实例
checkpointer_manager = CheckpointerManager()


async def get_checkpointer(conversation_id: Optional[UUID] = None):
    """便捷函数：获取 checkpointer 实例（异步版本）"""
    return await checkpointer_manager.get_checkpointer(conversation_id)


def get_conversation_config(conversation_id: UUID) -> dict:
    """便捷函数：获取对话配置"""
    return checkpointer_manager.get_conversation_config(conversation_id)


async def clear_conversation_checkpoint(conversation_id: UUID):
    """便捷函数：清除对话检查点（异步版本）"""
    await checkpointer_manager.clear_checkpoint(conversation_id)


def is_postgres_available() -> bool:
    """便捷函数：检查 PostgreSQL 是否可用"""
    return checkpointer_manager.is_postgres_available() 