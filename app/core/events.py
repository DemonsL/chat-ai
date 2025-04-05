from loguru import logger

from app.db.session import engine
from app.plugins.cache.manager import get_cache_client

# from app.plugins.logging.manager import configure_logging


async def startup_event_handler() -> None:
    """
    应用启动事件处理函数
    """
    logger.info("启动应用...")

    # 配置日志
    # configure_logging()

    # 初始化缓存连接
    await get_cache_client().connect()

    logger.info("应用启动完成")


async def shutdown_event_handler() -> None:
    """
    应用关闭事件处理函数
    """
    logger.info("关闭应用...")

    # 关闭数据库连接
    await engine.dispose()

    # 关闭缓存连接
    await get_cache_client().disconnect()

    logger.info("应用已关闭")
