import asyncio
import functools
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from celery import Task
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.tasks.celery import celery_app

# 创建异步数据库引擎
async_engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


class BaseTask(Task):
    """基础任务类，提供公共功能"""

    abstract = True  # 抽象类，不会被注册为任务

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败处理"""
        logger.error(
            f"任务失败: {self.name}[{task_id}], 异常: {exc}, 参数: {args}, {kwargs}"
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        """任务成功处理"""
        logger.info(f"任务成功: {self.name}[{task_id}], 结果: {retval}")
        super().on_success(retval, task_id, args, kwargs)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """任务重试处理"""
        logger.warning(
            f"任务重试: {self.name}[{task_id}], 异常: {exc}, 参数: {args}, {kwargs}"
        )
        super().on_retry(exc, task_id, args, kwargs, einfo)


def async_task(
    *celery_args,
    name=None,
    queue=None,
    retry_backoff=True,
    max_retries=3,
    bind=True,  # 默认绑定self参数以兼容现有代码
    **celery_kwargs,
):
    """异步任务装饰器，支持异步函数"""

    def decorator(async_func):
        # 创建同步包装函数
        @functools.wraps(async_func)
        def sync_wrapper(*args, **kwargs):
            # 检查是否已有事件循环运行
            try:
                loop = asyncio.get_running_loop()
                logger.debug(f"检测到运行中的事件循环: {loop}")
                # 如果已有事件循环，使用新的循环在线程中运行
                import threading
                import concurrent.futures
                
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(async_func(*args, **kwargs))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_new_loop)
                    return future.result()
                    
            except RuntimeError:
                # 没有运行中的事件循环，可以安全使用 asyncio.run
                logger.debug("没有检测到运行中的事件循环，使用 asyncio.run")
                return asyncio.run(async_func(*args, **kwargs))

        # 设置Celery任务参数
        task_options = {
            "base": BaseTask,
            "bind": bind,  # 使用传入的bind参数
            "autoretry_for": (Exception,),
            "retry_backoff": retry_backoff,
            "retry_kwargs": {"max_retries": max_retries},
        }

        # 添加队列和名称，如果指定
        if queue:
            task_options["queue"] = queue
        if name:
            task_options["name"] = name

        # 合并自定义参数（bind参数会被celery_kwargs中的同名参数覆盖）
        task_options.update(celery_kwargs)

        # 创建并注册Celery任务
        return celery_app.task(*celery_args, **task_options)(sync_wrapper)

    return decorator


async def get_async_db_session():
    """获取异步数据库会话上下文管理器"""
    async with AsyncSessionLocal() as session:
        yield session
