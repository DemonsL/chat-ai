import os
from datetime import datetime, timedelta

from loguru import logger

from app.core.config import settings
from app.tasks.base import async_task, get_async_db_session


@async_task(name="tasks.stats.update_system_stats", queue="scheduled")
async def update_system_stats_task(self):
    """
    更新系统统计数据
    """
    async for session in get_async_db_session():
        from app.db.repositories.conversation_repository import \
            ConversationRepository
        from app.db.repositories.message_repository import MessageRepository
        from app.db.repositories.user_file_repository import UserFileRepository
        from app.db.repositories.user_repository import UserRepository

        # 获取各个仓库
        user_repo = UserRepository(session)
        file_repo = UserFileRepository(session)
        message_repo = MessageRepository(session)
        conversation_repo = ConversationRepository(session)

        try:
            # 获取各种统计数据
            user_count = await user_repo.count()
            active_user_count = await user_repo.count_active()
            file_count = await file_repo.count()

            # 获取过去24小时的统计
            yesterday = datetime.now() - timedelta(days=1)
            new_user_count = await user_repo.count_created_after(yesterday)
            new_file_count = await file_repo.count_created_after(yesterday)

            # 获取消息统计
            message_count = await message_repo.count()
            conversation_count = await conversation_repo.count()

            # 获取存储统计
            storage_stats = {"total_storage": 0, "upload_dir_size": 0}

            # 计算上传目录大小
            if os.path.exists(settings.UPLOAD_DIR):
                total_size = 0
                for path, dirs, files in os.walk(settings.UPLOAD_DIR):
                    for f in files:
                        fp = os.path.join(path, f)
                        total_size += os.path.getsize(fp)
                storage_stats["upload_dir_size"] = total_size

            # 构建统计数据
            stats = {
                "timestamp": datetime.now().isoformat(),
                "users": {
                    "total": user_count,
                    "active": active_user_count,
                    "new_24h": new_user_count,
                },
                "files": {"total": file_count, "new_24h": new_file_count},
                "conversations": {"total": conversation_count},
                "messages": {"total": message_count},
                "storage": storage_stats,
            }

            # 将统计数据保存到缓存或数据库
            # 这里示例使用Redis缓存
            from app.plugins.cache.manager import get_cache_client

            cache = get_cache_client()
            await cache.set("system:stats:latest", stats, expire=86400 * 7)  # 保存7天

            # 同时保存当前时间点的统计数据
            timestamp_key = f"system:stats:{datetime.now().strftime('%Y%m%d%H')}"
            await cache.set(timestamp_key, stats, expire=86400 * 30)  # 保存30天

            logger.info(f"系统统计数据已更新: {stats}")
            return stats

        except Exception as e:
            logger.error(f"更新系统统计数据失败: {str(e)}")
            raise e
