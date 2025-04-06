from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.tasks.celery import celery_app


class TaskMonitorService:
    """
    Celery任务监控服务
    """

    def __init__(self, db_session: AsyncSession = None):
        self.db_session = db_session

    async def get_task_status(self, task_id: str) -> Dict:
        """
        获取任务状态

        参数:
            task_id: 任务ID

        返回:
            任务状态信息
        """
        # 从Celery获取任务状态
        async_result = celery_app.AsyncResult(task_id)

        result = {
            "task_id": task_id,
            "status": async_result.status,
            "successful": async_result.successful(),
        }

        # 如果任务成功，获取结果
        if async_result.ready():
            if async_result.successful():
                try:
                    result["result"] = async_result.result
                except Exception as e:
                    result["result"] = str(e)
            else:
                result["error"] = str(async_result.result)

        return result

    async def get_active_tasks(self) -> List[Dict]:
        """
        获取当前活动的任务

        返回:
            活动任务列表
        """
        # 通过Celery检查活动任务
        # 这需要启用events: celery worker -E
        inspector = celery_app.control.inspect()

        # 获取各种状态的任务
        active_tasks = inspector.active() or {}
        reserved_tasks = inspector.reserved() or {}
        scheduled_tasks = inspector.scheduled() or {}

        # 合并所有任务
        all_tasks = []

        # 添加活动任务
        for worker, tasks in active_tasks.items():
            for task in tasks:
                all_tasks.append(
                    {
                        "id": task["id"],
                        "name": task["name"],
                        "worker": worker,
                        "status": "active",
                        "args": task.get("args", []),
                        "kwargs": task.get("kwargs", {}),
                        "started_at": task.get("time_start"),
                    }
                )

        # 添加预留任务
        for worker, tasks in reserved_tasks.items():
            for task in tasks:
                all_tasks.append(
                    {
                        "id": task["id"],
                        "name": task["name"],
                        "worker": worker,
                        "status": "reserved",
                        "args": task.get("args", []),
                        "kwargs": task.get("kwargs", {}),
                    }
                )

        # 添加计划任务
        for worker, tasks in scheduled_tasks.items():
            for task in tasks:
                all_tasks.append(
                    {
                        "id": task["id"],
                        "name": task["name"],
                        "worker": worker,
                        "status": "scheduled",
                        "args": task.get("args", []),
                        "kwargs": task.get("kwargs", {}),
                        "eta": task.get("eta"),
                    }
                )

        return all_tasks

    async def revoke_task(self, task_id: str, terminate: bool = False) -> bool:
        """
        撤销任务

        参数:
            task_id: 任务ID
            terminate: 是否终止正在运行的任务

        返回:
            是否成功
        """
        celery_app.control.revoke(task_id, terminate=terminate)
        return True

    async def get_task_result(self, task_id: str) -> Optional[Dict]:
        """
        获取任务结果

        参数:
            task_id: 任务ID

        返回:
            任务结果
        """
        async_result = celery_app.AsyncResult(task_id)

        if not async_result.ready():
            return None

        if async_result.successful():
            try:
                return async_result.result
            except Exception:
                return None
        else:
            return None
