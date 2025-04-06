from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from app.tasks.celery import celery_app


class TaskScheduler:
    """任务调度管理器"""

    @staticmethod
    def delay_task(
        task_name: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        eta: Optional[datetime] = None,
        countdown: Optional[int] = None,
        queue: Optional[str] = None,
        priority: Optional[int] = None,
    ):
        """延迟执行任务

        参数:
            task_name: 任务名称
            args: 位置参数
            kwargs: 关键字参数
            eta: 指定执行时间
            countdown: 指定延迟秒数
            queue: 指定队列
            priority: 优先级(0-9)

        返回:
            任务实例
        """
        args = args or ()
        kwargs = kwargs or {}

        options = {}
        if eta:
            options["eta"] = eta
        elif countdown:
            options["countdown"] = countdown

        if queue:
            options["queue"] = queue

        if priority is not None:
            options["priority"] = priority

        task = celery_app.signature(
            task_name, args=args, kwargs=kwargs, options=options
        )
        return task.apply_async()

    @staticmethod
    def schedule_periodic(
        task_name: str,
        schedule: Union[int, timedelta, str],
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        name: Optional[str] = None,
        queue: Optional[str] = None,
    ):
        """动态添加定时任务
        注意: 这需要重启celery beat才能生效

        参数:
            task_name: 任务名称
            schedule: 调度周期(秒数、timedelta或crontab表达式)
            args: 位置参数
            kwargs: 关键字参数
            name: 任务标识名
            queue: 指定队列
        """
        args = args or ()
        kwargs = kwargs or {}
        name = name or f"{task_name}-{datetime.now().timestamp()}"

        # 创建调度条目
        options = {"queue": queue} if queue else {}
        entry = {
            "task": task_name,
            "schedule": schedule,
            "args": args,
            "kwargs": kwargs,
            "options": options,
        }

        # 添加到调度器
        celery_app.conf.beat_schedule[name] = entry
        return name

    @staticmethod
    def cancel_task(task_id: str):
        """取消任务

        参数:
            task_id: 任务ID
        """
        celery_app.control.revoke(task_id, terminate=True)
        return True
