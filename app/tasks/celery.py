from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue

from app.core.config import settings

# 定义交换机
default_exchange = Exchange("default", type="direct")
priority_exchange = Exchange("priority", type="direct")

# 定义队列
task_queues = (
    Queue("default", default_exchange, routing_key="default"),  # 默认队列
    Queue("priority", priority_exchange, routing_key="priority"),  # 高优先级队列
    Queue("file_tasks", default_exchange, routing_key="file"),  # 文件处理队列
    Queue("email_tasks", default_exchange, routing_key="email"),  # 邮件队列
    Queue("inventory", default_exchange, routing_key="inventory"),  # 库存管理队列
    Queue("credits", default_exchange, routing_key="credits"),  # 积分管理队列
    Queue("export", default_exchange, routing_key="export"),  # 数据导出队列
    Queue("api_calls", default_exchange, routing_key="api"),  # API调用队列
    Queue("scheduled", default_exchange, routing_key="scheduled"),  # 定时任务队列
)

# 创建Celery应用
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# 配置
celery_app.conf.update(
    # 队列配置
    task_queues=task_queues,
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # 任务路由
    task_routes={
        "tasks.file.*": {"queue": "file_tasks"},
        "tasks.email.*": {"queue": "email_tasks"},
        "tasks.api.*": {"queue": "api_calls"},
        "app.tasks.jobs.file.*": {"queue": "file_tasks"},
        "app.tasks.jobs.email.*": {"queue": "email_tasks"},
        "app.tasks.jobs.inventory.*": {"queue": "inventory"},
        "app.tasks.jobs.credits.*": {"queue": "credits"},
        "app.tasks.jobs.export.*": {"queue": "export"},
        "app.tasks.jobs.api.*": {"queue": "api_calls"},
        "app.tasks.jobs.scripts.*": {"queue": "default"},
    },
    # 任务执行设置
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_time_limit=3600,  # 1小时
    task_soft_time_limit=3000,  # 50分钟
    worker_max_tasks_per_child=200,
    # 结果存储
    result_expires=60 * 60 * 24 * 7,  # 一周
    # 任务跟踪和监控
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
    # 错误处理和重试
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# 定义定时任务
beat_schedule = {
    # 每天凌晨3点执行数据库备份
    "daily-database-backup": {
        "task": "app.tasks.jobs.scripts.database_backup_task",
        "schedule": crontab(hour=3, minute=0),
        "kwargs": {"backup_type": "full"},
        "options": {"queue": "scheduled"},
    },
    # 每天检查低库存产品
    "daily-check-low-stock": {
        "task": "app.tasks.jobs.inventory.check_low_stock_task",
        "schedule": crontab(hour=7, minute=30),
        "kwargs": {"threshold": 10},
        "options": {"queue": "scheduled"},
    },
    # 每周日凌晨2点清理临时文件
    # "weekly-cleanup-temp-files": {
    #     "task": "app.tasks.jobs.scripts.run_shell_script_task",
    #     "schedule": crontab(hour=2, minute=0, day_of_week=0),
    #     "kwargs": {
    #         "script_path": "/scripts/cleanup_temp_files.sh",
    #         "arguments": [settings.TEMP_DIR, "--older-than=7d"]
    #     },
    #     "options": {"queue": "scheduled"},
    # },
    # 每小时更新系统状态统计
    "hourly-system-stats-update": {
        "task": "app.tasks.jobs.stats.update_system_stats_task",
        "schedule": crontab(minute=5),
        "options": {"queue": "scheduled"},
    },
}

# 将定时任务添加到Celery配置
celery_app.conf.beat_schedule = beat_schedule

# 导入任务模块，使任务对Celery可见
celery_app.autodiscover_tasks(["app.tasks.jobs"])
