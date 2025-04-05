from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.task_routes = {"app.tasks.jobs.*": "main-queue"}
celery_app.conf.result_expires = 60 * 60 * 24  # 1 day

# 导入任务模块，使任务对Celery可见
celery_app.autodiscover_tasks(["app.tasks.jobs"])
