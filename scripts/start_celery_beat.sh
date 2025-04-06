#!/bin/bash
# 启动Celery Beat

# 确保正确的Python环境
source .venv/bin/activate

# 启动Celery Beat
# -A: 指定应用
# -l: 日志级别
# --scheduler: 指定调度器
celery -A app.tasks.celery:celery_app beat \
  -l INFO \
  --scheduler django_celery_beat.schedulers:DatabaseScheduler