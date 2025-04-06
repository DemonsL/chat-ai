#!/bin/bash
# 启动Celery Worker

# 确保正确的Python环境
source .venv/bin/activate

# 启动Celery Worker
# -A: 指定应用
# -l: 日志级别
# -Q: 指定队列，可以运行特定队列，这里运行所有队列
# -c: 并发worker数量
# -n: worker名称
# -E: 启用事件，用于监控
celery -A app.tasks.celery:celery_app worker \
  -l INFO \
  -c 4 \
  -n worker1@%h \
  -E