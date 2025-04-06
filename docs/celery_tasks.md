# Celery 异步任务系统

本文档说明如何配置和运行 Celery 异步任务系统。

## 前提条件

- RabbitMQ 服务器已启动并配置
- Redis 服务器已启动并配置 
- 项目环境变量中已配置好 `CELERY_BROKER_URL` 和 `CELERY_RESULT_BACKEND`

## 启动 Celery Worker

Celery Worker 负责执行异步任务。

```bash
# 启动默认 Worker
celery -A app.tasks.celery:celery_app worker -l INFO -c 4 -n worker1@%h -E

# 启动特定队列的 Worker
# 例如：文件处理队列和邮件队列
celery -A app.tasks.celery:celery_app worker -l INFO -c 2 -n file_worker@%h -Q file_tasks,export

# 启动高优先级队列的 Worker
celery -A app.tasks.celery:celery_app worker -l INFO -c 2 -n priority_worker@%h -Q priority
```

## 启动 Celery Beat

Celery Beat 负责调度定时任务。

```bash
celery -A app.tasks.celery:celery_app beat -l INFO
```

## 监控 Celery 任务

您可以使用 Flower 来监控 Celery 任务：

```bash
celery -A app.tasks.celery:celery_app flower --port=5555
```

然后访问 http://localhost:5555 查看 Celery 任务监控页面。

## 可用的异步任务队列

本系统配置了以下任务队列：

1. `default` - 默认队列
2. `priority` - 高优先级队列
3. `file_tasks` - 文件处理队列
4. `email_tasks` - 邮件队列
5. `inventory` - 库存管理队列
6. `credits` - 积分管理队列
7. `export` - 数据导出队列
8. `api_calls` - API调用队列
9. `scheduled` - 定时任务队列

您可以为不同的队列启动专用 Worker 以优化资源分配。

## 定时任务

系统已配置以下定时任务：

1. 每天凌晨3点执行数据库备份
2. 每天早7:30检查低库存产品
3. 每周日凌晨2点清理临时文件
4. 每小时5分更新系统状态统计 