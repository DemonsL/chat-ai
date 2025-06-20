# Celery 异步任务系统配置指南

## 概述

本项目使用 Celery 作为异步任务处理系统，支持文件处理、邮件发送、数据分析等后台任务。

## 关键修复说明

### 1. 任务参数绑定问题修复

**问题**: 异步任务执行时出现参数错误，`self` 参数传递混乱。

**解决方案**: 
- 修改文件任务装饰器，使用 `bind=False` 避免 `self` 参数混乱
- 其他任务保持 `bind=True` 以维持兼容性

### 2. 事件循环冲突修复

**问题**: `asyncio.run()` 与现有事件循环冲突，导致 "attached to a different loop" 错误。

**解决方案**:
- 在 `async_task` 装饰器中添加事件循环检测
- 如果检测到运行中的事件循环，在新线程中创建独立事件循环
- 确保数据库连接和异步操作在正确的循环中执行

## 系统要求

- Python 3.8+
- Redis (消息代理)
- PostgreSQL (数据库)
- 所需Python包: celery, redis, sqlalchemy, asyncpg

## 启动步骤

### 1. 启动依赖服务

```bash
# 启动 Redis (端口 6380)
redis-server --port 6380

# 确保 PostgreSQL 运行
```

### 2. 启动 Celery Worker

```bash
# 在项目根目录下
python run_celery_worker.py
```

### 3. 验证启动

查看日志输出，确认以下信息：
- 所有任务模块导入成功
- Worker 连接到 Redis
- 关键任务已注册

## 任务类型

### 文件处理任务 (`tasks.file.*`)

- `process_file`: 处理上传的文件，创建向量索引
- `analyze_file`: 分析文件内容
- `export_file`: 导出多个文件为ZIP
- `bulk_upload`: 批量处理服务器文件

### 邮件任务 (`tasks.email.*`)

- `send_email`: 发送单个邮件
- `send_notification`: 发送通知邮件给多个收件人

### 其他任务

- `tasks.credits.*`: 积分相关操作
- `tasks.api.*`: 外部API调用
- `tasks.stats.*`: 系统统计

## 配置说明

### Celery 配置 (`app/tasks/celery.py`)

```python
# 关键配置项
broker_url = "redis://localhost:6380/0"  # Redis 代理
result_backend = "redis://localhost:6380/0"  # 结果存储
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "Asia/Shanghai"
enable_utc = True
```

### 任务路由配置

```python
task_routes = {
    "tasks.file.*": {"queue": "file_tasks"},
    "tasks.email.*": {"queue": "email_tasks"},
    "tasks.api.*": {"queue": "api_calls"},
    "tasks.credits.*": {"queue": "credits"},
    # 导出任务使用专用队列
    "tasks.file.export_file": {"queue": "export"},
}
```

## 故障排除

### 1. 任务执行失败

**症状**: 任务状态为 FAILURE，错误信息为任务名称字符串

**检查步骤**:
1. 确认 Worker 进程正在运行
2. 检查任务是否正确注册: `celery_app.tasks.keys()`
3. 验证任务名称是否匹配 (`tasks.file.process_file`)
4. 重启 Worker 进程确保代码更新

### 2. 参数传递错误

**症状**: `badly formed hexadecimal UUID string` 或参数类型错误

**解决方法**:
- 检查任务函数的 `bind` 参数设置
- 确认调用任务时参数顺序正确
- 文件任务应使用 `bind=False`

### 3. 事件循环冲突

**症状**: `RuntimeError: got Future attached to a different loop`

**解决方法**:
- 已在 `async_task` 装饰器中修复
- 如果仍有问题，检查数据库连接配置
- 确保异步数据库引擎配置正确

### 4. Worker 无法连接

**症状**: Worker 无法启动或连接超时

**检查步骤**:
1. 确认 Redis 在正确端口运行 (6380)
2. 检查防火墙设置
3. 验证网络连接
4. 检查 Redis 配置

### 5. 任务重试问题

**症状**: 任务不断重试但从不成功

**排查方法**:
1. 检查任务函数内部逻辑
2. 查看详细错误日志
3. 验证数据库连接
4. 检查依赖服务状态

## 监控和日志

### 查看任务状态

```python
from app.tasks.celery import celery_app

# 检查已注册任务
print(list(celery_app.tasks.keys()))

# 检查任务状态
result = celery_app.AsyncResult(task_id)
print(result.state, result.result)
```

### 日志配置

Worker 日志包含：
- 任务启动/完成信息
- 错误和异常详情
- 重试和失败记录
- 性能统计

## 生产环境建议

1. **Worker 进程数**: 根据 CPU 核数调整
2. **内存监控**: 设置内存限制防止内存泄漏
3. **任务超时**: 为长时间运行的任务设置合理超时
4. **错误处理**: 实现完善的错误处理和报警机制
5. **监控**: 使用 Flower 或其他工具监控 Celery 状态

## 常用命令

```bash
# 启动 Worker
python run_celery_worker.py

# 检查 Worker 状态
celery -A app.tasks.celery inspect active

# 停止所有任务
celery -A app.tasks.celery control purge

# 查看任务统计
celery -A app.tasks.celery inspect stats
```

## 开发指南

### 创建新任务

```python
from app.tasks.base import async_task

@async_task(
    name="tasks.module.task_name",
    queue="custom_queue",
    max_retries=3,
    bind=False,  # 文件任务使用 False，其他通常使用 True
)
async def my_task(param1, param2):
    # 任务逻辑
    return result
```

### 调用任务

```python
# 异步调用
result = my_task.delay(param1, param2)

# 获取结果
if result.ready():
    print(result.result)
```

## 更新历史

- **2025-06-19**: 修复任务参数绑定问题和事件循环冲突
- **2025-06-19**: 统一任务命名约定为 `tasks.*` 模式
- **2025-06-19**: 添加详细的故障排除指南 