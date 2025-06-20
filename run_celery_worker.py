#!/usr/bin/env python3
"""
Celery Worker启动脚本

用于启动Celery worker进程来处理异步任务
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault("PYTHONPATH", str(project_root))

if __name__ == "__main__":
    # 导入Celery应用 - 这会触发任务注册
    from app.tasks.celery import celery_app
    
    print("=== Celery Worker启动 ===")
    print(f"已注册任务数量: {len(celery_app.tasks)}")
    print(f"项目路径: {project_root}")
    print(f"Python路径: {sys.path[0]}")
    
    # 显示所有注册的任务
    print("\n已注册的任务:")
    for task_name in sorted(celery_app.tasks.keys()):
        if not task_name.startswith('celery.'):
            print(f"  - {task_name}")
    
    print("\n关键任务检查:")
    
    # 检查关键任务是否存在
    key_tasks = [
        "tasks.file.process_file",
        "tasks.file.analyze_file", 
        "test_celery_connection"
    ]
    
    for task_name in key_tasks:
        if task_name in celery_app.tasks:
            task_obj = celery_app.tasks[task_name]
            print(f"  ✓ {task_name} - {task_obj}")
        else:
            print(f"  ✗ {task_name}")
    
    print(f"\nBroker: {celery_app.conf.broker_url}")
    print(f"Backend: {celery_app.conf.result_backend}")
    
    print("\n启动Worker...")
    
    # 启动worker，监听所有队列
    try:
        celery_app.worker_main([
            'worker',
            '--loglevel=info',
            '--queues=default,file_tasks,email_tasks,priority,inventory,credits,export,api_calls,scheduled',
            '--concurrency=1',  # 降低并发数便于调试
            '--pool=solo',      # 使用单线程模式便于调试
            '--without-heartbeat',  # 在开发环境中禁用心跳
            '--without-mingle',     # 在开发环境中禁用mingle
            '--without-gossip',     # 禁用gossip
        ])
    except KeyboardInterrupt:
        print("\nWorker已停止")
    except Exception as e:
        print(f"\nWorker启动失败: {e}")
        import traceback
        traceback.print_exc() 