"""
Celery异步任务模块

此包包含所有Celery异步任务的定义和实现。
"""

# 导入所有任务模块，确保任务被注册到Celery应用中
try:
    from . import file
    from . import email
    from . import api
    from . import scripts
    from . import stats
    from . import inventory
    from . import credits
    
    # 导出主要任务函数
    from .file import process_file_task, analyze_file_task, export_file_task, bulk_upload_task
    from .email import send_email_task, send_notification_task
    
    __all__ = [
        'process_file_task',
        'analyze_file_task', 
        'export_file_task',
        'bulk_upload_task',
        'send_email_task',
        'send_notification_task',
    ]
    
except ImportError as e:
    # 如果某些模块导入失败，记录但不中断
    print(f"部分任务模块导入失败: {e}")
    __all__ = [] 