#!/usr/bin/env python3
"""
本地API服务启动脚本
用于本地开发调试，连接到Docker中的其他服务
"""

import os
import uvicorn

def setup_local_env():
    """设置本地开发环境变量"""
    # Redis配置 - 连接到Docker中的Redis
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6380")
    
    # 数据库配置 - 连接到Docker中的PostgreSQL
    os.environ.setdefault("POSTGRES_SERVER", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5433")
    os.environ.setdefault("POSTGRES_USER", "postgres")
    os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
    os.environ.setdefault("POSTGRES_DB", "chatai")
    
    # Celery配置
    os.environ.setdefault("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
    os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6380/0")
    
    # MinIO配置
    os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
    os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
    
    # 其他配置
    os.environ.setdefault("DEBUG", "True")
    os.environ.setdefault("SECRET_KEY", "dev-secret-key-change-in-production")
    
    print("🔧 本地开发环境变量设置完成")
    print(f"📡 Redis: {os.environ.get('REDIS_HOST')}:{os.environ.get('REDIS_PORT')}")
    print(f"🗄️  PostgreSQL: {os.environ.get('POSTGRES_SERVER')}:{os.environ.get('POSTGRES_PORT')}")
    print(f"🐰 RabbitMQ: localhost:5672")
    print(f"📦 MinIO: {os.environ.get('MINIO_ENDPOINT')}")

if __name__ == "__main__":
    setup_local_env()
    
    print("\n🚀 启动本地API服务...")
    print("📄 API文档: http://localhost:8000/docs")
    print("⚡ 重新加载模式已启用")
    print("\n按 Ctrl+C 停止服务\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        log_level="info"
    ) 