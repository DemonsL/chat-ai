#!/usr/bin/env python3
"""测试配置脚本"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv("dev.env")

from app.core.config import settings

print("=== 数据库配置检查 ===")
print(f"POSTGRES_SERVER: {settings.POSTGRES_SERVER}")
print(f"POSTGRES_PORT: {settings.POSTGRES_PORT}")
print(f"POSTGRES_USER: {settings.POSTGRES_USER}")
print(f"POSTGRES_DB: {settings.POSTGRES_DB}")
print(f"完整数据库URL: {settings.SQLALCHEMY_DATABASE_URI}")

print("\n=== 环境变量检查 ===")
print(f"ENV POSTGRES_PORT: {os.getenv('POSTGRES_PORT')}")
print(f"ENV POSTGRES_DB: {os.getenv('POSTGRES_DB')}")

print("\n=== 测试连接 ===")
try:
    import asyncpg
    import asyncio
    
    async def test_connection():
        try:
            conn = await asyncpg.connect(
                host=settings.POSTGRES_SERVER,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB
            )
            print("✓ 数据库连接成功!")
            await conn.close()
            return True
        except Exception as e:
            print(f"✗ 数据库连接失败: {e}")
            return False
    
    # 运行测试
    result = asyncio.run(test_connection())
    
except ImportError:
    print("⚠ asyncpg未安装，跳过连接测试") 