"""
测试基类，提供测试所需的基础设施
"""

import os
from typing import AsyncGenerator, Dict, Generator, List

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app

# 创建测试数据库引擎
TEST_DATABASE_URL = settings.DATABASE_URL.replace("/chatai", "/test_chatai")
test_engine = create_async_engine(str(TEST_DATABASE_URL), echo=False)
TestingSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


# 数据库会话依赖覆盖
@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    # 创建测试数据库表
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 使用事务进行测试，测试完成后回滚
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()

    # 清理测试数据库表
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# 覆盖app的依赖
@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    # 替换应用程序的数据库会话依赖
    async def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # 创建测试客户端
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    # 恢复原始依赖
    app.dependency_overrides = {}


# 模拟数据
class TestData:
    """测试数据工具类"""

    @staticmethod
    def sample_user() -> Dict:
        """生成样例用户数据"""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
            "full_name": "Test User",
        }

    @staticmethod
    def sample_conversation() -> Dict:
        """生成样例会话数据"""
        return {
            "title": "测试会话",
            "model_id": "gpt-4o",
            "mode": "chat",
            "system_prompt": "你是一个测试助手",
        }

    @staticmethod
    def sample_message() -> Dict:
        """生成样例消息数据"""
        return {"content": "这是一条测试消息", "role": "user"}

    @staticmethod
    def sample_model_config() -> Dict:
        """生成样例模型配置数据"""
        return {
            "model_id": "test-model",
            "display_name": "测试模型",
            "provider": "test",
            "capabilities": ["chat"],
            "max_tokens": 2048,
            "is_active": True,
            "config": {"temperature": 0.7},
            "api_key_env_name": "TEST_API_KEY",
        }
