"""
Pytest配置文件，提供全局fixture和配置
"""

import asyncio
from typing import Dict

import pytest
import pytest_asyncio


def pytest_configure(config):
    """配置pytest"""
    # 注册自定义标记
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )
    config.addinivalue_line("markers", "unit: mark a test as a unit test")


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> Dict:
    """测试配置"""
    return {
        "test_user_email": "test@example.com",
        "test_user_password": "password123",
        "test_user_username": "testuser",
        "test_admin_email": "admin@example.com",
        "test_admin_password": "admin123",
        "test_admin_username": "adminuser",
    }
