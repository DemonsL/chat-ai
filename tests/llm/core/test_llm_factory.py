"""
测试LLM工厂类
"""

from unittest.mock import patch

import pytest

from app.llm.core.anthropic_llm import AnthropicLLM
from app.llm.core.base import BaseLLM, LLMFactory
from app.llm.core.openai_llm import OpenAILLM


class TestLLMFactory:
    """测试LLM工厂类"""

    def test_create_openai_llm(self):
        """测试创建OpenAI LLM实例"""
        with patch.object(OpenAILLM, "__init__", return_value=None):
            llm = LLMFactory.create("openai")
            assert isinstance(llm, OpenAILLM)

    def test_create_anthropic_llm(self):
        """测试创建Anthropic LLM实例"""
        with patch.object(AnthropicLLM, "__init__", return_value=None):
            llm = LLMFactory.create("anthropic")
            assert isinstance(llm, AnthropicLLM)

    def test_create_invalid_provider(self):
        """测试创建无效提供商的LLM实例"""
        with pytest.raises(ValueError):
            LLMFactory.create("invalid_provider")
