"""
测试LangChain LLM
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import LLMResult, Generation

from app.llm.core.base import Message
from app.llm.core.langchain_llm import LangChainLLM, StreamingCallbackHandler


class TestLangChainLLM:
    """测试LangChain LLM类"""

    def setup_method(self):
        """测试前准备"""
        self.provider = "openai"
        self.api_key = "test_api_key"
        
        # 模拟LangChain模型
        self.mock_model = MagicMock()
        
        # 创建带模拟模型的LLM实例
        with patch.object(LangChainLLM, '_create_langchain_model', return_value=self.mock_model):
            self.llm = LangChainLLM(provider=self.provider, api_key=self.api_key)
        
        self.messages = [
            Message(role="system", content="你是一个有用的助手"),
            Message(role="user", content="你好，请问今天天气如何？"),
        ]
        self.model_id = "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate(self):
        """测试生成文本"""
        # 模拟LangChain响应
        generation = Generation(text="今天天气晴朗，气温适宜，非常适合户外活动。")
        mock_response = LLMResult(
            generations=[[generation]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 15,
                    "total_tokens": 35
                }
            }
        )
        
        # 设置模拟对象
        self.mock_model.agenerate = AsyncMock(return_value=mock_response)
        
        # 测试生成方法
        response = await self.llm.generate(
            messages=self.messages,
            model_id=self.model_id,
            temperature=0.7
        )
        
        # 验证结果
        assert response.content == "今天天气晴朗，气温适宜，非常适合户外活动。"
        assert response.model == self.model_id
        assert response.usage == {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35
        }
        
        # 验证模型参数设置
        assert self.mock_model.model_name == self.model_id
        assert self.mock_model.temperature == 0.7

    @pytest.mark.asyncio
    async def test_count_tokens(self):
        """测试token计数"""
        # 设置模拟对象
        self.mock_model.get_num_tokens_from_messages = MagicMock(return_value=30)
        
        # 测试token计数方法
        tokens = await self.llm.count_tokens(self.messages, self.model_id)
        
        # 验证结果
        assert tokens == 30
        
        # 验证模型参数设置
        assert self.mock_model.model_name == self.model_id

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        """测试token估算"""
        # 设置模拟对象抛出异常，触发估算逻辑
        self.mock_model.get_num_tokens_from_messages = MagicMock(side_effect=Exception("测试异常"))
        
        # 测试估算
        tokens = await self.llm.count_tokens(self.messages, self.model_id)
        
        # 验证使用了估算
        assert isinstance(tokens, int)
        assert tokens > 0 