"""
测试Anthropic LLM
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageParam

from app.llm.core.anthropic_llm import AnthropicLLM
from app.llm.core.base import Message


class TestAnthropicLLM:
    """测试Anthropic LLM类"""

    def setup_method(self):
        """测试前准备"""
        self.llm = AnthropicLLM()
        self.messages = [
            Message(role="system", content="你是一个有用的助手"),
            Message(role="user", content="你好，请问今天天气如何？"),
        ]
        self.model_id = "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_generate(self):
        """测试生成文本"""
        # 模拟Anthropic API响应
        mock_response = MagicMock()
        mock_response.content = [
            AnthropicMessage(
                type="text", text="今天天气晴朗，气温适宜，非常适合户外活动。"
            )
        ]
        mock_response.model = self.model_id
        mock_response.usage = {"input_tokens": 20, "output_tokens": 15}

        # 模拟API调用
        with patch.object(
            self.llm.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await self.llm.generate(
                messages=self.messages, model_id=self.model_id, temperature=0.7
            )

            # 验证结果
            assert response.content == "今天天气晴朗，气温适宜，非常适合户外活动。"
            assert response.model == self.model_id
            assert response.usage == {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35,
            }

    @pytest.mark.asyncio
    async def test_count_tokens(self):
        """测试计算token数量"""
        # 创建mock函数
        with patch.object(self.llm, "_estimate_tokens", return_value=35):
            # 调用被测试的方法
            tokens = await self.llm.count_tokens(
                messages=self.messages, model_id=self.model_id
            )

            # 验证结果
            assert tokens == 35

    def test_convert_to_anthropic_messages(self):
        """测试消息转换"""
        # 准备测试数据
        messages = [
            Message(role="system", content="你是一个有用的助手"),
            Message(role="user", content="你好"),
            Message(role="assistant", content="有什么可以帮到你的？"),
            Message(role="user", content="请问今天天气如何？"),
        ]

        # 调用被测试的方法
        result = self.llm._convert_to_anthropic_messages(messages)

        # 验证结果
        assert len(result) == 4
        assert isinstance(result[0], MessageParam)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "你是一个有用的助手"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_map_role(self):
        """测试角色映射"""
        assert self.llm._map_role("user") == "user"
        assert self.llm._map_role("assistant") == "assistant"
        assert self.llm._map_role("system") == "system"
        # 测试未知角色
        with pytest.raises(ValueError):
            self.llm._map_role("unknown")

    def test_estimate_tokens(self):
        """测试预估tokens"""
        texts = ["Hello, world!", "This is a test message."]
        # 因为是估算，我们只测试返回值是否合理
        result = self.llm._estimate_tokens(texts)
        assert result > 0
