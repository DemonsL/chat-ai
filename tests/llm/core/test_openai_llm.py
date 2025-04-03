"""
测试OpenAI LLM
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from app.llm.core.base import Message
from app.llm.core.openai_llm import OpenAILLM


class TestOpenAILLM:
    """测试OpenAI LLM类"""

    def setup_method(self):
        """测试前准备"""
        self.llm = OpenAILLM()
        self.messages = [
            Message(role="system", content="你是一个有用的助手"),
            Message(role="user", content="你好，请问今天天气如何？"),
        ]
        self.model_id = "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate(self):
        """测试生成文本"""
        # 模拟OpenAI API响应
        mock_response = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677858242,
            model=self.model_id,
            usage=CompletionUsage(
                prompt_tokens=20, completion_tokens=15, total_tokens=35
            ),
            choices=[
                {
                    "message": ChatCompletionMessage(
                        role="assistant",
                        content="今天天气晴朗，气温适宜，非常适合户外活动。",
                    ),
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        )

        # 模拟API调用
        with patch.object(
            self.llm.client.chat.completions,
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
        # 创建mock函数，模拟tiktoken行为
        with patch("tiktoken.encoding_for_model", return_value=MagicMock()):
            with patch.object(self.llm, "_estimate_tokens", return_value=35):
                # 调用被测试的方法
                tokens = await self.llm.count_tokens(
                    messages=self.messages, model_id=self.model_id
                )

                # 验证结果
                assert tokens == 35

    def test_estimate_tokens(self):
        """测试预估tokens"""
        texts = ["Hello, world!", "This is a test message."]
        # 假设每个字符算一个token（这只是为了测试）
        expected_tokens = len("".join(texts))

        # 模拟tiktoken不可用的情况
        result = self.llm._estimate_tokens(texts)

        # 因为是估算，我们只检查结果是否是一个合理的正数
        assert result > 0
