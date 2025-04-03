import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from app.core.config import settings
from app.core.exceptions import LLMAPIException
from app.llm.core.base import BaseLLM, LLMResponse, Message, StreamingChunk


class AnthropicLLM(BaseLLM):
    """Anthropic LLM实现，封装与Anthropic API的交互"""

    def __init__(self):
        """初始化Anthropic客户端"""
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def generate(
        self,
        messages: List[Message],
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """生成完整的响应"""
        try:
            # 转换消息格式为Anthropic格式
            anthropic_messages = self._convert_to_anthropic_messages(messages)

            # 调用Anthropic API
            response = await self.client.messages.create(
                model=model_id,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                **kwargs,
            )

            # 提取并返回结果
            content = response.content[0].text

            return LLMResponse(
                content=content,
                model=model_id,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                },
            )
        except Exception as e:
            raise LLMAPIException(detail=f"Anthropic API调用失败: {str(e)}")

    async def generate_stream(
        self,
        messages: List[Message],
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """流式生成响应"""
        try:
            # 转换消息格式
            anthropic_messages = self._convert_to_anthropic_messages(messages)

            # 调用Anthropic流式API
            with await self.client.messages.stream(
                model=model_id,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                stream=True,
                **kwargs,
            ) as stream:
                # 处理流式响应
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        delta = chunk.delta.text
                        yield StreamingChunk(content=delta, done=False)
                    elif chunk.type == "message_stop":
                        yield StreamingChunk(content="", done=True)

        except Exception as e:
            yield StreamingChunk(
                content="",
                done=True,
                error=True,
                message=f"Anthropic API调用失败: {str(e)}",
            )

    async def count_tokens(self, messages: List[Message], model_id: str) -> int:
        """计算token数量"""
        try:
            # 目前Anthropic提供了一个简单的估算方法
            # https://docs.anthropic.com/claude/docs/how-to-count-tokens-in-claude
            anthropic_messages = self._convert_to_anthropic_messages(messages)

            # 将消息转换为JSON字符串
            messages_json = json.dumps({"messages": anthropic_messages})

            # 使用Anthropic提供的方法计算tokens
            tokens = anthropic.count_tokens(messages_json)
            return tokens
        except Exception as e:
            # 如果计算失败，返回估计值
            return self._estimate_tokens([msg.content for msg in messages])

    def _convert_to_anthropic_messages(
        self, messages: List[Message]
    ) -> List[MessageParam]:
        """将通用消息格式转换为Anthropic格式"""
        anthropic_messages = []

        for msg in messages:
            # 映射角色
            role = self._map_role(msg.role)
            # 创建Anthropic消息
            anthropic_message = {"role": role, "content": msg.content}
            anthropic_messages.append(anthropic_message)

        return anthropic_messages

    def _map_role(self, role: str) -> str:
        """将通用角色映射到Anthropic角色"""
        role_mapping = {
            "user": "user",
            "assistant": "assistant",
            "system": "user",  # Anthropic不支持系统角色，需要特殊处理
        }
        return role_mapping.get(role, "user")

    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        估算文本的token数量

        这是一个粗略的估计，每4个字符约为1个token
        注意：Claude的token计算与OpenAI不同
        """
        total_chars = sum(len(text) for text in texts)
        # 每个消息有额外开销
        message_overhead = 5 * len(texts)
        # 估算tokens：大约每4个字符是1个token，加上消息开销
        estimated_tokens = (total_chars / 4) + message_overhead
        return round(estimated_tokens)
