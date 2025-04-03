import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import convert_message_to_dict
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

from app.core.config import settings
from app.core.exceptions import LLMAPIException
from app.llm.core.base import BaseLLM, LLMResponse, Message, StreamingChunk


class OpenAILLM(BaseLLM):
    """OpenAI LLM实现，封装与OpenAI API的交互"""

    def __init__(self):
        """初始化OpenAI客户端"""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.tiktoken_cache = {}

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
            # 转换消息格式
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # 调用OpenAI API
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # 提取并返回结果
            content = response.choices[0].message.content or ""

            return LLMResponse(
                content=content,
                model=model_id,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        except Exception as e:
            raise LLMAPIException(detail=f"OpenAI API调用失败: {str(e)}")

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
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # 调用OpenAI流式API
            stream = await self.client.chat.completions.create(
                model=model_id,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            # 生成流式响应
            async for chunk in self._process_openai_stream(stream):
                yield chunk

        except Exception as e:
            yield StreamingChunk(
                content="",
                done=True,
                error=True,
                message=f"OpenAI API调用失败: {str(e)}",
            )

    async def _process_openai_stream(
        self, stream: AsyncStream[ChatCompletionChunk]
    ) -> AsyncGenerator[StreamingChunk, None]:
        """处理OpenAI的流式响应"""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""

            # 检查是否是最后一个chunk
            finish_reason = chunk.choices[0].finish_reason
            is_done = finish_reason is not None

            yield StreamingChunk(content=content, done=is_done)

    async def count_tokens(self, messages: List[Message], model_id: str) -> int:
        """计算token数量"""
        try:
            # 使用LangChain的ChatOpenAI来计算tokens
            chat = ChatOpenAI(
                model=model_id, openai_api_key=settings.OPENAI_API_KEY, temperature=0
            )

            # 转换为LangChain格式的消息
            lc_messages = []
            for msg in messages:
                message_dict = {"role": msg.role, "content": msg.content}
                lc_message = convert_message_to_dict(message_dict)
                lc_messages.append(lc_message)

            # 计算tokens
            tokens = chat.get_num_tokens_from_messages(lc_messages)
            return tokens
        except Exception as e:
            # 如果计算失败，返回估计值
            return self._estimate_tokens([msg.content for msg in messages])

    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        估算文本的token数量

        这是一个粗略的估计，基于经验规则：每75个字符约为1个token
        注意：这只是一个应急方案，准确的计算应该使用tiktoken库
        """
        total_chars = sum(len(text) for text in texts)
        # 每个消息有额外开销，加上角色标记和格式标记
        message_overhead = 4 * len(texts)
        # 估算tokens：大约每75个字符是1个token，加上固定开销和消息开销
        estimated_tokens = (total_chars / 75) + message_overhead + 3
        return round(estimated_tokens)
