from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """LLM响应的基本结构"""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # token 使用情况


class StreamingChunk(BaseModel):
    """流式响应的单个数据块"""

    content: str
    done: bool = False
    error: bool = False
    message: Optional[str] = None


class Message(BaseModel):
    """消息的基本结构"""

    role: str
    content: str


class BaseLLM(ABC):
    """LLM基类，定义与大语言模型交互的基础接口"""

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        生成完整的响应

        参数:
            messages: 对话消息历史
            model_id: 模型ID
            temperature: 生成多样性参数
            max_tokens: 最大生成的token数
            **kwargs: 其他模型特定参数

        返回:
            LLMResponse对象
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        model_id: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        流式生成响应

        参数:
            messages: 对话消息历史
            model_id: 模型ID
            temperature: 生成多样性参数
            max_tokens: 最大生成的token数
            **kwargs: 其他模型特定参数

        返回:
            StreamingChunk对象的异步生成器
        """
        pass

    @abstractmethod
    async def count_tokens(self, messages: List[Message], model_id: str) -> int:
        """
        计算消息列表的token数量

        参数:
            messages: 要计算的消息列表
            model_id: 目标模型ID

        返回:
            token数量
        """
        pass


class LLMFactory:
    """LLM工厂类，根据提供商类型创建对应的LLM实例"""

    @staticmethod
    def create(provider: str) -> BaseLLM:
        """
        创建LLM实例

        参数:
            provider: 提供商类型，例如 "openai", "anthropic"

        返回:
            对应提供商的LLM实例
        """
        if provider == "openai":
            from app.llm.core.openai_llm import OpenAILLM

            return OpenAILLM()
        elif provider == "anthropic":
            from app.llm.core.anthropic_llm import AnthropicLLM

            return AnthropicLLM()
        elif provider == "deepseek":
            from app.llm.core.deepseek_llm import DeepseekLLM

            return DeepseekLLM()
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
