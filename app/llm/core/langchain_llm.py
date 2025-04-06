import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from app.core.config import settings
from app.core.exceptions import LLMAPIException
from app.llm.core.base import BaseLLM, LLMResponse, Message, StreamingChunk


class StreamingCallbackHandler(AsyncCallbackHandler):
    """处理LangChain流式输出的回调处理器"""

    def __init__(self):
        super().__init__()
        self.chunks = []
        self.error = None
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):
        """收到新token时的回调"""
        self.chunks.append(token)
        await self.queue.put(token)

    async def on_llm_error(self, error: Exception, **kwargs):
        """发生错误时的回调"""
        self.error = error
        await self.queue.put(None)  # 出错时发送None作为信号


class LangChainLLM(BaseLLM):
    """基于LangChain的LLM实现，作为统一适配器"""

    def __init__(self, provider: str, api_key: str, **kwargs):
        """
        初始化LangChain LLM

        参数:
            provider: LLM提供商
            api_key: API密钥
            **kwargs: 其他配置参数
        """
        self.provider = provider
        self.api_key = api_key
        self.model = self._create_langchain_model(**kwargs)

    def _create_langchain_model(self, **kwargs) -> BaseChatModel:
        """创建对应的LangChain模型实例"""
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(openai_api_key=self.api_key, **kwargs)
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(anthropic_api_key=self.api_key, **kwargs)
        elif self.provider == "deepseek":
            from langchain_community.chat_models import ChatDeepSeek

            return ChatDeepSeek(api_key=self.api_key, **kwargs)
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")

    def _convert_to_langchain_messages(self, messages: List[Message]):
        """将内部消息格式转换为LangChain消息格式"""
        lc_messages = []
        for msg in messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
        return lc_messages

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
            # 设置模型参数
            self.model.model_name = model_id
            self.model.temperature = temperature
            if max_tokens:
                self.model.max_tokens = max_tokens

            # 转换消息格式
            lc_messages = self._convert_to_langchain_messages(messages)

            # 执行生成
            response = await self.model.agenerate([lc_messages])
            generation = response.generations[0][0]

            # 处理token使用情况
            token_usage = {}
            if response.llm_output and "token_usage" in response.llm_output:
                token_usage = response.llm_output["token_usage"]
            else:
                # 尝试获取不同提供商的token使用格式
                if self.provider == "openai":
                    token_usage = {
                        "prompt_tokens": response.llm_output.get("prompt_tokens", 0),
                        "completion_tokens": response.llm_output.get(
                            "completion_tokens", 0
                        ),
                        "total_tokens": response.llm_output.get("total_tokens", 0),
                    }
                elif self.provider == "anthropic":
                    token_usage = {
                        "input_tokens": response.llm_output.get("input_tokens", 0),
                        "output_tokens": response.llm_output.get("output_tokens", 0),
                        "total_tokens": response.llm_output.get("input_tokens", 0)
                        + response.llm_output.get("output_tokens", 0),
                    }

            # 构建统一响应
            return LLMResponse(
                content=generation.text,
                model=model_id,
                usage=token_usage,
            )
        except Exception as e:
            raise LLMAPIException(detail=f"{self.provider} API调用失败: {str(e)}")

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
            # 设置模型参数
            self.model.model_name = model_id
            self.model.temperature = temperature
            self.model.streaming = True
            if max_tokens:
                self.model.max_tokens = max_tokens

            # 转换消息格式
            lc_messages = self._convert_to_langchain_messages(messages)

            # 创建回调处理器
            handler = StreamingCallbackHandler()

            # 创建生成任务
            task = asyncio.create_task(
                self.model.agenerate([lc_messages], callbacks=[handler])
            )

            # 从队列中获取流式输出
            while not task.done() or not handler.queue.empty():
                try:
                    # 尝试获取新token，设置超时以避免一直阻塞
                    token = await asyncio.wait_for(handler.queue.get(), timeout=0.1)

                    # 如果收到None，表示出错
                    if token is None:
                        if handler.error:
                            yield StreamingChunk(
                                content="",
                                done=True,
                                error=True,
                                message=f"{self.provider} API调用失败: {str(handler.error)}",
                            )
                        return

                    # 发送新token
                    yield StreamingChunk(content=token, done=False)

                except asyncio.TimeoutError:
                    # 队列超时，检查任务是否完成
                    if task.done():
                        break

            # 任务完成，发送最后一个chunk
            yield StreamingChunk(content="", done=True)

        except Exception as e:
            yield StreamingChunk(
                content="",
                done=True,
                error=True,
                message=f"{self.provider} API调用失败: {str(e)}",
            )

    async def count_tokens(self, messages: List[Message], model_id: str) -> int:
        """计算token数量"""
        try:
            # 设置模型ID
            self.model.model_name = model_id

            # 转换消息格式
            lc_messages = self._convert_to_langchain_messages(messages)

            # 使用LangChain的token计数功能
            return self.model.get_num_tokens_from_messages(lc_messages)
        except Exception as e:
            # 如果计算失败，返回估计值
            return self._estimate_tokens([msg.content for msg in messages])

    def _estimate_tokens(self, texts: List[str]) -> int:
        """
        估算文本的token数量

        这是一个粗略的估计，基于经验规则：每75个字符约为1个token
        注意：这只是一个应急方案，准确的计算应该使用专用的tokenizer
        """
        total_chars = sum(len(text) for text in texts)
        # 每个消息有额外开销，加上角色标记和格式标记
        message_overhead = 4 * len(texts)
        # 估算tokens：大约每75个字符是1个token，加上固定开销和消息开销
        estimated_tokens = (total_chars / 75) + message_overhead + 3
        return round(estimated_tokens)
