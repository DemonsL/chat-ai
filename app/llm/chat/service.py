import json
from typing import AsyncGenerator, Dict, List, Optional
from uuid import UUID

from app.core.exceptions import NotFoundException
from app.db.models.message import Message as DBMessage
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.llm.core.base import LLMFactory, Message, StreamingChunk


class ChatService:
    """
    聊天服务，处理常规聊天功能
    """

    def __init__(
        self,
        message_repo: MessageRepository,
        conversation_repo: ConversationRepository,
        model_repo: ModelConfigRepository,
    ):
        self.message_repo = message_repo
        self.conversation_repo = conversation_repo
        self.model_repo = model_repo

    async def process_message(
        self, conversation_id: UUID, content: str, metadata: Optional[Dict] = None
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        处理用户消息并返回AI响应

        参数:
            conversation_id: 会话ID
            content: 用户消息内容
            metadata: 可选的元数据

        返回:
            AI响应的流式生成器
        """
        # 获取会话信息
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")

        # 获取模型配置
        model_config = await self.model_repo.get_by_model_id(conversation.model_id)
        if not model_config or not model_config.is_active:
            raise NotFoundException(detail="所选模型不可用")

        # 获取LLM实例
        llm = LLMFactory.create(model_config.provider)

        # 获取历史消息
        history = await self.message_repo.get_conversation_history(conversation_id)

        # 构建消息列表
        messages = []

        # 添加系统提示（如果有）
        if conversation.system_prompt:
            messages.append(Message(role="system", content=conversation.system_prompt))

        # 添加历史消息
        for msg in history:
            messages.append(Message(role=msg.role, content=msg.content))

        # 添加当前用户消息
        messages.append(Message(role="user", content=content))

        # 生成响应
        model_params = {}
        if model_config.config:
            # 添加模型特定配置
            model_params = model_config.config

        # 获取流式响应
        response_stream = llm.generate_stream(
            messages=messages,
            model_id=model_config.model_id,
            max_tokens=model_config.max_tokens,
            **model_params
        )

        # 返回响应
        async for chunk in response_stream:
            # 转换为JSON字符串以便于前端解析
            chunk_dict = {"content": chunk.content, "done": chunk.done}

            # 如果出错，添加错误信息
            if chunk.error:
                chunk_dict["error"] = True
                chunk_dict["message"] = chunk.message

            yield json.dumps(chunk_dict)

    async def get_estimated_tokens(
        self, conversation_id: UUID, new_content: str
    ) -> Dict[str, int]:
        """
        估算会话的token使用情况

        参数:
            conversation_id: 会话ID
            new_content: 新的用户消息内容

        返回:
            token使用情况的字典
        """
        # 获取会话信息
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")

        # 获取模型配置
        model_config = await self.model_repo.get_by_model_id(conversation.model_id)
        if not model_config:
            raise NotFoundException(detail="所选模型不可用")

        # 获取LLM实例
        llm = LLMFactory.create(model_config.provider)

        # 获取历史消息
        history = await self.message_repo.get_conversation_history(conversation_id)

        # 构建消息列表
        messages = []

        # 添加系统提示（如果有）
        if conversation.system_prompt:
            messages.append(Message(role="system", content=conversation.system_prompt))

        # 添加历史消息
        for msg in history:
            messages.append(Message(role=msg.role, content=msg.content))

        # 添加当前用户消息
        messages.append(Message(role="user", content=new_content))

        # 计算tokens
        tokens = await llm.count_tokens(messages, model_config.model_id)

        # 返回token使用情况
        return {
            "prompt_tokens": tokens,
            "max_tokens": model_config.max_tokens,
            "available_tokens": max(0, model_config.max_tokens - tokens),
        }
