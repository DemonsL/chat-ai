"""
测试聊天服务
"""

import json
import uuid
from typing import AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.models.conversation import Conversation
from app.db.models.message import Message as DBMessage
from app.db.models.user import User
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.llm.chat.service import ChatService
from app.llm.core.base import LLMFactory, Message, StreamingChunk
from app.schemas.message import MessageRole


class TestChatService:
    """测试聊天服务"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟的仓库对象
        self.message_repo = MagicMock(spec=MessageRepository)
        self.conversation_repo = MagicMock(spec=ConversationRepository)
        self.model_repo = MagicMock(spec=ModelConfigRepository)

        # 创建测试数据
        self.user_id = uuid.uuid4()
        self.conversation_id = uuid.uuid4()
        self.model_id = "gpt-4o"

        # 创建聊天服务实例
        self.chat_service = ChatService(
            message_repo=self.message_repo,
            conversation_repo=self.conversation_repo,
            model_repo=self.model_repo,
        )

        # 设置模拟会话
        self.conversation = MagicMock(spec=Conversation)
        self.conversation.id = self.conversation_id
        self.conversation.user_id = self.user_id
        self.conversation.model_id = self.model_id
        self.conversation.system_prompt = "你是一个AI助手"

        # 设置模拟消息
        self.db_message = MagicMock(spec=DBMessage)
        self.db_message.id = uuid.uuid4()
        self.db_message.role = "user"
        self.db_message.content = "你好，请问今天天气如何？"

    @pytest.mark.asyncio
    async def test_process_message_success(self):
        """测试消息处理成功场景"""
        # 设置获取会话的模拟
        self.conversation_repo.get_by_id.return_value = self.conversation

        # 设置获取消息历史的模拟
        system_message = MagicMock(spec=DBMessage)
        system_message.role = "system"
        system_message.content = "你是一个AI助手"

        user_message = MagicMock(spec=DBMessage)
        user_message.role = "user"
        user_message.content = "你好"

        assistant_message = MagicMock(spec=DBMessage)
        assistant_message.role = "assistant"
        assistant_message.content = "你好！有什么可以帮到你的？"

        # 设置历史消息
        self.message_repo.get_conversation_history.return_value = [
            system_message,
            user_message,
            assistant_message,
        ]

        # 设置创建消息的模拟
        self.message_repo.create.return_value = self.db_message

        # 模拟LLM及其响应
        mock_llm = AsyncMock()
        mock_stream_chunks = [
            StreamingChunk(content="今天", done=False),
            StreamingChunk(content="天气", done=False),
            StreamingChunk(content="晴朗", done=True),
        ]

        # 设置模拟流式生成
        async def mock_stream():
            for chunk in mock_stream_chunks:
                yield chunk

        mock_llm.generate_stream.return_value = mock_stream()

        # 模拟LLMFactory
        with patch.object(LLMFactory, "create", return_value=mock_llm):
            # 执行测试
            response_chunks = []
            async for chunk in self.chat_service.process_message(
                conversation_id=self.conversation_id, content="请问今天天气如何？"
            ):
                response_chunks.append(chunk)

            # 验证结果
            assert len(response_chunks) == 3
            assert response_chunks[0].content == "今天"
            assert response_chunks[1].content == "天气"
            assert response_chunks[2].content == "晴朗"
            assert response_chunks[2].done is True

            # 验证调用
            self.conversation_repo.get_by_id.assert_called_once_with(
                self.conversation_id
            )
            self.message_repo.get_conversation_history.assert_called_once()
            self.message_repo.create.assert_called()

    @pytest.mark.asyncio
    async def test_get_estimated_tokens(self):
        """测试预估tokens"""
        # 设置获取会话的模拟
        self.conversation_repo.get_by_id.return_value = self.conversation

        # 设置获取消息历史的模拟
        system_message = MagicMock(spec=DBMessage)
        system_message.role = "system"
        system_message.content = "你是一个AI助手"

        user_message = MagicMock(spec=DBMessage)
        user_message.role = "user"
        user_message.content = "你好"

        # 设置历史消息
        self.message_repo.get_conversation_history.return_value = [
            system_message,
            user_message,
        ]

        # 模拟LLM及其响应
        mock_llm = AsyncMock()
        mock_llm.count_tokens.return_value = 30

        # 模拟LLMFactory
        with patch.object(LLMFactory, "create", return_value=mock_llm):
            # 执行测试
            result = await self.chat_service.get_estimated_tokens(
                conversation_id=self.conversation_id, new_content="请问今天天气如何？"
            )

            # 验证结果
            assert result["total"] == 30
            assert "prompt_tokens" in result

            # 验证调用
            self.conversation_repo.get_by_id.assert_called_once_with(
                self.conversation_id
            )
            self.message_repo.get_conversation_history.assert_called_once()
