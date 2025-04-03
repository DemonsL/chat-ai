"""
测试RAG服务
"""

import json
import uuid
from typing import AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

from app.db.models.conversation import Conversation
from app.db.models.message import Message as DBMessage
from app.db.models.user_file import UserFile
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.repositories.user_file_repository import UserFileRepository
from app.llm.core.base import LLMFactory, Message, StreamingChunk
from app.llm.rag.service import RAGService
from app.schemas.message import MessageRole


class TestRAGService:
    """测试RAG服务"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟的仓库对象
        self.message_repo = MagicMock(spec=MessageRepository)
        self.conversation_repo = MagicMock(spec=ConversationRepository)
        self.model_repo = MagicMock(spec=ModelConfigRepository)
        self.file_repo = MagicMock(spec=UserFileRepository)

        # 创建测试数据
        self.user_id = uuid.uuid4()
        self.conversation_id = uuid.uuid4()
        self.model_id = "gpt-4o"
        self.file_id = uuid.uuid4()

        # 创建RAG服务实例
        self.rag_service = RAGService(
            message_repo=self.message_repo,
            conversation_repo=self.conversation_repo,
            model_repo=self.model_repo,
            file_repo=self.file_repo,
        )

        # 设置模拟会话
        self.conversation = MagicMock(spec=Conversation)
        self.conversation.id = self.conversation_id
        self.conversation.user_id = self.user_id
        self.conversation.model_id = self.model_id
        self.conversation.system_prompt = "你是一个AI助手"

        # 设置模拟文件
        self.file = MagicMock(spec=UserFile)
        self.file.id = self.file_id
        self.file.user_id = self.user_id
        self.file.filename = "test.pdf"
        self.file.original_filename = "test.pdf"
        self.file.file_type = "pdf"
        self.file.status = "indexed"
        self.file.metadata = {"page_count": 5}

        # 设置模拟消息
        self.db_message = MagicMock(spec=DBMessage)
        self.db_message.id = uuid.uuid4()
        self.db_message.role = "user"
        self.db_message.content = "文档中包含哪些重要信息？"

    @pytest.mark.asyncio
    async def test_process_message_success(self):
        """测试消息处理成功场景"""
        # 设置获取会话的模拟
        self.conversation_repo.get_by_id.return_value = self.conversation
        self.conversation.files = [self.file]

        # 设置获取消息历史的模拟
        system_message = MagicMock(spec=DBMessage)
        system_message.role = "system"
        system_message.content = "你是一个AI助手"

        user_message = MagicMock(spec=DBMessage)
        user_message.role = "user"
        user_message.content = "这个文档是关于什么的？"

        assistant_message = MagicMock(spec=DBMessage)
        assistant_message.role = "assistant"
        assistant_message.content = "这个文档是关于人工智能的介绍。"

        # 设置历史消息
        self.message_repo.get_conversation_history.return_value = [
            system_message,
            user_message,
            assistant_message,
        ]

        # 设置创建消息的模拟
        self.message_repo.create.return_value = self.db_message

        # 模拟文档
        mock_docs = [
            Document(
                page_content="人工智能是计算机科学的一个分支，致力于开发能够执行需要人类智能的任务的计算机系统。",
                metadata={"source": "test.pdf", "page": 1},
            ),
            Document(
                page_content="机器学习是人工智能的一个子领域，涉及开发能够从数据中学习的算法。",
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]

        # 模拟Chroma和检索
        mock_chroma = MagicMock(spec=Chroma)
        mock_chroma.similarity_search.return_value = mock_docs

        # 模拟LLM及其响应
        mock_llm = AsyncMock()
        mock_stream_chunks = [
            StreamingChunk(content="文档包含", done=False),
            StreamingChunk(content="关于人工智能", done=False),
            StreamingChunk(content="和机器学习的信息", done=True),
        ]

        # 设置模拟流式生成
        async def mock_stream():
            for chunk in mock_stream_chunks:
                yield chunk

        mock_llm.generate_stream.return_value = mock_stream()

        # 模拟依赖
        with patch("app.llm.rag.service.Chroma", return_value=mock_chroma), patch(
            "app.llm.rag.service.OpenAIEmbeddings", return_value=MagicMock()
        ), patch.object(LLMFactory, "create", return_value=mock_llm):

            # 执行测试
            response_chunks = []
            async for chunk in self.rag_service.process_message(
                conversation_id=self.conversation_id, content="文档中包含哪些重要信息？"
            ):
                response_chunks.append(chunk)

            # 验证结果
            assert len(response_chunks) == 3
            assert response_chunks[0].content == "文档包含"
            assert response_chunks[1].content == "关于人工智能"
            assert response_chunks[2].content == "和机器学习的信息"
            assert response_chunks[2].done is True

            # 验证调用
            self.conversation_repo.get_by_id.assert_called_once_with(
                self.conversation_id
            )
            self.message_repo.get_conversation_history.assert_called_once()
            self.message_repo.create.assert_called()
            mock_chroma.similarity_search.assert_called()
