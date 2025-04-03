import json
from typing import AsyncGenerator, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.conversation import Conversation
from app.db.models.message import Message
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.repositories.user_file_repository import UserFileRepository
from app.llm.agent.service import AgentService
from app.llm.chat.service import ChatService
from app.llm.rag.service import RAGService
from app.schemas.message import MessageRole


class MessageOrchestrator:
    """
    消息协调器，负责处理不同类型的消息并路由到正确的处理服务
    """

    def __init__(self, db_session: AsyncSession):
        """初始化协调器"""
        self.db_session = db_session
        self.conversation_repo = ConversationRepository(db_session)
        self.message_repo = MessageRepository(db_session)
        self.model_repo = ModelConfigRepository(db_session)
        self.file_repo = UserFileRepository(db_session)

        # 创建聊天服务
        self.chat_service = ChatService(
            message_repo=self.message_repo,
            conversation_repo=self.conversation_repo,
            model_repo=self.model_repo,
        )

        # 创建RAG服务
        self.rag_service = RAGService(
            message_repo=self.message_repo,
            conversation_repo=self.conversation_repo,
            model_repo=self.model_repo,
            file_repo=self.file_repo,
        )

        # 创建Agent服务
        self.agent_service = AgentService(
            message_repo=self.message_repo,
            conversation_repo=self.conversation_repo,
            model_repo=self.model_repo,
        )

    async def handle_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        处理新消息

        参数:
            conversation_id: 会话ID
            user_id: 用户ID
            content: 消息内容
            metadata: 可选的元数据

        返回:
            流式响应生成器
        """
        # 检查权限
        conversation = await self.conversation_repo.get_by_id_for_user(
            id=conversation_id, user_id=user_id
        )
        if not conversation:
            raise PermissionDeniedException(detail="没有权限访问此会话或会话不存在")

        # 存储用户消息
        await self._store_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            metadata=metadata,
        )

        try:
            # 根据会话模式选择处理服务
            if conversation.mode == "chat":
                # 使用聊天服务
                service_stream = self.chat_service.process_message(
                    conversation_id=conversation_id, content=content, metadata=metadata
                )
            elif conversation.mode == "rag":
                # 使用RAG服务
                service_stream = self.rag_service.process_message(
                    conversation_id=conversation_id, content=content, metadata=metadata
                )
            elif conversation.mode == "deepresearch":
                # 使用Agent服务
                service_stream = self.agent_service.process_message(
                    conversation_id=conversation_id, content=content, metadata=metadata
                )
            else:
                # 默认使用聊天服务
                service_stream = self.chat_service.process_message(
                    conversation_id=conversation_id, content=content, metadata=metadata
                )

            # 用于收集完整的助手回复
            full_response = ""
            full_metadata = {}

            # 转发流式响应
            async for chunk in service_stream:
                # 解析JSON块
                chunk_data = json.loads(chunk)
                yield chunk

                # 累积响应内容
                if not chunk_data.get("error", False):
                    # 如果是工具使用步骤，不累积到最终响应
                    if not chunk_data.get("is_tool_use", False):
                        full_response += chunk_data.get("content", "")

                    # 保存来源引用和其他元数据
                    if "sources" in chunk_data and chunk_data.get("done", False):
                        full_metadata["sources"] = chunk_data["sources"]

            # 存储完整的助手回复
            if full_response:
                await self._store_message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=full_response,
                    metadata=full_metadata or metadata,
                )

        except Exception as e:
            # 记录错误消息
            error_msg = f"处理消息时出错: {str(e)}"
            await self._store_message(
                conversation_id=conversation_id,
                role=MessageRole.SYSTEM,
                content=error_msg,
                metadata={"error": True},
            )
            # 返回错误信息
            yield json.dumps({"content": error_msg, "done": True, "error": True})
            raise

    async def _store_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """存储消息到数据库"""
        message_data = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": metadata,
        }
        return await self.message_repo.create(obj_in=message_data)

    async def get_conversation_messages(
        self, conversation_id: UUID, user_id: UUID, skip: int = 0, limit: int = 50
    ) -> List[Message]:
        """获取会话的消息列表"""
        # 检查用户是否有权限访问该会话
        conversation = await self.conversation_repo.get_by_id_for_user(
            id=conversation_id, user_id=user_id
        )
        if not conversation:
            raise PermissionDeniedException(detail="没有权限访问此会话或会话不存在")

        return await self.message_repo.get_by_conversation_id(
            conversation_id=conversation_id, skip=skip, limit=limit
        )
