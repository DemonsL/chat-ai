from typing import AsyncGenerator, Dict, Optional
from uuid import UUID

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.conversation import Conversation
from app.db.models.message import Message
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.schemas.conversation import ConversationMode
from app.schemas.message import MessageCreate, MessageCreateRequest, MessageRole


class MessageOrchestrator:
    """
    消息协调器，负责根据会话模式路由消息到合适的处理服务
    """
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.conversation_repo = ConversationRepository(db_session)
        self.message_repo = MessageRepository(db_session)
        # TODO: 注入 LLM 服务
        # self.chat_service = ChatService()
        # self.rag_service = RAGService()
        # self.agent_service = AgentService()

    async def handle_message(
        self, 
        conversation_id: UUID, 
        user_id: UUID, 
        message_request: MessageCreateRequest
    ) -> AsyncGenerator[str, None]:
        """
        处理新消息
        """
        # 获取会话并检查权限
        conversation = await self.conversation_repo.get_by_id_for_user(conversation_id, user_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")

        # 存储用户消息
        user_message = await self._store_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=message_request.content
        )

        # 根据会话模式选择处理服务
        try:
            # 为演示，我们在此仅生成一个示例响应
            # TODO: 调用实际的 LLM 服务
            response_content = "这是一个示例响应。实际实现应该根据会话模式调用不同的LLM服务。"
            
            # 模拟流式响应
            yield "这是"
            yield "一个"
            yield "示例"
            yield "响应。"
            yield "实际实现"
            yield "应该根据"
            yield "会话模式"
            yield "调用不同的"
            yield "LLM服务。"
            
            # 存储 AI 响应消息
            await self._store_message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=response_content
            )
        except Exception as e:
            # 处理错误，添加错误消息
            error_message = f"消息处理出错: {str(e)}"
            await self._store_message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=error_message,
                metadata={"error": True, "error_detail": str(e)}
            )
            yield error_message
            raise

    async def _store_message(
        self, 
        conversation_id: UUID, 
        role: MessageRole, 
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        存储消息
        """
        message_create = MessageCreate(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata
        )
        return await self.message_repo.create(obj_in=message_create)

    async def get_conversation_messages(
        self, conversation_id: UUID, user_id: UUID, skip: int = 0, limit: int = 50
    ) -> list[Message]:
        """
        获取会话消息
        """
        # 检查会话权限
        conversation = await self.conversation_repo.get_by_id_for_user(conversation_id, user_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")
        
        # 获取消息
        return await self.message_repo.get_by_conversation_id(
            conversation_id=conversation_id,
            skip=skip,
            limit=limit
        )