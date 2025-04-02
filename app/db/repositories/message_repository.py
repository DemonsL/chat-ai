from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.message import Message
from app.db.repositories.base_repository import BaseRepository
from app.schemas.message import MessageCreate, MessageUpdate


class MessageRepository(BaseRepository[Message, MessageCreate, MessageUpdate]):
    """
    消息仓库类
    """
    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, Message)

    async def get_by_conversation_id(
        self, conversation_id: UUID, *, skip: int = 0, limit: int = 50
    ) -> List[Message]:
        """
        获取特定会话的消息列表
        """
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at))
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_conversation_history(
        self, conversation_id: UUID, *, limit: int = 20
    ) -> List[Message]:
        """
        获取会话历史（按时间正序排列，用于构建上下文）
        """
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all() 