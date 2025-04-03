from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models.conversation import Conversation
from app.db.repositories.base_repository import BaseRepository
from app.schemas.conversation import ConversationCreate, ConversationUpdate


class ConversationRepository(
    BaseRepository[Conversation, ConversationCreate, ConversationUpdate]
):
    """
    会话仓库类
    """

    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, Conversation)

    async def get_by_user_id(
        self, user_id: UUID, *, skip: int = 0, limit: int = 100
    ) -> List[Conversation]:
        """
        获取用户的所有会话
        """
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_by_id_with_messages(self, id: UUID) -> Optional[Conversation]:
        """
        获取会话及其消息
        """
        query = (
            select(Conversation)
            .where(Conversation.id == id)
            .options(joinedload(Conversation.messages))
        )
        result = await self.db.execute(query)
        return result.scalars().first()

    async def get_by_id_for_user(
        self, id: UUID, user_id: UUID
    ) -> Optional[Conversation]:
        """
        获取特定用户的特定会话
        """
        query = select(Conversation).where(
            Conversation.id == id, Conversation.user_id == user_id
        )
        result = await self.db.execute(query)
        return result.scalars().first()

    async def update_files(
        self, conversation: Conversation, file_ids: List[UUID]
    ) -> Conversation:
        """
        更新会话关联的文件
        """
        # 清空当前关联
        conversation.files = []

        if file_ids:
            # 如需实现，此处应查询相应的UserFile对象并添加到conversation.files
            # 为简化起见，现在假设已经在服务层处理了文件的存在性验证
            pass

        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation
