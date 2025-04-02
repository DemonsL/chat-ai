from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user_file import UserFile
from app.db.repositories.base_repository import BaseRepository
from app.schemas.file import UserFileCreate, UserFileUpdate


class UserFileRepository(BaseRepository[UserFile, UserFileCreate, UserFileUpdate]):
    """
    用户文件仓库类
    """
    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, UserFile)

    async def get_by_user_id(
        self, user_id: UUID, *, skip: int = 0, limit: int = 20
    ) -> List[UserFile]:
        """
        获取用户的所有文件
        """
        query = (
            select(UserFile)
            .where(UserFile.user_id == user_id)
            .order_by(UserFile.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_by_id_for_user(self, id: UUID, user_id: UUID) -> Optional[UserFile]:
        """
        获取特定用户的特定文件
        """
        query = (
            select(UserFile)
            .where(UserFile.id == id, UserFile.user_id == user_id)
        )
        result = await self.db.execute(query)
        return result.scalars().first()

    async def get_by_ids_for_user(self, ids: List[UUID], user_id: UUID) -> List[UserFile]:
        """
        获取特定用户的多个文件
        """
        query = (
            select(UserFile)
            .where(UserFile.id.in_(ids), UserFile.user_id == user_id)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_status(
        self, file_id: UUID, status: str, error_message: Optional[str] = None
    ) -> Optional[UserFile]:
        """
        更新文件状态
        """
        file = await self.get_by_id(file_id)
        if file:
            file.status = status
            if error_message:
                file.error_message = error_message
            self.db.add(file)
            await self.db.commit()
            await self.db.refresh(file)
        return file 