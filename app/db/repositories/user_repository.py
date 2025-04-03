from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User
from app.db.repositories.base_repository import BaseRepository
from app.schemas.user import UserCreate, UserUpdate


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """
    用户仓库类
    """

    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, User)

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        通过邮箱获取用户
        """
        query = select(User).where(User.email == email)
        result = await self.db.execute(query)
        return result.scalars().first()

    async def get_by_username(self, username: str) -> Optional[User]:
        """
        通过用户名获取用户
        """
        query = select(User).where(User.username == username)
        result = await self.db.execute(query)
        return result.scalars().first()
