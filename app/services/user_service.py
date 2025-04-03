from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.core.security import get_password_hash
from app.db.models.user import User
from app.db.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserUpdate


class UserService:
    """
    用户服务，处理用户信息的获取和更新
    """

    def __init__(self, db_session: AsyncSession):
        self.user_repo = UserRepository(db_session)

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """
        根据ID获取用户信息
        """
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundException(detail="用户不存在")
        return user

    async def update(
        self, user_id: UUID, user_update: UserUpdate, current_user_id: UUID
    ) -> User:
        """
        更新用户信息
        """
        # 检查权限（只能更新自己的信息，除非是管理员）
        if user_id != current_user_id:
            # 这里可以添加管理员权限检查，目前简单处理
            current_user = await self.user_repo.get_by_id(current_user_id)
            if not current_user or not current_user.is_admin:
                raise PermissionDeniedException(detail="无权更新其他用户的信息")

        # 获取用户
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundException(detail="用户不存在")

        # 准备更新数据
        update_data = user_update.model_dump(exclude_unset=True)

        # 如果要更新密码，需要哈希处理
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password

        # 执行更新
        return await self.user_repo.update(db_obj=user, obj_in=update_data)

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """
        获取所有用户（通常只有管理员使用）
        """
        return await self.user_repo.get_multi(skip=skip, limit=limit)

    async def deactivate(self, user_id: UUID, current_user_id: UUID) -> User:
        """
        停用用户账号
        """
        # 检查权限（通常只有管理员可以停用账号）
        if user_id == current_user_id:
            raise PermissionDeniedException(detail="不能停用自己的账号")

        current_user = await self.user_repo.get_by_id(current_user_id)
        if not current_user or not current_user.is_admin:
            raise PermissionDeniedException(detail="无权停用用户账号")

        # 获取用户
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundException(detail="用户不存在")

        # 执行停用
        return await self.user_repo.update(db_obj=user, obj_in={"is_active": False})
