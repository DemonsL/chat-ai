from datetime import timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import CredentialsException, UserExistsException
from app.core.security import (create_access_token, get_password_hash,
                               verify_password)
from app.db.models.user import User
from app.db.repositories.user_repository import UserRepository
from app.schemas.token import Token
from app.schemas.user import UserCreate, UserInDB


class AuthService:
    """
    认证服务，处理用户注册和登录
    """

    def __init__(self, db_session: AsyncSession):
        self.user_repo = UserRepository(db_session)

    async def register(self, user_in: UserCreate) -> User:
        """
        注册新用户
        """
        # 检查邮箱是否已存在
        existing_user = await self.user_repo.get_by_email(user_in.email)
        if existing_user:
            raise UserExistsException(detail="该邮箱已被注册")

        # 检查用户名是否已存在
        existing_user = await self.user_repo.get_by_username(user_in.username)
        if existing_user:
            raise UserExistsException(detail="该用户名已被使用")

        # 创建新用户
        hashed_password = get_password_hash(user_in.password)
        user_data = user_in.model_dump(exclude={"password"})
        
        # 添加额外字段
        user_data.update({
            "hashed_password": hashed_password,
            "is_admin": False  # 默认不是管理员
        })

        # 保存到数据库
        return await self.user_repo.create(obj_in=user_data)

    async def login(self, username: str, password: str) -> Token:
        """
        用户登录，验证用户名和密码
        """
        # 通过用户名查找用户
        user = await self.user_repo.get_by_username(username)

        # 如果找不到用户，尝试通过邮箱查找
        if not user:
            user = await self.user_repo.get_by_email(username)

        # 如果仍然找不到，或密码验证失败
        if not user or not verify_password(password, user.hashed_password):
            raise CredentialsException(detail="用户名/邮箱或密码错误")

        # 检查用户状态
        if not user.is_active:
            raise CredentialsException(detail="用户账号已停用")

        # 创建 JWT token
        token = create_access_token(subject=str(user.id))

        return Token(access_token=token, token_type="bearer")
