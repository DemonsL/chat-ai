from typing import Generator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import ALGORITHM
from app.db.models.user import User
from app.db.repositories.user_repository import UserRepository
from app.db.session import get_db
from app.schemas.token import TokenPayload
from app.services.auth_service import AuthService
from app.services.conversation_service import ConversationService
from app.services.file_service import FileManagementService
from app.services.message_service import MessageService
from app.services.model_service import ModelService
from app.services.task_monitor_service import TaskMonitorService
from app.services.user_service import UserService

# OAuth2 token URL和scheme
reusable_oauth2 = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


# 依赖项: 获取数据库会话
async def get_db_session() -> AsyncSession:
    # async for session in get_db():
    async with get_db() as session:
        yield session


# 依赖项: 获取当前用户（可选）
async def get_current_user_optional(
    db_session: AsyncSession = Depends(get_db_session),
    token: Optional[str] = Depends(reusable_oauth2),
) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        return None

    user_repo = UserRepository(db_session)
    user = await user_repo.get_by_id(token_data.sub)

    if not user:
        return None
    return user


# 依赖项: 获取当前用户（必须）
async def get_current_user(
    db_session: AsyncSession = Depends(get_db_session),
    token: str = Depends(reusable_oauth2),
) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
        )

    user_repo = UserRepository(db_session)
    user = await user_repo.get_by_id(token_data.sub)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户未找到")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="用户未激活")
    return user


# 依赖项: 获取当前活跃用户
async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="用户未激活")
    return current_user


# 依赖项: 获取当前管理员用户
async def get_current_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="权限不足，需要管理员权限"
        )
    return current_user


# 服务依赖项
async def get_auth_service(
    db_session: AsyncSession = Depends(get_db_session),
) -> AuthService:
    return AuthService(db_session)


async def get_user_service(
    db_session: AsyncSession = Depends(get_db_session),
) -> UserService:
    return UserService(db_session)


async def get_conversation_service(
    db_session: AsyncSession = Depends(get_db_session),
) -> ConversationService:
    return ConversationService(db_session)


async def get_message_orchestrator(
    db_session: AsyncSession = Depends(get_db_session),
) -> MessageService:
    return MessageService(db_session)


async def get_file_service(
    db_session: AsyncSession = Depends(get_db_session),
) -> FileManagementService:
    return FileManagementService(db_session)


async def get_model_service(
    db_session: AsyncSession = Depends(get_db_session),
) -> ModelService:
    return ModelService(db_session)


def get_task_monitor_service(
    db_session: AsyncSession = Depends(get_db_session),
) -> TaskMonitorService:
    """获取任务监控服务"""
    return TaskMonitorService(db_session)
