from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str
    version: str
    environment: str


@router.get("", response_model=HealthResponse)
async def health_check(db_session: AsyncSession = Depends(get_db_session)):
    """
    健康检查端点

    返回API的当前状态和版本信息。
    """
    return {
        "status": "ok",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }
