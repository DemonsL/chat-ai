from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.plugins.cache.manager import get_cache_client

router = APIRouter()


class HealthCheck(BaseModel):
    status: str
    database: bool
    cache: bool


@router.get(
    "",
    response_model=HealthCheck,
    status_code=status.HTTP_200_OK,
    summary="系统健康检查",
    description="检查API和依赖服务的状态",
)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    系统健康检查
    """
    # 数据库连接检查
    db_status = True
    try:
        # 进行简单的数据库查询
        await db.execute("SELECT 1")
    except Exception:
        db_status = False

    # 缓存连接检查
    cache_status = True
    try:
        cache = get_cache_client()
        await cache.set("health_check", "ok", expire=10)
        test_value = await cache.get("health_check")
        if test_value != "ok":
            cache_status = False
    except Exception:
        cache_status = False

    return {
        "status": "healthy" if db_status and cache_status else "unhealthy",
        "database": db_status,
        "cache": cache_status,
    }
