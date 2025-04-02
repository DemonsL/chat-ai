from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.model_config import ModelConfig
from app.db.repositories.base_repository import BaseRepository
from app.schemas.model import ModelConfigCreate, ModelConfigUpdate


class ModelConfigRepository(BaseRepository[ModelConfig, ModelConfigCreate, ModelConfigUpdate]):
    """
    模型配置仓库类
    """
    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session, ModelConfig)

    async def get_by_model_id(self, model_id: str) -> Optional[ModelConfig]:
        """
        通过模型ID获取配置
        """
        query = select(ModelConfig).where(ModelConfig.model_id == model_id)
        result = await self.db.execute(query)
        return result.scalars().first()

    async def get_active_models(self) -> List[ModelConfig]:
        """
        获取所有活跃模型的配置
        """
        query = select(ModelConfig).where(ModelConfig.is_active.is_(True))
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_models_with_capability(self, capability: str) -> List[ModelConfig]:
        """
        获取具有特定能力的模型
        注意：这需要使用JSONB类型的特殊查询，PostgreSQL 特定
        """
        # 使用JSON操作符 ? 查询具有特定能力的模型
        # 需要使用文本SQL以便使用PostgreSQL的JSONB特定功能
        from sqlalchemy import text
        sql = text(f"SELECT * FROM modelconfig WHERE capabilities ? :capability AND is_active = true")
        result = await self.db.execute(sql, {"capability": capability})
        
        # 处理原生查询结果
        records = result.fetchall()
        models = []
        for record in records:
            # 将记录转换为字典，然后创建ModelConfig对象
            model_data = {col: getattr(record, col) for col in record._mapping.keys()}
            models.append(ModelConfig(**model_data))
        
        return models 