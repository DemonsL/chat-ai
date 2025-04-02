from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.model_config import ModelConfig
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.schemas.model import ModelCapability, ModelConfigCreate, ModelConfigUpdate, ModelInfo


class ModelService:
    """
    模型服务，处理模型信息的获取和管理
    """
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.model_repo = ModelConfigRepository(db_session)

    async def get_active_models(self) -> List[ModelInfo]:
        """
        获取所有激活的模型
        """
        models = await self.model_repo.get_active_models()
        return [
            ModelInfo(
                id=model.model_id,
                name=model.display_name,
                provider=model.provider,
                capabilities=model.capabilities,
                max_tokens=model.max_tokens
            )
            for model in models
        ]

    async def get_models_with_capability(self, capability: ModelCapability) -> List[ModelInfo]:
        """
        获取支持特定能力的模型
        """
        models = await self.model_repo.get_models_with_capability(capability)
        return [
            ModelInfo(
                id=model.model_id,
                name=model.display_name,
                provider=model.provider,
                capabilities=model.capabilities,
                max_tokens=model.max_tokens
            )
            for model in models
        ]

    async def get_by_model_id(self, model_id: str) -> Optional[ModelConfig]:
        """
        根据ID获取模型配置
        """
        model = await self.model_repo.get_by_model_id(model_id)
        if not model:
            raise NotFoundException(detail="模型不存在")
        return model

    async def create_model(self, model_in: ModelConfigCreate, is_admin: bool) -> ModelConfig:
        """
        创建新模型配置（仅管理员）
        """
        if not is_admin:
            raise PermissionDeniedException(detail="需要管理员权限")
            
        # 检查模型ID是否已存在
        existing_model = await self.model_repo.get_by_model_id(model_in.model_id)
        if existing_model:
            raise PermissionDeniedException(detail="该模型ID已存在")
            
        return await self.model_repo.create(obj_in=model_in)

    async def update_model(
        self, model_id: str, model_update: ModelConfigUpdate, is_admin: bool
    ) -> ModelConfig:
        """
        更新模型配置（仅管理员）
        """
        if not is_admin:
            raise PermissionDeniedException(detail="需要管理员权限")
            
        # 获取现有模型
        model = await self.model_repo.get_by_model_id(model_id)
        if not model:
            raise NotFoundException(detail="模型不存在")
            
        return await self.model_repo.update(db_obj=model, obj_in=model_update) 