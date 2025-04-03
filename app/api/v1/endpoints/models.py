from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (get_current_active_user, get_current_admin_user,
                          get_model_service)
from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.user import User
from app.schemas.model import (ModelCapability, ModelConfigCreate,
                               ModelConfigResponse, ModelConfigUpdate,
                               ModelInfo)
from app.services.model_service import ModelService

router = APIRouter()


@router.get("", response_model=List[ModelInfo])
async def get_models(
    capability: Optional[ModelCapability] = Query(None, description="按能力筛选模型"),
    model_service: ModelService = Depends(get_model_service),
):
    """
    获取所有可用模型列表

    可选按能力筛选：chat, vision, embedding
    """
    if capability:
        models = await model_service.get_models_with_capability(capability)
    else:
        models = await model_service.get_active_models()
    return models


@router.get("/{model_id}", response_model=ModelConfigResponse)
async def get_model_by_id(
    model_id: str,
    current_user: User = Depends(get_current_admin_user),
    model_service: ModelService = Depends(get_model_service),
):
    """
    根据ID获取模型详细配置（需要管理员权限）
    """
    try:
        model = await model_service.get_by_model_id(model_id)
        return model
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "", response_model=ModelConfigResponse, status_code=status.HTTP_201_CREATED
)
async def create_model(
    model_in: ModelConfigCreate,
    current_user: User = Depends(get_current_admin_user),
    model_service: ModelService = Depends(get_model_service),
):
    """
    创建新模型配置（需要管理员权限）
    """
    try:
        model = await model_service.create_model(
            model_in=model_in, is_admin=current_user.is_admin
        )
        return model
    except PermissionDeniedException as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.put("/{model_id}", response_model=ModelConfigResponse)
async def update_model(
    model_id: str,
    model_in: ModelConfigUpdate,
    current_user: User = Depends(get_current_admin_user),
    model_service: ModelService = Depends(get_model_service),
):
    """
    更新模型配置（需要管理员权限）
    """
    try:
        model = await model_service.update_model(
            model_id=model_id, model_update=model_in, is_admin=current_user.is_admin
        )
        return model
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PermissionDeniedException as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
