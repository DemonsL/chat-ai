from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_active_user, get_current_admin_user, get_user_service
from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.user import User
from app.schemas.user import UserResponse, UserUpdate
from app.services.user_service import UserService

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def read_current_user(
    current_user: User = Depends(get_current_active_user),
):
    """
    获取当前登录用户信息
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    更新当前登录用户信息
    """
    try:
        updated_user = await user_service.update(
            user_id=current_user.id, 
            user_update=user_in, 
            current_user=current_user
        )
        return updated_user
    except (NotFoundException, PermissionDeniedException) as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get("", response_model=List[UserResponse])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    获取所有用户列表（需要管理员权限）
    """
    users = await user_service.get_all(skip=skip, limit=limit)
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def read_user_by_id(
    user_id: UUID,
    current_user: User = Depends(get_current_active_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    根据ID获取特定用户
    
    - 普通用户只能获取自己的信息
    - 管理员可以获取任何用户的信息
    """
    try:
        if current_user.id != user_id and not current_user.is_admin:
            raise PermissionDeniedException(detail="没有权限访问其他用户信息")
        
        user = await user_service.get_by_id(user_id=user_id)
        return user
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


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_in: UserUpdate,
    current_user: User = Depends(get_current_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    更新指定用户信息（需要管理员权限）
    """
    try:
        updated_user = await user_service.update(
            user_id=user_id, 
            user_update=user_in, 
            current_user=current_user
        )
        return updated_user
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user(
    user_id: UUID,
    current_user: User = Depends(get_current_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    停用指定用户账号（需要管理员权限）
    """
    try:
        await user_service.deactivate(user_id=user_id, current_user=current_user)
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