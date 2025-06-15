from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import (get_conversation_service,
                                  get_current_active_user,
                                  get_message_orchestrator)
from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.user import User
from app.schemas.conversation import (Conversation, ConversationCreate,
                                      ConversationUpdate)
from app.schemas.message import MessageResponse
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService

router = APIRouter()


@router.post("", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_in: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
):
    """
    创建新对话
    """
    try:
        conversation = await conversation_service.create(
            user_id=current_user.id, conv_create=conversation_in
        )
        return conversation
    except (NotFoundException, ValueError, PermissionDeniedException) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=List[Conversation])
async def read_conversations(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
):
    """
    获取当前用户的所有对话
    """
    conversations = await conversation_service.get_by_user_id(
        user_id=current_user.id, skip=skip, limit=limit
    )
    return conversations


@router.get("/{conversation_id}", response_model=Conversation)
async def read_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_active_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
):
    """
    根据ID获取特定对话
    """
    try:
        conversation = await conversation_service.get_by_id(
            conversation_id=conversation_id, user_id=current_user.id
        )
        return conversation
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


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def read_conversation_messages(
    conversation_id: UUID,
    current_user: User = Depends(get_current_active_user),
    message_orchestrator: MessageService = Depends(get_message_orchestrator),
):
    """
    获取对话中的所有消息
    """
    try:
        messages = await message_orchestrator.get_conversation_messages(
            conversation_id=conversation_id, user_id=current_user.id
        )
        
        # 手动处理字段映射：msg_metadata -> metadata
        result = []
        for message in messages:
            message_dict = {
                "id": str(message.id),
                "conversation_id": str(message.conversation_id),
                "role": message.role,
                "content": message.content,
                "tokens": message.tokens,
                "metadata": message.msg_metadata,  # 关键修复：映射字段名
                "created_at": message.created_at.isoformat() if message.created_at else None,
                "updated_at": message.updated_at.isoformat() if message.updated_at else None,
            }
            result.append(message_dict)
        
        return result
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


@router.put("/{conversation_id}", response_model=Conversation)
async def update_conversation(
    conversation_id: UUID,
    conversation_in: ConversationUpdate,
    current_user: User = Depends(get_current_active_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
):
    """
    更新对话设置
    """
    try:
        conversation = await conversation_service.update(
            conversation_id=conversation_id,
            conversation_in=conversation_in,
            user_id=current_user.id,
        )
        return conversation
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
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_active_user),
    conversation_service: ConversationService = Depends(get_conversation_service),
):
    """
    删除对话
    """
    try:
        await conversation_service.delete(
            conversation_id=conversation_id, user_id=current_user.id
        )
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
