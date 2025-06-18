from typing import List, Optional
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Form
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_current_active_user, get_file_service, get_conversation_service
from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.user import User
from app.schemas.file import UserFileResponse
from app.services.file_service import FileManagementService
from app.services.conversation_service import ConversationService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload", response_model=UserFileResponse, status_code=status.HTTP_201_CREATED
)
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    conversation_id: Optional[UUID] = Form(None),
    sync_process: bool = Form(False),  # 改为默认异步处理
    current_user: User = Depends(get_current_active_user),
    file_service: FileManagementService = Depends(get_file_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
):
    """
    上传文件

    接收文件上传，验证文件类型和大小，并将其保存到系统中。
    支持文本文件(txt, pdf, docx等)和图片文件(png, jpg, jpeg等)。
    
    参数:
        file: 上传的文件
        description: 文件描述
        conversation_id: 可选的对话ID，上传后自动关联到该对话
        sync_process: 是否同步处理文件（默认False，异步处理更快）
    """
    try:
        # 1. 上传文件
        file_record = await file_service.upload_file(
            file=file,
            user_id=current_user.id,
            description=description,
            sync_process=sync_process,
        )
        
        # 2. 如果指定了对话ID，将文件关联到对话
        if conversation_id:
            logger.info(f"尝试将文件 {file_record.id} 关联到对话 {conversation_id}")
            try:
                # 验证对话是否存在且属于当前用户
                conversation = await conversation_service.get_by_id(
                    conversation_id=conversation_id, 
                    user_id=current_user.id
                )
                if conversation:
                    # 获取当前对话已关联的文件ID列表
                    current_file_ids = [f.id for f in (conversation.files or [])]
                    logger.info(f"对话 {conversation_id} 当前关联文件: {current_file_ids}")
                    
                    # 添加新文件ID
                    updated_file_ids = current_file_ids + [file_record.id]
                    logger.info(f"更新后的文件列表: {updated_file_ids}")
                    
                    # 更新对话的文件关联，只更新file_ids，不更新其他字段
                    await conversation_service.conversation_repo.update_files(
                        conversation, updated_file_ids
                    )
                    logger.info(f"成功将文件 {file_record.id} 关联到对话 {conversation_id}")
                    
                    # 验证文件关联是否成功
                    updated_conversation = await conversation_service.get_by_id(
                        conversation_id=conversation_id, 
                        user_id=current_user.id
                    )
                    logger.info(f"验证关联结果: 对话 {conversation_id} 现在关联了 {len(updated_conversation.files) if updated_conversation.files else 0} 个文件")
                else:
                    logger.warning(f"对话 {conversation_id} 不存在或无权限")
            except NotFoundException as e:
                # 对话不存在，忽略关联操作
                logger.warning(f"关联文件到对话失败: {str(e)}")
                pass
        
        return file_record
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=List[UserFileResponse])
async def get_user_files(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    file_service: FileManagementService = Depends(get_file_service),
):
    """
    获取当前用户的所有文件
    """
    files = await file_service.get_user_files(
        user_id=current_user.id,
        skip=skip,
        limit=limit,
    )
    return files


@router.get("/{file_id}", response_model=UserFileResponse)
async def get_file_info(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user),
    file_service: FileManagementService = Depends(get_file_service),
):
    """
    获取特定文件的详细信息
    """
    try:
        file = await file_service.get_file_by_id(
            file_id=file_id,
            user_id=current_user.id,
        )
        return file
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


@router.get("/{file_id}/download")
async def download_file(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user),
    file_service: FileManagementService = Depends(get_file_service),
):
    """
    下载特定文件
    """
    try:
        file_info, file_stream = await file_service.download_file(
            file_id=file_id,
            user_id=current_user.id,
        )

        return StreamingResponse(
            file_stream,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{file_info.filename}"'
            },
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


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user),
    file_service: FileManagementService = Depends(get_file_service),
):
    """
    删除特定文件
    """
    try:
        await file_service.delete_file(
            file_id=file_id,
            user_id=current_user.id,
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


@router.get("/{file_id}/status", response_model=UserFileResponse)
async def get_file_status(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user),
    file_service: FileManagementService = Depends(get_file_service),
):
    """
    获取文件处理状态
    
    用于查询文件上传和处理的当前状态，支持轮询检查。
    """
    try:
        file = await file_service.get_file_by_id(
            file_id=file_id,
            user_id=current_user.id,
        )
        return file
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
