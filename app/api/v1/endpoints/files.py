from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_current_active_user, get_file_service
from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.user import User
from app.schemas.file import UserFileResponse
from app.services.file_management_service import FileManagementService

router = APIRouter()


@router.post(
    "/upload", response_model=UserFileResponse, status_code=status.HTTP_201_CREATED
)
async def upload_file(
    file: UploadFile = File(...),
    description: str = None,
    current_user: User = Depends(get_current_active_user),
    file_service: FileManagementService = Depends(get_file_service),
):
    """
    上传文件

    接收文件上传，验证文件类型和大小，并将其保存到系统中。
    支持文本文件(txt, pdf, docx等)和图片文件(png, jpg, jpeg等)。
    """
    try:
        file_record = await file_service.upload_file(
            file=file,
            user_id=current_user.id,
            description=description,
        )
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
