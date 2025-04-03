import os
import shutil
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (FileTooLargeException,
                                 InvalidFileTypeException, NotFoundException,
                                 PermissionDeniedException)
from app.db.models.user_file import UserFile
from app.db.repositories.user_file_repository import UserFileRepository
from app.llm.rag.file_processor import FileProcessor
from app.schemas.file import FileStatus, FileType


class FileManagementService:
    """
    文件管理服务
    """

    ALLOWED_EXTENSIONS = {
        ".txt": FileType.TXT,
        ".pdf": FileType.PDF,
        ".docx": FileType.DOCX,
        ".png": FileType.IMAGE,
        ".jpg": FileType.IMAGE,
        ".jpeg": FileType.IMAGE,
    }

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.file_repo = UserFileRepository(db_session)
        self.upload_dir = settings.UPLOAD_DIR
        self.file_processor = FileProcessor(self.file_repo)

        # 确保上传目录存在
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload_file(
        self,
        file: UploadFile,
        user_id: UUID,
        background_tasks: BackgroundTasks,
        description: Optional[str] = None,
    ) -> UserFile:
        """
        上传文件

        参数:
            file: 上传的文件
            user_id: 用户ID
            background_tasks: 后台任务
            description: 文件描述

        返回:
            UserFile 记录
        """
        # 检查文件大小
        file.file.seek(0, 2)  # 移动到文件末尾
        file_size = file.file.tell()  # 获取当前位置即文件大小
        file.file.seek(0)  # 重置文件指针到开头

        if file_size > settings.MAX_UPLOAD_SIZE:
            raise FileTooLargeException(
                detail=f"文件过大，最大允许{settings.MAX_UPLOAD_SIZE / (1024 * 1024)}MB"
            )

        # 检查文件类型
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise InvalidFileTypeException(detail=f"不支持的文件类型: {file_ext}")

        # 生成唯一文件名
        unique_filename = f"{user_id}_{UUID().hex}{file_ext}"
        storage_path = os.path.join(self.upload_dir, unique_filename)

        # 保存文件
        with open(storage_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 获取文件类型
        file_type = self._get_file_type(file.filename)

        # 创建数据库记录
        file_data = {
            "user_id": user_id,
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_type": file_type.value if file_type else "unknown",
            "file_size": file_size,
            "storage_path": storage_path,
            "status": FileStatus.PENDING,
            "metadata": {"description": description} if description else {},
        }

        file_record = await self.file_repo.create(obj_in=file_data)

        # 添加后台任务处理文件
        background_tasks.add_task(self._process_file, file_record.id)

        return file_record

    async def get_user_files(
        self, user_id: UUID, skip: int = 0, limit: int = 20
    ) -> List[UserFile]:
        """获取用户的所有文件"""
        return await self.file_repo.get_by_user_id(user_id, skip=skip, limit=limit)

    async def get_file_by_id(self, file_id: UUID, user_id: UUID) -> Optional[UserFile]:
        """获取特定文件（检查权限）"""
        file = await self.file_repo.get_by_id_for_user(file_id, user_id)
        if not file:
            raise NotFoundException(detail="文件不存在")
        return file

    async def delete_file(self, file_id: UUID, user_id: UUID) -> bool:
        """删除文件（检查权限）"""
        # 获取文件并检查权限
        file = await self.file_repo.get_by_id_for_user(file_id, user_id)
        if not file:
            raise NotFoundException(detail="文件不存在")

        # 删除物理文件
        try:
            if os.path.exists(file.storage_path):
                os.remove(file.storage_path)
        except Exception as e:
            # 记录错误但继续删除数据库记录
            print(f"删除文件时出错: {str(e)}")

        # 删除数据库记录
        await self.file_repo.delete(id=file_id)
        return True

    async def download_file(
        self, file_id: UUID, user_id: UUID
    ) -> Tuple[UserFile, BinaryIO]:
        """下载文件（检查权限）"""
        # 获取文件并检查权限
        file = await self.file_repo.get_by_id_for_user(file_id, user_id)
        if not file:
            raise NotFoundException(detail="文件不存在")

        # 检查物理文件是否存在
        if not os.path.exists(file.storage_path):
            raise NotFoundException(detail="文件不存在")

        # 打开文件流
        file_stream = open(file.storage_path, "rb")

        # 返回文件记录和文件流
        return file, file_stream

    def _get_file_type(self, filename: Optional[str]) -> Optional[FileType]:
        """根据文件名确定文件类型"""
        if not filename:
            return None

        file_ext = Path(filename).suffix.lower()
        return self.ALLOWED_EXTENSIONS.get(file_ext)

    async def _process_file(self, file_id: UUID) -> None:
        """
        处理文件（后台任务）

        提取文本内容并创建向量索引
        """
        try:
            # 使用文件处理器进行处理
            result = await self.file_processor.process_file(file_id)
            print(f"文件处理成功: {result}")
        except Exception as e:
            # 记录错误
            print(f"文件处理失败: {str(e)}")
            # 更新文件状态
            await self.file_repo.update_status(
                file_id=file_id, status=FileStatus.ERROR, error_message=str(e)
            )
