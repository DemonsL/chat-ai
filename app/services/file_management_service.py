import os
import shutil
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (FileTooLargeException,
                                 InvalidFileTypeException, NotFoundException,
                                 PermissionDeniedException)
from app.db.models.user_file import UserFile
from app.db.repositories.user_file_repository import UserFileRepository
from app.schemas.file import FileStatus, FileType
from app.tasks.jobs.file import (analyze_file_task, bulk_upload_task,
                                 export_file_task, process_file_task)


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

        # 确保上传目录存在
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload_file(
        self,
        file: UploadFile,
        user_id: UUID,
        description: Optional[str] = None,
        analyze: bool = False,
    ) -> UserFile:
        """
        上传文件

        参数:
            file: 上传的文件
            user_id: 用户ID
            description: 文件描述
            analyze: 是否执行分析

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

        # 启动Celery任务处理文件
        process_file_task.delay(str(file_record.id))

        # 如果需要，启动文件分析任务
        if analyze:
            analyze_file_task.delay(str(file_record.id))

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
            logger.error(f"删除文件时出错: {str(e)}")

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

    async def export_files(
        self, user_id: UUID, file_ids: List[UUID], notify: bool = True
    ) -> Dict:
        """
        导出多个文件

        参数:
            user_id: 用户ID
            file_ids: 文件ID列表
            notify: 是否通知用户

        返回:
            导出任务信息
        """
        # 检查权限和文件存在
        valid_files = []
        for file_id in file_ids:
            try:
                file = await self.get_file_by_id(file_id, user_id)
                valid_files.append(file)
            except NotFoundException:
                # 跳过不存在的文件
                continue

        if not valid_files:
            raise NotFoundException(detail="未找到可导出的文件")

        # 启动导出任务
        file_id_strs = [str(file.id) for file in valid_files]
        task = export_file_task.delay(str(user_id), file_id_strs, notify=notify)

        return {
            "task_id": task.id,
            "status": "pending",
            "file_count": len(valid_files),
        }

    async def bulk_upload(
        self, user_id: UUID, file_paths: List[str], analyze: bool = False
    ) -> Dict:
        """
        批量上传服务器上已有的文件

        参数:
            user_id: 用户ID
            file_paths: 服务器上的文件路径列表
            analyze: 是否执行分析

        返回:
            任务信息
        """
        # 启动批量上传任务
        task = bulk_upload_task.delay(str(user_id), file_paths, analyze)

        return {
            "task_id": task.id,
            "status": "pending",
            "file_count": len(file_paths),
        }

    async def analyze_existing_file(
        self, file_id: UUID, user_id: UUID, analysis_type: str = "basic"
    ) -> Dict:
        """
        分析现有文件

        参数:
            file_id: 文件ID
            user_id: 用户ID
            analysis_type: 分析类型

        返回:
            任务信息
        """
        # 检查文件存在和权限
        file = await self.file_repo.get_by_id_for_user(file_id, user_id)
        if not file:
            raise NotFoundException(detail="文件不存在")

        # 启动分析任务
        task = analyze_file_task.delay(str(file_id), analysis_type)

        return {
            "task_id": task.id,
            "status": "pending",
            "file_id": str(file_id),
            "analysis_type": analysis_type,
        }

    def _get_file_type(self, filename: Optional[str]) -> Optional[FileType]:
        """根据文件名确定文件类型"""
        if not filename:
            return None

        file_ext = Path(filename).suffix.lower()
        return self.ALLOWED_EXTENSIONS.get(file_ext)

    # 原_process_file方法已由Celery任务替代，可以删除
