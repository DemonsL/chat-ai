import os
import uuid
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import BackgroundTasks, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    FileTooLargeException,
    InvalidFileTypeException,
    NotFoundException,
    PermissionDeniedException,
)
from app.db.models.user_file import UserFile
from app.db.repositories.user_file_repository import UserFileRepository
from app.schemas.file import FileStatus, FileType, UserFileCreate, UserFileUpdate


class FileManagementService:
    """
    文件管理服务，处理文件上传和管理
    """
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.user_file_repo = UserFileRepository(db_session)
        
        # 确保上传目录存在
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    async def upload_file(
        self, file: UploadFile, user_id: UUID, background_tasks: BackgroundTasks
    ) -> UserFile:
        """
        上传文件
        """
        # 检查文件类型
        file_type = self._get_file_type(file.filename)
        if not file_type:
            raise InvalidFileTypeException(detail="不支持的文件类型")

        # 检查文件大小
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)  # 重置文件指针
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise FileTooLargeException(detail=f"文件太大，最大允许 {settings.MAX_UPLOAD_SIZE / (1024 * 1024)} MB")

        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if file.filename:
            extension = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{unique_filename}{extension}"
        
        # 保存文件到本地
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 创建文件记录
        user_file_in = UserFileCreate(
            user_id=user_id,
            filename=unique_filename,
            original_filename=file.filename or "未命名文件",
            file_type=file_type,
            file_size=file_size,
            storage_path=file_path,
            status=FileStatus.PENDING
        )
        user_file = await self.user_file_repo.create(obj_in=user_file_in)

        # 在后台处理文件（索引等）
        # TODO: 实现文件处理任务
        background_tasks.add_task(self._process_file, user_file.id)

        return user_file

    async def get_user_files(
        self, user_id: UUID, skip: int = 0, limit: int = 20
    ) -> List[UserFile]:
        """
        获取用户的文件列表
        """
        return await self.user_file_repo.get_by_user_id(user_id, skip=skip, limit=limit)

    async def get_file_by_id(self, file_id: UUID, user_id: UUID) -> Optional[UserFile]:
        """
        获取文件详情
        """
        file = await self.user_file_repo.get_by_id_for_user(file_id, user_id)
        if not file:
            raise NotFoundException(detail="文件不存在")
        return file

    async def delete_file(self, file_id: UUID, user_id: UUID) -> bool:
        """
        删除文件
        """
        # 获取文件并检查权限
        file = await self.user_file_repo.get_by_id_for_user(file_id, user_id)
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
        await self.user_file_repo.delete(id=file_id)
        
        # TODO: 删除相关向量数据库中的索引

        return True

    def _get_file_type(self, filename: Optional[str]) -> Optional[FileType]:
        """
        根据文件名确定文件类型
        """
        if not filename:
            return None
            
        extension = os.path.splitext(filename)[1].lower()
        if extension in ['.pdf']:
            return FileType.PDF
        elif extension in ['.docx', '.doc']:
            return FileType.DOCX
        elif extension in ['.txt', '.md', '.csv']:
            return FileType.TXT
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return FileType.IMAGE
        return None

    async def _process_file(self, file_id: UUID) -> None:
        """
        处理文件（提取文本、索引等）
        这是一个示例方法，实际实现应使用异步处理
        """
        # 更新状态为处理中
        await self.user_file_repo.update_status(file_id, FileStatus.PROCESSING)
        
        try:
            # TODO: 实际文件处理逻辑，如：
            # 1. 从文件中提取文本
            # 2. 分割文本为块
            # 3. 进行嵌入处理
            # 4. 存储到向量数据库

            # 模拟延迟
            import asyncio
            await asyncio.sleep(2)

            # 更新状态为已索引
            await self.user_file_repo.update_status(file_id, FileStatus.INDEXED)
        except Exception as e:
            # 处理错误
            await self.user_file_repo.update_status(file_id, FileStatus.ERROR, str(e))
            raise 