import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (FileTooLargeException,
                                 InvalidFileTypeException, NotFoundException,
                                 PermissionDeniedException, FileProcessingException)
from app.db.models.user_file import UserFile
from app.db.repositories.user_file_repository import UserFileRepository
from app.schemas.file import FileStatus, FileType
from app.tasks.jobs.file import (analyze_file_task, bulk_upload_task,
                                 export_file_task, process_file_task)
from app.llm.rag.file_processor import LLMFileProcessor


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

        # 使用LLM层的文件处理器
        self.llm_file_processor = LLMFileProcessor()

    async def upload_file(
        self,
        file: UploadFile,
        user_id: UUID,
        description: Optional[str] = None,
        analyze: bool = False,
        sync_process: bool = False,
    ) -> UserFile:
        """
        上传文件

        参数:
            file: 上传的文件
            user_id: 用户ID
            description: 文件描述
            analyze: 是否执行分析
            sync_process: 是否同步处理文件（默认异步）

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
        unique_filename = f"{user_id}_{uuid.uuid4().hex}{file_ext}"
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
            "file_metadata": {"description": description} if description else {},
        }

        file_record = await self.file_repo.create(obj_in=file_data)

        # 根据sync_process参数决定处理方式
        if sync_process:
            # 同步处理文件
            try:
                await self.process_file(file_record.id, user_id)
                logger.info(f"文件 {file_record.id} 同步处理完成")
            except Exception as e:
                logger.error(f"文件 {file_record.id} 同步处理失败: {str(e)}")
                # 更新文件状态为错误
                await self.file_repo.update_status(
                    file_id=file_record.id, 
                    status=FileStatus.ERROR, 
                    error_message=str(e)
                )
        else:
            # 异步处理文件
            try:
                # 启动Celery任务处理文件
                process_file_task.delay(str(file_record.id), str(user_id))
                logger.info(f"文件 {file_record.id} 异步处理任务已启动")
            except Exception as e:
                logger.warning(f"启动异步任务失败，改为同步处理: {str(e)}")
                # 如果异步任务启动失败，改为同步处理
                try:
                    await self.process_file(file_record.id, user_id)
                    logger.info(f"文件 {file_record.id} 同步处理完成（异步失败后的备选）")
                except Exception as sync_e:
                    logger.error(f"文件 {file_record.id} 同步处理也失败: {str(sync_e)}")
                    # 更新文件状态为错误
                    await self.file_repo.update_status(
                        file_id=file_record.id, 
                        status=FileStatus.ERROR, 
                        error_message=str(sync_e)
                    )

        # 如果需要，启动文件分析任务
        if analyze:
            try:
                analyze_file_task.delay(str(file_record.id))
            except Exception as e:
                logger.warning(f"启动分析任务失败: {str(e)}")

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

    async def process_file(self, file_id: UUID, user_id: UUID) -> Dict:
        """
        处理文件并建立向量索引
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            
        Returns:
            处理结果信息
        """
        # 验证用户权限并获取文件信息
        file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
        if not file_record:
            raise FileProcessingException(detail="文件不存在或无权限访问")

        # 更新文件状态为处理中
        await self.file_repo.update_status(
            file_id=file_id, status=FileStatus.PROCESSING
        )

        try:
            # 验证文件类型
            if not self.llm_file_processor.validate_file_type(file_record.file_type):
                raise FileProcessingException(detail=f"不支持的文件类型: {file_record.file_type}")

            # 使用LLM层处理文件内容，直接生成Document对象
            document_objects, content_metadata = await self.llm_file_processor.process_file_to_documents(
                file_path=file_record.storage_path,
                file_type=file_record.file_type,
                file_id=str(file_id),
                user_id=str(user_id),
                file_name=file_record.original_filename,
                chunk_size=1000,
                chunk_overlap=200
            )

            # 添加到向量存储（使用Document对象）
            success = await self.llm_file_processor.add_documents_to_vector_store(
                documents=document_objects
            )

            if not success:
                raise FileProcessingException(detail="向量索引创建失败")

            # 更新文件状态为已索引
            processing_metadata = {
                "chunk_count": len(document_objects),
                "character_count": content_metadata.get("character_count", 0),
                "processing_timestamp": content_metadata,
                **content_metadata
            }

            await self.file_repo.update(
                db_obj=file_record,
                obj_in={"status": FileStatus.INDEXED, "file_metadata": processing_metadata},
            )

            logger.info(f"文件 {file_id} 处理成功，用户 {user_id}")

            return {
                "status": "success",
                "file_id": str(file_id),
                "metadata": processing_metadata,
            }

        except Exception as e:
            # 更新文件状态为错误
            await self.file_repo.update_status(
                file_id=file_id, status=FileStatus.ERROR, error_message=str(e)
            )
            
            logger.error(f"文件 {file_id} 处理失败，用户 {user_id}: {str(e)}")
            raise FileProcessingException(detail=f"文件处理失败: {str(e)}")

    async def remove_file_from_index(self, file_id: UUID, user_id: UUID) -> bool:
        """
        从向量索引中移除文件
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            
        Returns:
            是否成功移除
        """
        try:
            # 验证文件访问权限
            file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
            if not file_record:
                logger.warning(f"用户 {user_id} 尝试删除不存在或无权限的文件 {file_id}")
                return False

            # 从向量存储中删除
            success = await self.llm_file_processor.remove_from_vector_store(str(file_id))

            if success:
                logger.info(f"成功从向量存储删除文件 {file_id}，用户 {user_id}")
            else:
                logger.warning(f"从向量存储删除文件 {file_id} 失败，用户 {user_id}")

            return success

        except Exception as e:
            logger.error(f"删除文件索引失败，文件 {file_id}，用户 {user_id}: {str(e)}")
            raise FileProcessingException(detail=f"删除文件索引失败: {str(e)}")

    async def reprocess_file(self, file_id: UUID, user_id: UUID) -> Dict:
        """
        重新处理文件
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            
        Returns:
            处理结果信息
        """
        try:
            # 先从向量存储中删除旧数据
            await self.remove_file_from_index(file_id, user_id)
            
            # 重新处理文件
            result = await self.process_file(file_id, user_id)
            
            logger.info(f"文件 {file_id} 重新处理成功，用户 {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"文件 {file_id} 重新处理失败，用户 {user_id}: {str(e)}")
            raise

    async def get_file_processing_status(self, file_id: UUID, user_id: UUID) -> Dict:
        """
        获取文件处理状态
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            
        Returns:
            文件状态信息
        """
        file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
        if not file_record:
            raise FileProcessingException(detail="文件不存在或无权限访问")

        return {
            "file_id": str(file_id),
            "status": file_record.status,
            "file_metadata": file_record.file_metadata,
            "created_at": file_record.created_at.isoformat() if file_record.created_at else None,
            "updated_at": file_record.updated_at.isoformat() if file_record.updated_at else None,
        }

    async def batch_process_files(self, file_ids: List[UUID], user_id: UUID) -> Dict:
        """
        批量处理文件
        
        Args:
            file_ids: 文件ID列表
            user_id: 用户ID
            
        Returns:
            批量处理结果
        """
        results = []
        successful = 0
        failed = 0

        for file_id in file_ids:
            try:
                result = await self.process_file(file_id, user_id)
                results.append({
                    "file_id": str(file_id),
                    "success": True,
                    "result": result
                })
                successful += 1
            except Exception as e:
                results.append({
                    "file_id": str(file_id),
                    "success": False,
                    "error": str(e)
                })
                failed += 1

        logger.info(f"批量处理完成，用户 {user_id}: 成功 {successful}, 失败 {failed}")

        return {
            "total": len(file_ids),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    def get_supported_file_types(self) -> List[str]:
        """
        获取支持的文件类型
        
        Returns:
            支持的文件类型列表
        """
        return self.llm_file_processor.get_supported_file_types()

    async def validate_file_for_processing(self, file_id: UUID, user_id: UUID) -> bool:
        """
        验证文件是否可以处理
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            
        Returns:
            是否可以处理
        """
        try:
            file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
            if not file_record:
                return False

            # 检查文件是否存在
            if not os.path.exists(file_record.storage_path):
                return False

            # 检查文件类型是否支持
            return self.llm_file_processor.validate_file_type(file_record.file_type)

        except Exception as e:
            logger.error(f"验证文件 {file_id} 失败: {str(e)}")
            return False
