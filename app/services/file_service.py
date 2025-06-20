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
from app.llm.manage import LLMManager

# 导入Celery任务 - 确保在模块级别导入
try:
    # 先导入Celery应用以确保任务注册
    from app.tasks.celery import celery_app
    # 然后导入具体任务
    from app.tasks.jobs.file import (analyze_file_task, bulk_upload_task,
                                     export_file_task, process_file_task)
    CELERY_AVAILABLE = True
    logger.info("Celery任务导入成功")
    
    # 验证关键任务是否已注册
    if "tasks.file.process_file" in celery_app.tasks:
        logger.info("文件处理任务已正确注册")
    else:
        logger.warning("文件处理任务未找到，可能影响异步处理")
        
except ImportError as e:
    logger.warning(f"Celery任务导入失败: {str(e)}")
    CELERY_AVAILABLE = False


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

        # LLM管理器延迟初始化，只在需要时创建
        self._llm_mgr = None

    @property 
    def llm_mgr(self):
        """延迟初始化LLM管理器"""
        if self._llm_mgr is None:
            logger.info("初始化LLM管理器用于文件处理")
            from app.llm.manage import LLMManager
            self._llm_mgr = LLMManager()
        return self._llm_mgr

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
            if not CELERY_AVAILABLE:
                logger.warning("Celery不可用，改为同步处理")
                # 如果Celery不可用，直接同步处理
                try:
                    logger.info(f"开始同步处理文件 {file_record.id}...")
                    await self.process_file(file_record.id, user_id)
                    logger.info(f"文件 {file_record.id} 同步处理完成（Celery不可用）")
                except Exception as sync_e:
                    logger.error(f"文件 {file_record.id} 同步处理失败: {str(sync_e)}")
                    await self.file_repo.update_status(
                        file_id=file_record.id, 
                        status=FileStatus.ERROR, 
                        error_message=f"处理失败: {str(sync_e)}"
                    )
                    raise sync_e
            else:
                # 测试Celery连接
                try:
                    from app.tasks.celery import celery_app
                    test_result = celery_app.send_task('test_celery_connection')
                    logger.info(f"Celery连接测试任务ID: {test_result.id}")
                except Exception as test_e:
                    logger.error(f"Celery连接测试失败: {str(test_e)}")
                
                try:
                    # 尝试启动Celery任务处理文件
                    logger.info(f"准备启动异步任务处理文件 {file_record.id}")
                    task_result = process_file_task.delay(str(file_record.id), str(user_id))
                    logger.info(f"文件 {file_record.id} 异步处理任务已启动, 任务ID: {task_result.id}")
                    
                    # 等待任务状态更新
                    import asyncio
                    await asyncio.sleep(0.2)  # 等待200ms让任务有时间启动
                    
                    # 检查任务是否成功启动
                    try:
                        task_state = task_result.state
                        logger.info(f"任务状态: {task_state}")
                        if task_state == 'FAILURE':
                            error_info = getattr(task_result, 'info', task_result.result)
                            logger.warning(f"异步任务立即失败，改为同步处理: {error_info}")
                            raise Exception(f"Celery任务失败: {error_info}")
                        elif task_state in ['PENDING', 'STARTED', 'SUCCESS']:
                            logger.info(f"异步任务启动成功，状态: {task_state}")
                    except AttributeError as attr_error:
                        logger.info(f"无法检查任务状态属性，任务可能已启动: {str(attr_error)}")
                        # 如果无法检查状态，可能是任务已经成功启动，继续执行
                        
                except Exception as e:
                    logger.warning(f"启动异步任务失败，改为同步处理: {str(e)}")
                    # 如果异步任务启动失败，立即改为同步处理
                    try:
                        logger.info(f"开始同步处理文件 {file_record.id}...")
                        await self.process_file(file_record.id, user_id)
                        logger.info(f"文件 {file_record.id} 同步处理完成（异步失败后的备选）")
                    except Exception as sync_e:
                        logger.error(f"文件 {file_record.id} 同步处理也失败: {str(sync_e)}")
                        # 更新文件状态为错误
                        await self.file_repo.update_status(
                            file_id=file_record.id, 
                            status=FileStatus.ERROR, 
                            error_message=f"处理失败: {str(sync_e)}"
                        )
                        # 重新抛出异常
                        raise sync_e

        # 如果需要，启动文件分析任务
        if analyze and CELERY_AVAILABLE:
            try:
                analyze_task_result = analyze_file_task.delay(str(file_record.id))
                logger.info(f"文件分析任务已启动: {file_record.id}, 任务ID: {analyze_task_result.id}")
            except Exception as e:
                logger.warning(f"启动分析任务失败: {str(e)}")
        elif analyze and not CELERY_AVAILABLE:
            logger.warning("Celery不可用，跳过文件分析任务")

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

        if not CELERY_AVAILABLE:
            raise FileProcessingException(detail="导出功能需要Celery支持，但Celery当前不可用")

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
        if not CELERY_AVAILABLE:
            raise FileProcessingException(detail="批量上传功能需要Celery支持，但Celery当前不可用")

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

        if not CELERY_AVAILABLE:
            raise FileProcessingException(detail="文件分析功能需要Celery支持，但Celery当前不可用")

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

    async def process_file_with_fallback(self, file_id: UUID, user_id: UUID) -> Dict:
        """
        使用降级嵌入服务处理文件
        当主要嵌入服务不可用时使用此方法
        
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
            # 创建一个使用OpenAI作为备用嵌入服务的LLM管理器
            from app.llm.rag.retrieval_service import LLMRetrievalService
            
            logger.info("使用OpenAI作为降级嵌入服务处理文件")
            
            # 创建备用检索服务（强制使用OpenAI）
            fallback_retrieval = LLMRetrievalService()
            fallback_retrieval.embedding_provider = 'openai'
            fallback_retrieval.embeddings = fallback_retrieval._fallback_to_openai()
            fallback_retrieval._initialize_vector_store()
            
            if not fallback_retrieval.is_ready:
                raise FileProcessingException(detail="降级嵌入服务初始化失败")
            
            # 使用文件管理器处理文件内容
            document_objects, content_metadata = await self.llm_mgr.process_file_to_documents(
                file_path=file_record.storage_path,
                file_type=file_record.file_type,
                chunk_size=1000,
                chunk_overlap=200,
                user_id=str(user_id),
                file_id=str(file_id),
                file_name=file_record.original_filename
            )

            if not document_objects:
                raise FileProcessingException(detail="无法从文件中提取内容或文件类型不支持")

            # 使用降级检索服务添加文档
            success = await fallback_retrieval.add_documents(
                documents=document_objects,
                user_id=str(user_id),
                file_id=str(file_id),
                conversation_id=None
            )

            if not success:
                raise FileProcessingException(detail="降级向量索引创建失败")

            # 更新文件状态为已索引
            processing_metadata = {
                "chunk_count": len(document_objects),
                "character_count": content_metadata.get("character_count", 0),
                "processing_timestamp": content_metadata,
                "embedding_provider": "openai_fallback",  # 标记使用了降级服务
                **content_metadata
            }

            await self.file_repo.update(
                db_obj=file_record,
                obj_in={"status": FileStatus.INDEXED, "file_metadata": processing_metadata},
            )

            logger.info(f"文件 {file_id} 降级处理成功，用户 {user_id}")

            return {
                "status": "success",
                "file_id": str(file_id),
                "metadata": processing_metadata,
                "note": "使用降级嵌入服务处理"
            }

        except Exception as e:
            # 更新文件状态为错误
            await self.file_repo.update_status(
                file_id=file_id, status=FileStatus.ERROR, error_message=f"降级处理失败: {str(e)}"
            )
            
            logger.error(f"文件 {file_id} 降级处理失败，用户 {user_id}: {str(e)}")
            raise FileProcessingException(detail=f"文件降级处理失败: {str(e)}")

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
            # 使用LLM层处理文件内容，直接生成Document对象
            # process_file_to_documents 方法内部已经包含文件类型验证和元数据增强
            document_objects, content_metadata = await self.llm_mgr.process_file_to_documents(
                file_path=file_record.storage_path,
                file_type=file_record.file_type,
                chunk_size=1000,
                chunk_overlap=200,
                user_id=str(user_id),
                file_id=str(file_id),
                file_name=file_record.original_filename
            )

            if not document_objects:
                raise FileProcessingException(detail="无法从文件中提取内容或文件类型不支持")

            # 添加文档到向量存储，并传递必要的隔离信息
            success = await self.llm_mgr.retrieval_service.add_documents(
                documents=document_objects,
                user_id=str(user_id),
                file_id=str(file_id),
                conversation_id=None  # 文件级别的文档不绑定特定对话
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
