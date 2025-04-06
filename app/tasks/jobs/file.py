from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from loguru import logger

from app.core.exceptions import FileProcessingException
from app.schemas.file import FileStatus
from app.tasks.base import async_task, get_async_db_session
from app.tasks.jobs.email import send_email_task


@async_task(
    name="tasks.file.process_file",
    queue="file_tasks",
    max_retries=3,
    retry_backoff=True,
)
async def process_file_task(self, file_id: str):
    """
    处理文件任务

    参数:
        file_id: 文件ID字符串

    返回:
        处理结果信息
    """
    from app.db.repositories.user_file_repository import UserFileRepository
    from app.llm.rag.file_processor import FileProcessor

    async for session in get_async_db_session():
        file_repo = UserFileRepository(session)
        file_processor = FileProcessor(file_repo)

        file_id_uuid = UUID(file_id)

        try:
            # 获取文件信息
            file_record = await file_repo.get_by_id(file_id_uuid)
            if not file_record:
                raise FileProcessingException(detail="文件不存在")

            # 更新文件状态为处理中
            await file_repo.update_status(
                file_id=file_id_uuid, status=FileStatus.PROCESSING
            )

            # 使用文件处理器进行处理
            result = await file_processor.process_file(file_id_uuid)
            logger.info(f"文件处理成功: {result}")
            return result

        except Exception as e:
            # 记录错误
            logger.error(f"文件处理失败: {str(e)}")
            # 更新文件状态
            await file_repo.update_status(
                file_id=file_id_uuid, status=FileStatus.ERROR, error_message=str(e)
            )
            raise e


@async_task(name="tasks.file.analyze_file", queue="file_tasks", max_retries=2)
async def analyze_file_task(self, file_id: str, analysis_type: str = "basic"):
    """
    分析文件内容任务

    参数:
        file_id: 文件ID
        analysis_type: 分析类型(basic, deep, summary等)

    返回:
        分析结果
    """
    from datetime import datetime

    from app.db.repositories.user_file_repository import UserFileRepository

    async for session in get_async_db_session():
        file_repo = UserFileRepository(session)
        file_id_uuid = UUID(file_id)

        # 获取文件信息
        file_record = await file_repo.get_by_id(file_id_uuid)
        if not file_record:
            raise FileProcessingException(detail="文件不存在")

        # 根据文件类型和分析类型执行不同的分析
        if file_record.file_type == "pdf":
            # PDF分析逻辑
            result = {"analysis_type": analysis_type, "content_type": "pdf"}
        elif file_record.file_type in ["image", "png", "jpg", "jpeg"]:
            # 图像分析逻辑
            result = {"analysis_type": analysis_type, "content_type": "image"}
        else:
            # 其他文件类型分析
            result = {"analysis_type": analysis_type, "content_type": "text"}

        # 更新文件元数据
        metadata = file_record.metadata or {}
        metadata["analysis"] = {
            "type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }

        await file_repo.update(db_obj=file_record, obj_in={"metadata": metadata})

        return result


@async_task(name="tasks.file.export_file", queue="export")
async def export_file_task(
    self,
    user_id: str,
    file_ids: List[str],
    export_format: str = "zip",
    notify: bool = True,
):
    """
    导出文件任务

    参数:
        user_id: 用户ID
        file_ids: 文件ID列表
        export_format: 导出格式
        notify: 是否通知用户
    """
    import os
    import tempfile
    import zipfile

    from app.db.repositories.user_file_repository import UserFileRepository
    from app.db.repositories.user_repository import UserRepository

    # 生成导出文件名
    export_filename = (
        f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
    )
    export_path = os.path.join(tempfile.gettempdir(), export_filename)

    user_id_uuid = UUID(user_id)
    file_id_uuids = [UUID(file_id) for file_id in file_ids]

    async for session in get_async_db_session():
        file_repo = UserFileRepository(session)
        user_repo = UserRepository(session)

        # 获取用户
        user = await user_repo.get_by_id(user_id_uuid)
        if not user:
            raise ValueError(f"用户不存在: {user_id}")

        # 获取所有文件
        files = []
        for file_id in file_id_uuids:
            file = await file_repo.get_by_id_for_user(file_id, user_id_uuid)
            if file:
                files.append(file)

        if not files:
            raise ValueError("没有找到有效的文件")

        # 创建ZIP文件
        with zipfile.ZipFile(export_path, "w") as zip_file:
            for file in files:
                if os.path.exists(file.storage_path):
                    # 使用原始文件名添加到ZIP
                    zip_file.write(file.storage_path, arcname=file.original_filename)

        # 记录导出信息到数据库
        export_record = {
            "user_id": user_id_uuid,
            "file_path": export_path,
            "file_name": export_filename,
            "file_count": len(files),
            "created_at": datetime.now().isoformat(),
        }

        # 如果配置了通知
        if notify and user.email:
            await send_email_task(
                email_to=user.email,
                subject="您的文件导出已完成",
                html_content=f"""
                <p>您好 {user.username},</p>
                <p>您请求的{len(files)}个文件导出已完成，请登录系统下载。</p>
                <p>谢谢!</p>
                """,
            )

        return {
            "success": True,
            "export_path": export_path,
            "file_count": len(files),
            "format": export_format,
        }


@async_task(name="tasks.file.bulk_upload", queue="file_tasks")
async def bulk_upload_task(
    self, user_id: str, file_paths: List[str], analyze: bool = False
):
    """
    批量上传处理本地文件

    参数:
        user_id: 用户ID
        file_paths: 服务器上文件路径列表
        analyze: 是否执行分析
    """
    import os
    import shutil
    from pathlib import Path

    from app.db.repositories.user_file_repository import UserFileRepository
    from app.schemas.file import FileStatus, FileType

    user_id_uuid = UUID(user_id)
    results = []

    ALLOWED_EXTENSIONS = {
        ".txt": "txt",
        ".pdf": "pdf",
        ".docx": "docx",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
    }

    async for session in get_async_db_session():
        file_repo = UserFileRepository(session)

        for file_path in file_paths:
            if not os.path.exists(file_path):
                results.append(
                    {"path": file_path, "success": False, "error": "文件不存在"}
                )
                continue

            try:
                # 获取文件信息
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                file_ext = Path(file_name).suffix.lower()

                # 检查文件类型
                if file_ext not in ALLOWED_EXTENSIONS:
                    results.append(
                        {
                            "path": file_path,
                            "success": False,
                            "error": f"不支持的文件类型: {file_ext}",
                        }
                    )
                    continue

                # 生成唯一文件名
                unique_filename = f"{user_id}_{UUID().hex}{file_ext}"
                storage_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

                # 复制文件到存储目录
                shutil.copy2(file_path, storage_path)

                # 获取文件类型
                file_type = ALLOWED_EXTENSIONS.get(file_ext, "unknown")

                # 创建数据库记录
                file_data = {
                    "user_id": user_id_uuid,
                    "filename": unique_filename,
                    "original_filename": file_name,
                    "file_type": file_type,
                    "file_size": file_size,
                    "storage_path": storage_path,
                    "status": FileStatus.PENDING,
                    "metadata": {"source": "bulk_upload"},
                }

                file_record = await file_repo.create(obj_in=file_data)

                # 启动处理任务
                process_file_task.delay(str(file_record.id))

                # 启动分析任务（如果需要）
                if analyze:
                    analyze_file_task.delay(str(file_record.id))

                results.append(
                    {
                        "path": file_path,
                        "success": True,
                        "file_id": str(file_record.id),
                        "file_type": file_type,
                    }
                )

            except Exception as e:
                results.append({"path": file_path, "success": False, "error": str(e)})

        return {
            "success": True,
            "total": len(file_paths),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results,
        }
