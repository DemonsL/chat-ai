import subprocess
from datetime import datetime

from loguru import logger

from app.core.config import settings
from app.tasks.base import async_task


@async_task(name="tasks.scripts.run_shell_script", max_retries=1)
async def run_shell_script_task(
    self, script_path: str, arguments: list = None, timeout: int = 300
):
    """
    运行Shell脚本

    参数:
        script_path: 脚本路径
        arguments: 命令行参数
        timeout: 超时时间(秒)
    """
    logger.info(f"执行脚本: {script_path}")

    try:
        cmd = [script_path]
        if arguments:
            cmd.extend(arguments)

        # 执行脚本
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        )

        logger.info(f"脚本执行成功: {script_path}")
        return {
            "success": True,
            "script": script_path,
            "arguments": arguments,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "executed_at": datetime.now().isoformat(),
        }

    except subprocess.TimeoutExpired:
        logger.error(f"脚本执行超时: {script_path}")
        return {
            "success": False,
            "script": script_path,
            "error": "执行超时",
            "timeout": timeout,
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"脚本执行失败: {script_path}, 返回值: {e.returncode}")
        return {
            "success": False,
            "script": script_path,
            "error": "执行失败",
            "returncode": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr,
        }

    except Exception as e:
        logger.error(f"脚本执行异常: {script_path}, 错误: {str(e)}")
        return {"success": False, "script": script_path, "error": str(e)}


@async_task(name="tasks.scripts.database_backup", queue="scheduled")
async def database_backup_task(self, backup_type: str = "full"):
    """
    数据库备份任务

    参数:
        backup_type: 备份类型 (full, incremental)
    """
    import os
    from datetime import datetime

    backup_dir = settings.BACKUP_DIR
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(
        backup_dir, f"{settings.PROJECT_NAME}_{backup_type}_{timestamp}.sql"
    )

    # 构建备份命令
    command = [
        "pg_dump",
        "-h",
        settings.POSTGRES_HOST,
        "-p",
        str(settings.POSTGRES_PORT),
        "-U",
        settings.POSTGRES_USER,
        "-d",
        settings.POSTGRES_DB,
        "-f",
        backup_file,
    ]

    # 如果是增量备份，添加额外参数
    if backup_type == "incremental":
        # 添加增量备份相关参数
        pass

    # 设置环境变量
    env = os.environ.copy()
    env["PGPASSWORD"] = settings.POSTGRES_PASSWORD

    try:
        # 执行备份命令
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30分钟超时
            check=True,
        )

        # 检查备份文件是否存在且大小合理
        if os.path.exists(backup_file) and os.path.getsize(backup_file) > 0:
            logger.info(f"数据库备份成功: {backup_file}")
            return {
                "success": True,
                "backup_file": backup_file,
                "backup_type": backup_type,
                "size": os.path.getsize(backup_file),
                "executed_at": datetime.now().isoformat(),
            }
        else:
            logger.error(f"数据库备份失败: 备份文件为空或不存在")
            return {
                "success": False,
                "error": "备份文件为空或不存在",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

    except Exception as e:
        logger.error(f"数据库备份异常: {str(e)}")
        return {"success": False, "error": str(e)}
