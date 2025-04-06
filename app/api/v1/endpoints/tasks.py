from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.dependencies import (get_current_active_user,
                                  get_current_admin_user,
                                  get_task_monitor_service)
from app.db.models.user import User
from app.services.task_monitor_service import TaskMonitorService

router = APIRouter()


@router.get("/{task_id}/status", response_model=Dict)
async def get_task_status(
    task_id: str = Path(..., description="任务ID"),
    current_user: User = Depends(get_current_active_user),
    task_service: TaskMonitorService = Depends(get_task_monitor_service),
):
    """
    获取任务状态
    """
    try:
        return await task_service.get_task_status(task_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务状态失败: {str(e)}",
        )


@router.get("/active", response_model=List[Dict])
async def get_active_tasks(
    current_user: User = Depends(get_current_admin_user),
    task_service: TaskMonitorService = Depends(get_task_monitor_service),
):
    """
    获取活动任务列表（仅管理员）
    """
    try:
        return await task_service.get_active_tasks()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取活动任务失败: {str(e)}",
        )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_task(
    task_id: str = Path(..., description="任务ID"),
    terminate: bool = Query(False, description="是否终止正在运行的任务"),
    current_user: User = Depends(get_current_admin_user),
    task_service: TaskMonitorService = Depends(get_task_monitor_service),
):
    """
    撤销任务（仅管理员）
    """
    try:
        await task_service.revoke_task(task_id, terminate)
        return {"success": True}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"撤销任务失败: {str(e)}",
        )
