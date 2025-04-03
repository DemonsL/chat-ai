from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_active_user, get_message_orchestrator
from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.user import User
from app.schemas.message import MessageCreate
from app.services.message_orchestrator import MessageOrchestrator

router = APIRouter()


@router.post("/{conversation_id}/send", response_class=StreamingResponse)
async def send_message(
    conversation_id: UUID,
    message_in: MessageCreate,
    current_user: User = Depends(get_current_active_user),
    message_orchestrator: MessageOrchestrator = Depends(get_message_orchestrator),
):
    """
    发送消息并获取流式响应

    提交一条消息到指定对话中，并以流形式返回AI的响应。
    每次响应都是一个JSON对象，包含内容片段和完成状态。

    示例响应:
    ```
    {"content": "这是", "done": false}
    {"content": "一个", "done": false}
    {"content": "测试响应", "done": true}
    ```
    """
    try:
        # 创建流式响应
        async def event_generator():
            try:
                async for chunk in message_orchestrator.handle_message(
                    conversation_id=conversation_id,
                    user_id=current_user.id,
                    content=message_in.content,
                    metadata=message_in.metadata,
                ):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                # 记录错误但不暴露详情给客户端
                error_data = {
                    "error": True,
                    "message": "处理消息时发生错误",
                    "done": True,
                }
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
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
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
