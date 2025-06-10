from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import Field, validator

from app.schemas.base import BaseModelSchema, BaseSchema


class MessageRole(str, Enum):
    """
    消息角色枚举
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TokenCount(BaseSchema):
    """
    Token 计数
    """

    prompt: int = 0
    completion: int = 0
    total: int = 0


class MessageBase(BaseSchema):
    """
    消息基础信息
    """

    role: MessageRole = MessageRole.USER
    content: str
    metadata: Optional[Dict] = None


class MessageCreate(MessageBase):
    """
    创建消息时的数据格式
    """

    conversation_id: UUID


class MessageUpdate(BaseSchema):
    """
    更新消息时的数据格式
    """

    content: Optional[str] = None
    metadata: Optional[Dict] = None
    tokens: Optional[TokenCount] = None


class MessageInDBBase(MessageBase, BaseModelSchema):
    """
    数据库中的消息信息
    """

    conversation_id: UUID
    tokens: Optional[TokenCount] = None


class MessageResponse(MessageInDBBase):
    """
    API 返回的消息信息
    """

    pass


class MessageCreateRequest(BaseSchema):
    """
    客户端发送新消息的请求格式
    """

    content: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[Dict] = None


# 流式响应相关
class StreamMessageToken(BaseSchema):
    """
    流式消息的单个 token
    """

    token: str
    done: bool = False


class StreamMessageDelta(BaseSchema):
    """
    流式消息的增量部分
    """

    delta: str
    done: bool = False
    metadata: Optional[Dict] = None
