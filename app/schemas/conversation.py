from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseModelSchema, BaseSchema


class ConversationMode(str, Enum):
    """
    会话模式枚举
    """

    CHAT = "chat"
    RAG = "rag"
    DEEPRESEARCH = "deepresearch"


class ConversationBase(BaseSchema):
    """
    会话基础信息
    """

    title: Optional[str] = None
    model_id: Optional[str] = None
    mode: Optional[ConversationMode] = ConversationMode.CHAT
    system_prompt: Optional[str] = None


class ConversationCreate(ConversationBase):
    """
    创建会话时的数据格式
    """

    title: str = Field(..., min_length=1, max_length=255)
    model_id: str
    file_ids: Optional[List[UUID]] = None


class ConversationUpdate(ConversationBase):
    """
    更新会话时的数据格式
    """

    file_ids: Optional[List[UUID]] = None


class ConversationInDBBase(ConversationBase, BaseModelSchema):
    """
    数据库中的会话信息
    """

    user_id: UUID


class Conversation(ConversationInDBBase):
    """
    API 返回的会话信息
    """

    pass


class ConversationWithDetails(Conversation):
    """
    带文件信息的会话
    """

    file_ids: Optional[List[UUID]] = None
