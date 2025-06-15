import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from core.message_types import add_messages_liberal
from datetime import datetime


class AgentState(TypedDict):
    """
    AgentState 用于检索流程的状态管理。

    Attributes:
        messages: 消息历史，支持多种消息类型。
        msg_count: 消息计数，用于追踪消息流转次数。
        session_id: 会话唯一标识（可选，便于扩展）。
    """
    messages: Annotated[List[BaseMessage], add_messages_liberal]
    provider: str
    model: str
    session_id: str
    created_at: datetime
    updated_at: datetime

