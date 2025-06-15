from typing import TypedDict, Literal, List, Union, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

class ChatState(TypedDict):
    """聊天状态数据结构
    
    Attributes:
        messages: 消息历史记录列表
        provider: LLM 提供商 (例如: 'openai', 'anthropic')
        model: 具体模型名称
        session_id: 会话唯一标识
        created_at: 会话创建时间
        updated_at: 最后更新时间
    """
    messages: List[Union[HumanMessage, AIMessage]]
    provider: str
    model: str
    session_id: str
    created_at: datetime
    updated_at: datetime
