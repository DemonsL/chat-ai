from sqlalchemy import Column, Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.models.base import Base


class Message(Base):
    """
    聊天消息模型
    """

    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversation.id"), nullable=False
    )
    role = Column(
        Enum("user", "assistant", "system", name="message_role"), nullable=False
    )
    content = Column(Text, nullable=False)
    tokens = Column(
        JSONB, nullable=True
    )  # 存储Token统计信息 {"prompt": int, "completion": int, "total": int}

    # 辅助数据，可能用于RAG结果存储或Agent执行过程记录
    msg_metadata = Column(JSONB, nullable=True)

    # 关系
    conversation = relationship("Conversation", back_populates="messages")
