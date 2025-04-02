from sqlalchemy import Column, Enum, ForeignKey, String, Table, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship

from app.db.models.base import Base


# 会话与文件的多对多关系映射表
conversation_files = Table(
    "conversation_files",
    Base.metadata,
    Column("conversation_id", UUID(as_uuid=True), ForeignKey("conversation.id"), primary_key=True),
    Column("file_id", UUID(as_uuid=True), ForeignKey("userfile.id"), primary_key=True),
)


class Conversation(Base):
    """
    对话会话模型
    """
    title = Column(String(255), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    model_id = Column(String, nullable=False)  # 使用的模型ID
    mode = Column(
        Enum("chat", "rag", "deepresearch", name="conversation_mode"),
        nullable=False,
        default="chat"
    )
    system_prompt = Column(Text, nullable=True)  # 可选的自定义系统提示
    
    # 关系
    user = relationship("User", backref="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    files = relationship(
        "UserFile",
        secondary=conversation_files,
        backref="conversations"
    ) 