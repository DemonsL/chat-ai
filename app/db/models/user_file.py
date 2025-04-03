from sqlalchemy import Column, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.models.base import Base


class UserFile(Base):
    """
    用户上传的文件模型
    """

    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # "pdf", "docx", "txt", "image"
    file_size = Column(Integer, nullable=False)  # 文件大小 (bytes)
    storage_path = Column(String(255), nullable=False)  # 存储路径
    status = Column(
        Enum("pending", "processing", "indexed", "error", name="file_status"),
        nullable=False,
        default="pending",
    )
    error_message = Column(Text, nullable=True)

    # 文件处理和索引的元数据
    file_metadata = Column(JSONB, nullable=True)  # 可能包含页数、OCR信息、索引记录等

    # 关系
    user = relationship("User", backref="files")
