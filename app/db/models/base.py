import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import TIMESTAMP, Column, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import as_declarative, declared_attr


@as_declarative()
class Base:
    """
    SQLAlchemy 模型的基类
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __name__: str

    # 根据类名生成表名
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    def dict(self) -> Dict[str, Any]:
        """
        将模型转换为字典
        """
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
