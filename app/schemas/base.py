from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """
    所有模式类的基类
    """

    model_config = ConfigDict(from_attributes=True)


class IDSchema(BaseSchema):
    """
    带有 ID 的模式类
    """

    id: UUID


class TimestampSchema(BaseSchema):
    """
    带有时间戳的模式类
    """

    created_at: datetime
    updated_at: Optional[datetime] = None


class BaseModelSchema(IDSchema, TimestampSchema):
    """
    基础模型模式类，包含 ID 和时间戳
    """

    pass
