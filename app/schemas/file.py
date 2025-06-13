from enum import Enum
from typing import Dict, Optional
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseModelSchema, BaseSchema


class FileStatus(str, Enum):
    """
    文件状态枚举
    """

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"


class FileType(str, Enum):
    """
    文件类型枚举
    """

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"


class UserFileBase(BaseSchema):
    """
    用户文件基础信息
    """

    filename: Optional[str] = None
    original_filename: Optional[str] = None
    file_type: Optional[FileType] = None
    file_size: Optional[int] = None
    status: Optional[FileStatus] = FileStatus.PENDING
    error_message: Optional[str] = None
    file_metadata: Optional[Dict] = None


class UserFileCreate(UserFileBase):
    """
    创建用户文件时的数据格式
    """

    user_id: UUID
    filename: str
    original_filename: str
    file_type: FileType
    file_size: int
    storage_path: str


class UserFileUpdate(UserFileBase):
    """
    更新用户文件时的数据格式
    """

    status: Optional[FileStatus] = None
    error_message: Optional[str] = None
    file_metadata: Optional[Dict] = None


class UserFileInDBBase(UserFileBase, BaseModelSchema):
    """
    数据库中的用户文件信息
    """

    user_id: UUID
    storage_path: str


class UserFileResponse(UserFileInDBBase):
    """
    API 返回的用户文件信息
    """

    pass
