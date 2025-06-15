from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from app.schemas.base import BaseModelSchema, BaseSchema


class UserBase(BaseSchema):
    """
    用户基础信息
    """

    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    default_model_id: Optional[str] = None
    ui_theme: Optional[str] = "light"


class UserCreate(UserBase):
    """
    创建用户时的数据格式
    """

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserUpdate(UserBase):
    """
    更新用户时的数据格式
    """

    password: Optional[str] = Field(None, min_length=8)


class UserInDBBase(UserBase, BaseModelSchema):
    """
    数据库中的用户信息
    """

    is_admin: bool = False


class User(UserInDBBase):
    """
    API 返回的用户信息
    """

    pass


class UserInDB(UserInDBBase):
    """
    数据库中的用户信息（包含哈希密码）
    """

    hashed_password: str
