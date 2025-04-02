from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Token(BaseModel):
    """
    OAuth2 兼容的令牌模式
    """
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    """
    JWT 令牌的载荷数据
    """
    sub: Optional[UUID] = None
    exp: Optional[int] = None 