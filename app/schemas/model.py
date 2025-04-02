from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseModelSchema, BaseSchema


class ModelProvider(str, Enum):
    """
    模型提供商枚举
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OTHER = "other"


class ModelCapability(str, Enum):
    """
    模型能力枚举
    """
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"


class ModelConfigBase(BaseSchema):
    """
    模型配置基础信息
    """
    model_id: Optional[str] = None
    display_name: Optional[str] = None
    provider: Optional[ModelProvider] = None
    capabilities: Optional[List[ModelCapability]] = None
    max_tokens: Optional[int] = None
    is_active: Optional[bool] = True
    config: Optional[Dict] = None
    api_key_env_name: Optional[str] = None


class ModelConfigCreate(ModelConfigBase):
    """
    创建模型配置时的数据格式
    """
    model_id: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=100)
    provider: ModelProvider
    capabilities: List[ModelCapability]
    max_tokens: int = Field(..., gt=0)


class ModelConfigUpdate(ModelConfigBase):
    """
    更新模型配置时的数据格式
    """
    pass


class ModelConfigInDBBase(ModelConfigBase, BaseModelSchema):
    """
    数据库中的模型配置信息
    """
    pass


class ModelConfig(ModelConfigInDBBase):
    """
    API 返回的模型配置信息
    """
    pass


class ModelInfo(BaseSchema):
    """
    客户端使用的模型简要信息
    """
    id: str
    name: str
    provider: ModelProvider
    capabilities: List[ModelCapability]
    max_tokens: int 