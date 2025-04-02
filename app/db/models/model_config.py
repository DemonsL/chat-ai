from sqlalchemy import Boolean, Column, Enum, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from app.db.models.base import Base


class ModelConfig(Base):
    """
    LLM 模型配置
    """
    model_id = Column(String, unique=True, index=True, nullable=False)
    display_name = Column(String, nullable=False)
    provider = Column(
        Enum("openai", "anthropic", "deepseek", "other", name="model_provider"),
        nullable=False
    )
    capabilities = Column(JSONB, nullable=False)  # ["chat", "rag", "agent"]
    max_tokens = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # 模型特定配置
    config = Column(JSONB, nullable=True)  # 可能包含温度、top_p等模型特定配置
    
    # API 密钥引用 (不存储实际密钥，而是引用环境变量或密钥管理服务)
    api_key_env_name = Column(String, nullable=True) 