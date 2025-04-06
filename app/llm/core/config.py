from typing import Dict, Optional

from loguru import logger
from pydantic import BaseModel, Field

from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.session import get_db


class ModelParameters(BaseModel):
    """LLM模型参数配置"""

    # 基本参数
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="生成多样性参数")
    max_tokens: Optional[int] = Field(None, ge=1, description="最大生成的token数")

    # OpenAI特定参数
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="nucleus sampling参数"
    )
    frequency_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="频率惩罚参数"
    )
    presence_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="存在惩罚参数"
    )

    # Anthropic特定参数
    top_k: Optional[int] = Field(None, ge=1, description="top-k筛选参数")

    # 通用高级参数
    timeout: Optional[int] = Field(None, ge=1, description="API超时时间（秒）")
    cache: bool = Field(True, description="是否启用缓存")

    def to_dict(self) -> Dict:
        """将参数转换为字典，过滤掉None值"""
        return {k: v for k, v in self.dict().items() if v is not None}


class LLMProviderConfig(BaseModel):
    """LLM提供商配置"""

    name: str = Field(..., description="提供商名称")
    api_key_env_name: str = Field(..., description="API密钥环境变量名")
    base_url: Optional[str] = Field(None, description="API基础URL")
    organization_id: Optional[str] = Field(None, description="组织ID")
    default_model: str = Field(..., description="默认模型ID")
    default_parameters: ModelParameters = Field(
        default_factory=ModelParameters, description="默认参数设置"
    )


async def init_model_configs():
    """初始化模型配置"""
    logger.info("正在初始化模型配置...")

    # 创建默认模型配置
    default_models = [
        {
            "model_id": "gpt-4o",
            "display_name": "GPT-4o",
            "provider": "openai",
            "capabilities": ["chat", "rag", "agent"],
            "max_tokens": 4096,
            "is_active": True,
            "config": {"temperature": 0.7},
            "api_key_env_name": "OPENAI_API_KEY",
        },
        {
            "model_id": "gpt-3.5-turbo",
            "display_name": "GPT-3.5 Turbo",
            "provider": "openai",
            "capabilities": ["chat", "rag"],
            "max_tokens": 4096,
            "is_active": True,
            "config": {"temperature": 0.7},
            "api_key_env_name": "OPENAI_API_KEY",
        },
        {
            "model_id": "claude-3-opus-20240229",
            "display_name": "Claude 3 Opus",
            "provider": "anthropic",
            "capabilities": ["chat", "rag", "agent"],
            "max_tokens": 4096,
            "is_active": True,
            "config": {"temperature": 0.7},
            "api_key_env_name": "ANTHROPIC_API_KEY",
        },
    ]

    async with get_db() as session:
        model_repo = ModelConfigRepository(session)

        # 检查并添加每个默认模型
        for model_data in default_models:
            existing_model = await model_repo.get_by_model_id(model_data["model_id"])

            if not existing_model:
                logger.info(f"添加默认模型: {model_data['display_name']}")
                await model_repo.create(obj_in=model_data)
            else:
                logger.info(f"模型已存在: {model_data['display_name']}")

    logger.info("模型配置初始化完成")
