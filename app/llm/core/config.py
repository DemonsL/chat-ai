import logging

from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.session import get_db

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
