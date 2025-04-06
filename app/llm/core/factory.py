from typing import Dict, List, Optional

from app.core.config import settings
from app.llm.core.base import BaseLLM
from app.llm.core.config import LLMProviderConfig, ModelParameters
from app.llm.core.langchain_llm import LangChainLLM


class EnhancedLLMFactory:
    """增强的LLM工厂类，支持从配置创建LLM实例"""

    # 存储供应商配置
    _provider_configs: Dict[str, LLMProviderConfig] = {}

    @classmethod
    def register_provider(cls, config: LLMProviderConfig) -> None:
        """注册LLM提供商配置"""
        cls._provider_configs[config.name] = config

    @classmethod
    def get_registered_providers(cls) -> List[str]:
        """获取已注册的提供商列表"""
        return list(cls._provider_configs.keys())

    @classmethod
    def create(
        cls,
        provider: str,
        model_id: Optional[str] = None,
        parameters: Optional[ModelParameters] = None,
    ) -> BaseLLM:
        """
        创建LLM实例

        参数:
            provider: 提供商名称
            model_id: 模型ID，不指定则使用默认模型
            parameters: 模型参数，不指定则使用默认参数

        返回:
            LLM实例
        """
        # 获取提供商配置
        if provider not in cls._provider_configs:
            raise ValueError(f"未注册的LLM提供商: {provider}")

        config = cls._provider_configs[provider]

        # 获取API密钥
        api_key = getattr(settings, config.api_key_env_name, None)
        if not api_key:
            raise ValueError(f"未设置API密钥: {config.api_key_env_name}")

        # 合并参数
        kwargs = {}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.organization_id:
            kwargs["organization_id"] = config.organization_id

        # 添加模型参数
        model_params = parameters or config.default_parameters
        kwargs.update(model_params.to_dict())

        # 创建LLM实例
        return LangChainLLM(
            provider=provider,
            api_key=api_key,
            model_name=model_id or config.default_model,
            **kwargs,
        )


# 注册默认提供商
def register_default_providers():
    """注册默认的LLM提供商配置"""
    providers = [
        LLMProviderConfig(
            name="openai",
            api_key_env_name="OPENAI_API_KEY",
            default_model="gpt-4o",
            default_parameters=ModelParameters(temperature=0.7, max_tokens=4096),
        ),
        LLMProviderConfig(
            name="anthropic",
            api_key_env_name="ANTHROPIC_API_KEY",
            default_model="claude-3-opus-20240229",
            default_parameters=ModelParameters(temperature=0.7, max_tokens=4096),
        ),
        LLMProviderConfig(
            name="deepseek",
            api_key_env_name="DEEPSEEK_API_KEY",
            default_model="deepseek-chat",
            default_parameters=ModelParameters(temperature=0.7, max_tokens=2048),
        ),
    ]

    for config in providers:
        EnhancedLLMFactory.register_provider(config)
