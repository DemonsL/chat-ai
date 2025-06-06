"""
LLM模型配置示例
展示如何配置不同的提供商和模型
"""

from typing import Dict, Any
import os


# 支持的模型配置
MODEL_CONFIGS = {
    # OpenAI模型
    "openai": {
        "gpt-4o": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        },
        "gpt-4": {
            "provider": "openai", 
            "model_id": "gpt-4",
            "max_tokens": 8000,
            "temperature": 0.7,
            "extra_params": {
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "model_id": "gpt-3.5-turbo", 
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        }
    },
    
    # Anthropic模型
    "anthropic": {
        "claude-3-opus-20240229": {
            "provider": "anthropic",
            "model_id": "claude-3-opus-20240229",
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            }
        },
        "claude-3-sonnet-20240229": {
            "provider": "anthropic",
            "model_id": "claude-3-sonnet-20240229",
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            }
        },
        "claude-3-haiku-20240307": {
            "provider": "anthropic",
            "model_id": "claude-3-haiku-20240307",
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            }
        }
    },
    
    # Google模型
    "google-genai": {
        "gemini-1.5-pro": {
            "provider": "google-genai",
            "model_id": "gemini-1.5-pro",
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
            }
        },
        "gemini-1.5-flash": {
            "provider": "google-genai",
            "model_id": "gemini-1.5-flash",
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
            }
        }
    },
    
    # DeepSeek模型
    "deepseek": {
        "deepseek-chat": {
            "provider": "deepseek",
            "model_id": "deepseek-chat",
            "max_tokens": 4000,
            "temperature": 0.7,
            "extra_params": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": "https://api.deepseek.com",
            }
        }
    }
}


def get_model_config(provider: str, model: str) -> Dict[str, Any]:
    """
    获取指定模型的配置
    
    Args:
        provider: 提供商名称
        model: 模型名称
        
    Returns:
        模型配置字典
        
    Raises:
        ValueError: 如果模型不存在
    """
    if provider not in MODEL_CONFIGS:
        raise ValueError(f"不支持的提供商: {provider}")
    
    if model not in MODEL_CONFIGS[provider]:
        raise ValueError(f"提供商 {provider} 不支持模型: {model}")
    
    return MODEL_CONFIGS[provider][model].copy()


def list_available_models() -> Dict[str, list]:
    """
    列出所有可用的模型
    
    Returns:
        按提供商分组的模型列表
    """
    return {
        provider: list(models.keys()) 
        for provider, models in MODEL_CONFIGS.items()
    }


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    验证模型配置是否有效
    
    Args:
        config: 模型配置
        
    Returns:
        是否有效
    """
    required_fields = ["provider", "model_id"]
    return all(field in config for field in required_fields) 