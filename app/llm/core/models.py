import os
from typing import List, Dict, Any, Optional, TypeVar, Type
from dotenv import load_dotenv

import json
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# LangChain 核心导入
from langchain_core.language_models import BaseChatModel
# 导入各供应商模型集成
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq  # 用于访问 Grok 模型
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import QianwenChatEndpoint  # 用于千问模型


# 每个供应商支持的模型列表
AVAILABLE_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
    "grok": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "qianwen": [
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
    ],
}

# Default configurations for each model provider
# These can be overridden by environment variables or passed parameters
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model": "gpt-4o",
        "temperature": 0.7,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "max_tokens": 1000,
    },
    "anthropic": {
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.7,
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "max_tokens": 1000,
    },
    "gemini": {
        "model": "gemini-1.5-pro",
        "temperature": 0.7,
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "max_output_tokens": 1000,
    },
    "grok": {
        "model": "llama-3-70b-8192",
        "temperature": 0.7,
        "api_key": os.getenv("GROQ_API_KEY"),
        "max_tokens": 1000,
    },
    "deepseek": {
        "model_name": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 1000,
    },
    "qianwen": {
        "model": "qwen-max",
        "api_key": os.getenv("QIANWEN_API_KEY"),
        "dashscope_api_key": os.getenv("DASHSCOPE_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 1000,
    },
}


# 供应商的默认模型
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-sonnet-20240229",
    "gemini": "gemini-1.5-pro",
    "grok": "llama-3-70b-8192",
    "deepseek": "deepseek-chat",
    "qianwen": "qwen-max",
}

# 模型特定的参数重写
MODEL_SPECIFIC_PARAMS = {
    # DeepSeek模型需要使用model_name而不是model参数
    "deepseek-chat": {"param_name": "model_name"},
    "deepseek-coder": {"param_name": "model_name"},
    "deepseek-llm-67b": {"param_name": "model_name"},
}

def list_available_models(provider: str = None) -> Dict[str, List[str]]:
    """
    列出可用的模型
    
    Args:
        provider: 如果提供，只返回该供应商的模型；否则返回所有供应商的模型
        
    Returns:
        供应商及其可用模型的字典
    """
    if provider:
        if provider not in AVAILABLE_MODELS:
            raise ValueError(f"不支持的供应商: {provider}")
        return {provider: AVAILABLE_MODELS[provider]}
    
    return AVAILABLE_MODELS

def get_available_model_combinations() -> List[str]:
    """
    获取所有可用的供应商/模型组合
    
    Returns:
        包含所有可用模型组合的列表，格式为 ["provider/model", ...]
    """
    combinations = []
    
    for provider, models in AVAILABLE_MODELS.items():
        for model in models:
            combinations.append(f"{provider}/{model}")
    
    return combinations

def get_provider_and_model(model_string: str) -> Tuple[str, str]:
    """
    从组合字符串解析供应商和模型名称
    
    Args:
        model_string: 格式为 "provider/model" 或仅 "provider"
        
    Returns:
        (provider, model_name) 元组，如果没有指定模型则model_name为None
    """
    if "/" in model_string:
        provider, model_name = model_string.split("/", 1)
        if provider not in AVAILABLE_MODELS:
            raise ValueError(f"不支持的供应商: {provider}")
        if model_name not in AVAILABLE_MODELS[provider]:
            raise ValueError(f"供应商{provider}不支持模型: {model_name}")
        return provider, model_name
    else:
        if model_string not in AVAILABLE_MODELS:
            raise ValueError(f"不支持的供应商: {model_string}")
        return model_string, None

def get_model_config(provider: str, model_name: str = None) -> Dict[str, Any]:
    """
    获取特定供应商和模型的配置
    
    Args:
        provider: 供应商名称
        model_name: 模型名称，如果为None则使用默认模型
        
    Returns:
        模型配置字典
    """
    if provider not in MODEL_CONFIG:
        raise ValueError(f"不支持的供应商: {provider}")
        
    # 如果没有指定模型，则使用默认模型
    if model_name is None:
        model_name = DEFAULT_MODELS[provider]
    elif model_name not in AVAILABLE_MODELS[provider]:
        raise ValueError(f"供应商{provider}不支持模型: {model_name}")
    
    # 复制模板配置
    config = MODEL_CONFIG[provider].copy()
    
    # 添加模型名称
    # 某些模型可能需要特殊处理
    if model_name in MODEL_SPECIFIC_PARAMS:
        param_name = MODEL_SPECIFIC_PARAMS[model_name].get("param_name", "model")
        config[param_name] = model_name
    else:
        config["model"] = model_name
        
    return config


def get_model(model_name: str, **kwargs) -> BaseChatModel:
    """
    Get a LangChain chat model instance based on provider name.
    
    Args:
        model_name: Name of the model provider
        **kwargs: Additional parameters to pass to the model constructor
    
    Returns:
        A LangChain chat model instance
    
    Raises:
        ValueError: If the model provider is not supported
        Exception: If there's an error initializing the model
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model provider: {model_name}")
    
    config = MODEL_CONFIG[model_name].copy()
    
    # Override config with any provided kwargs
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    
    try:
        if model_name == "openai":
            return ChatOpenAI(**config)
        elif model_name == "anthropic":
            return ChatAnthropic(**config)
        elif model_name == "gemini":
            return ChatGoogleGenerativeAI(**config)
        elif model_name == "grok":
            return ChatGroq(**config)
        elif model_name == "deepseek":
            return ChatDeepSeek(**config)
        elif model_name == "qianwen":
            return QianwenChatEndpoint(**config)
        else:
            raise ValueError(f"Model implementation not found for: {model_name}")
    except Exception as e:
        raise Exception(f"Error initializing {model_name} model: {str(e)}")


T = TypeVar('T', bound=BaseModel)

def call_model(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both Deepseek and non-Deepseek models.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    llm = get_model(model_name)
    
    # For non-JSON support models, we can use structured output
    llm = llm.with_structured_output(
        pydantic_model,
        method="json_mode",
    )
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            
            # For non-JSON support models, we need to extract and parse the JSON manually
            parsed_result = extract_json_from_deepseek_response(result.content)
            if parsed_result:
                return pydantic_model(**parsed_result)
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None
