import json
from typing import AsyncGenerator, Dict, List, Optional, Any
from uuid import UUID
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from app.llm.core.base import (
    StreamingChunk,
    create_chat_model,
    convert_messages_to_langchain,
    stream_chat_model_response
)


class ConversationMode(str, Enum):
    """会话模式枚举"""
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"


class LLMManager:
    """
    LLM编排服务
    专门处理LLM相关的功能，不涉及数据库和业务逻辑
    使用LangChain标准接口
    """
    
    def __init__(self):
        self._model_cache = {}  # 缓存已创建的模型实例
    
    def _get_or_create_model(self, model_config: Dict[str, Any]) -> BaseChatModel:
        """获取或创建模型实例（带缓存）"""
        cache_key = f"{model_config['provider']}-{model_config['model_id']}"
        
        if cache_key not in self._model_cache:
            try:
                # 准备模型参数
                model_params = {
                    "temperature": model_config.get("temperature", 0.7),
                    "max_tokens": model_config.get("max_tokens"),
                }
                
                # 添加额外参数
                if model_config.get("extra_params"):
                    model_params.update(model_config["extra_params"])
                
                # 过滤None值
                model_params = {k: v for k, v in model_params.items() if v is not None}
                
                # 创建模型
                self._model_cache[cache_key] = create_chat_model(
                    provider=model_config["provider"],
                    model=model_config["model_id"],
                    **model_params
                )
            except Exception as e:
                raise ValueError(f"创建模型失败 {model_config['provider']}/{model_config['model_id']}: {str(e)}")
        
        return self._model_cache[cache_key]
    
    def clear_model_cache(self):
        """清除模型缓存"""
        self._model_cache.clear()
    
    def get_cached_models(self) -> List[str]:
        """获取已缓存的模型列表"""
        return list(self._model_cache.keys())
    
    async def process_chat(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        处理基础聊天请求
        
        Args:
            messages: 消息历史 [{"role": "user", "content": "..."}]
            model_config: 模型配置
            system_prompt: 系统提示
        """
        # 获取模型实例
        model = self._get_or_create_model(model_config)
        
        # 构建消息列表
        final_messages = []
        
        # 添加系统提示
        if system_prompt:
            final_messages.append(SystemMessage(content=system_prompt))
        
        # 转换并添加历史消息
        langchain_messages = convert_messages_to_langchain(messages)
        final_messages.extend(langchain_messages)
        
        # 生成流式响应
        async for chunk in stream_chat_model_response(model, final_messages):
            yield json.dumps({
                "content": chunk.content,
                "done": chunk.done,
                "error": chunk.error,
                "message": chunk.message,
                "metadata": chunk.metadata
            })
    
    async def process_rag(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        retrieved_documents: List[str],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        处理RAG请求
        
        Args:
            messages: 消息历史
            model_config: 模型配置
            retrieved_documents: 检索到的文档
            system_prompt: 系统提示
        """
        # 获取模型实例
        model = self._get_or_create_model(model_config)
        
        # 构建消息列表
        final_messages = []
        
        # 添加系统提示
        if not system_prompt:
            system_prompt = "你是一个有帮助的AI助手，会基于提供的文档回答问题。如果你在提供的文档中找不到答案，请说明无法回答，不要编造信息。"
        final_messages.append(SystemMessage(content=system_prompt))
        
        # 添加检索到的文档作为上下文
        if retrieved_documents:
            context = "\n\n".join(retrieved_documents)
            context_message = f"以下是与用户问题相关的文档内容，请基于这些内容回答问题:\n\n{context}"
            final_messages.append(SystemMessage(content=context_message))
        
        # 转换并添加历史消息
        langchain_messages = convert_messages_to_langchain(messages)
        final_messages.extend(langchain_messages)
        
        # 生成流式响应
        async for chunk in stream_chat_model_response(model, final_messages):
            chunk_dict = {
                "content": chunk.content,
                "done": chunk.done,
                "error": chunk.error,
                "message": chunk.message,
                "metadata": chunk.metadata
            }
            
            # 在最后一个chunk中添加来源信息
            if chunk.done and retrieved_documents:
                chunk_dict["sources"] = [
                    {"content": doc[:200] + "..." if len(doc) > 200 else doc} 
                    for doc in retrieved_documents
                ]
            
            yield json.dumps(chunk_dict)
    
    async def process_agent(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        available_tools: List[str],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        处理Agent请求
        
        Args:
            messages: 消息历史
            model_config: 模型配置
            available_tools: 可用工具列表
            system_prompt: 系统提示
        """
        # 获取模型实例
        model = self._get_or_create_model(model_config)
        
        # 构建消息列表
        final_messages = []
        
        # 添加系统提示（包含工具信息）
        if not system_prompt:
            system_prompt = "你是一个智能助手，可以使用以下工具来帮助用户："
        
        if available_tools:
            tool_info = f"\n可用工具: {', '.join(available_tools)}"
            system_prompt += tool_info
        
        final_messages.append(SystemMessage(content=system_prompt))
        
        # 转换并添加历史消息
        langchain_messages = convert_messages_to_langchain(messages)
        final_messages.extend(langchain_messages)
        
        # 生成流式响应
        async for chunk in stream_chat_model_response(model, final_messages):
            chunk_dict = {
                "content": chunk.content,
                "done": chunk.done,
                "error": chunk.error,
                "message": chunk.message,
                "metadata": chunk.metadata,
                "is_tool_use": False  # Agent模式可能涉及工具使用
            }
            
            yield json.dumps(chunk_dict)
    
    async def estimate_tokens(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        估算token使用量
        
        Args:
            messages: 消息历史
            model_config: 模型配置
            system_prompt: 系统提示
            
        Returns:
            token使用情况
        """
        # 获取模型实例
        model = self._get_or_create_model(model_config)
        
        # 构建消息列表
        final_messages = []
        if system_prompt:
            final_messages.append(SystemMessage(content=system_prompt))
        
        # 转换并添加历史消息
        langchain_messages = convert_messages_to_langchain(messages)
        final_messages.extend(langchain_messages)
        
        # 使用LangChain的token计算功能
        try:
            # 大多数LangChain模型都有get_num_tokens方法
            if hasattr(model, 'get_num_tokens_from_messages'):
                token_count = model.get_num_tokens_from_messages(final_messages)
            elif hasattr(model, 'get_num_tokens'):
                # 如果没有专门的消息token计算方法，尝试转换为文本
                text = "\n".join([msg.content for msg in final_messages])
                token_count = model.get_num_tokens(text)
            else:
                # 简单估算：每4个字符约1个token
                text = "\n".join([msg.content for msg in final_messages])
                token_count = len(text) // 4
        except Exception:
            # 回退到简单估算
            text = "\n".join([msg.content for msg in final_messages])
            token_count = len(text) // 4
        
        return {
            "prompt_tokens": token_count,
            "max_tokens": model_config.get("max_tokens", 4000),
            "available_tokens": max(0, model_config.get("max_tokens", 4000) - token_count),
        } 