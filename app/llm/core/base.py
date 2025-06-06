from typing import AsyncGenerator, Dict, List, Optional, Any
from pydantic import BaseModel

# LangChain标准导入
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model


class StreamingChunk(BaseModel):
    """流式响应块"""
    content: str
    done: bool = False
    error: bool = False
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def create_chat_model(provider: str, model: str, **kwargs) -> BaseChatModel:
    """
    使用LangChain标准方法创建聊天模型
    
    Args:
        provider: 提供商名称 (openai, anthropic, google-genai等)
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        LangChain BaseChatModel实例
    """
    return init_chat_model(
        model=model,
        model_provider=provider,
        **kwargs
    )


def convert_messages_to_langchain(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    将字典格式的消息转换为LangChain消息格式
    
    Args:
        messages: 字典格式消息列表 [{"role": "user", "content": "..."}]
        
    Returns:
        LangChain BaseMessage列表
    """
    langchain_messages = []
    
    for msg in messages:
        role = msg["role"].lower()
        content = msg["content"]
        
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role in ["assistant", "ai"]:
            langchain_messages.append(AIMessage(content=content))
        else:
            # 默认作为用户消息处理
            langchain_messages.append(HumanMessage(content=content))
    
    return langchain_messages


async def stream_chat_model_response(
    model: BaseChatModel,
    messages: List[BaseMessage]
) -> AsyncGenerator[StreamingChunk, None]:
    """
    流式调用聊天模型
    
    Args:
        model: LangChain聊天模型
        messages: 消息列表
        
    Yields:
        StreamingChunk: 流式响应块
    """
    chunk_count = 0
    full_content = ""
    
    try:
        async for chunk in model.astream(messages):
            chunk_count += 1
            content = ""
            
            # 处理不同类型的chunk
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                full_content += content
            
            yield StreamingChunk(
                content=content,
                done=False,
                error=False,
                metadata={"chunk_count": chunk_count}
            )
        
        # 发送结束标记
        yield StreamingChunk(
            content="",
            done=True,
            error=False,
            metadata={
                "total_chunks": chunk_count,
                "total_content_length": len(full_content)
            }
        )
        
    except Exception as e:
        # 记录错误详情
        error_msg = f"模型流式响应错误: {str(e)}"
        
        yield StreamingChunk(
            content="",
            done=True,
            error=True,
            message=error_msg,
            metadata={"chunk_count": chunk_count}
        ) 