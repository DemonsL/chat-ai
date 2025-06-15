from typing import Optional
from copy import deepcopy
import logging
from datetime import datetime

from langchain_core.language_models.base import LanguageModelLike
from langchain_core.messages import SystemMessage
from langchain.schema.output_parser import OutputParserException

from state import ChatState
from core.models import get_llm

logger = logging.getLogger(__name__)

def chatbot(state: ChatState) -> ChatState:
    """处理聊天请求并生成回复
    
    Args:
        state: 当前聊天状态
        
    Returns:
        更新后的聊天状态
        
    Raises:
        ValueError: 当必要参数缺失时
        RuntimeError: 当LLM调用失败时
    """
    try:
        # 深拷贝状态避免副作用
        new_state = deepcopy(state)
        
        # 参数验证
        if not new_state.get("provider") or not new_state.get("model"):
            raise ValueError("Missing required provider or model configuration")
            
        # 获取LLM实例
        llm = get_llm(new_state["provider"], new_state["model"])
        
        # 调用LLM生成回复
        try:
            response = llm.invoke(new_state["messages"])
            new_state["messages"].append(response)
        except Exception as e:
            logger.error(f"LLM invocation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
            
        # 更新时间戳
        new_state["updated_at"] = datetime.now()
        
        return new_state
        
    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
        raise 