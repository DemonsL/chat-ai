from typing import List
from state import AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from copy import deepcopy
from core.models import get_llm

from datetime import datetime

response_prompt_template = """{instructions}

Respond to the user using ONLY the context provided below. Do not make anything up.

{context}"""

system_message = "You are a helpful AI assistant."  # 可根据实际情况调整

def _get_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    整理历史消息和检索结果，生成系统提示。
    """
    chat_history = []
    for m in messages:
        if isinstance(m, AIMessage):
            if not getattr(m, "tool_calls", None):
                chat_history.append(m)
        if isinstance(m, HumanMessage):
            chat_history.append(m)
    # 假设最后一条消息为检索结果
    response = messages[-1].content
    # 若 response 为文档列表
    if isinstance(response, list) and response and isinstance(response[0], dict) and "page_content" in response[0]:
        content = "\n".join([d["page_content"] for d in response])
    else:
        content = str(response)
    return [
        SystemMessage(
            content=response_prompt_template.format(
                instructions=system_message, context=content
            )
        )
    ] + chat_history

def call_model(state: AgentState):
    """
    调用 LLM 生成最终回复。
    """
    try:
        # 深拷贝状态避免副作用
        new_state = deepcopy(state)
        
        # 参数验证
        if not new_state.get("provider") or not new_state.get("model"):
            raise ValueError("Missing required provider or model configuration")
            
        # 获取LLM实例
        llm = get_llm(new_state["provider"], new_state["model"])
        
        messages = new_state["messages"]
        prompt_messages = _get_messages(messages)
        response = llm.invoke(prompt_messages)
        new_state["messages"].append(response)

        # 更新时间戳
        new_state["updated_at"] = datetime.now()
        
        return new_state
    except Exception as e:
        # 可根据需要记录日志
        raise RuntimeError(f"call_model 节点异常: {e}")
