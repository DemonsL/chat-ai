from typing import Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph

from core.message_types import add_messages_liberal
from core.checkpoint import get_checkpoint
from nodes.chat import chatbot
from state import ChatState


def build_chat_graph() -> Runnable:
    """构建聊天图"""
    # 初始化图
    graph = StateGraph(ChatState)
    
    # 添加基础聊天节点
    graph.add_node("chat", Runnable(chatbot))
    
    # 设置入口和出口
    graph.set_entry_point("chat")
    graph.set_finish_point("chat")
    
    # 获取checkpoint配置
    checkpoint = get_checkpoint()
    
    return graph.compile(checkpoint)
