import pytest
from datetime import datetime
from copy import deepcopy

from state import ChatState
from nodes.chat import chatbot
from builder import build_chat_graph, ChatGraphConfig

def create_test_state() -> ChatState:
    """创建测试用的聊天状态"""
    return ChatState(
        messages=[],
        provider="test_provider",
        model="test_model",
        session_id="test_session",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

def test_chatbot_basic():
    """测试基本的chatbot功能"""
    initial_state = create_test_state()
    result_state = chatbot(initial_state)
    
    assert result_state is not initial_state  # 确保返回新的状态对象
    assert len(result_state["messages"]) > len(initial_state["messages"])
    assert result_state["updated_at"] > initial_state["updated_at"]

def test_chatbot_error_handling():
    """测试错误处理"""
    invalid_state = ChatState(messages=[], provider="", model="")
    
    with pytest.raises(ValueError):
        chatbot(invalid_state)

def test_build_chat_graph():
    """测试图构建"""
    graph = build_chat_graph()
    assert graph is not None
    
    # 测试自定义配置
    config = ChatGraphConfig(
        checkpoint_config={"path": "test_path"},
        custom_nodes={}
    )
    custom_graph = build_chat_graph(config)
    assert custom_graph is not None 