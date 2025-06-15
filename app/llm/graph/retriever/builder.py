from langchain_core.language_models.base import LanguageModelLike
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph
from graph.retriever.state import AgentState
from graph.retriever.nodes.call_model import call_model
from graph.retriever.nodes.invoke_retrieval import invoke_retrieval
from graph.retriever.nodes.retrieve import retrieve

from core.checkpoint import get_checkpoint


def buikd_retrieval_graph():
    """
    构建检索流程执行器
    """
    graph = StateGraph(AgentState)
    graph.add_node("invoke_retrieval", invoke_retrieval)
    graph.add_node("retrieve", retrieve)
    graph.add_node("response", call_model)
    graph.set_entry_point("invoke_retrieval")
    graph.add_edge("invoke_retrieval", "retrieve")
    graph.add_edge("retrieve", "response")
    graph.set_finish_point("response")

    # 获取checkpoint配置
    checkpoint = get_checkpoint()
    return graph.compile(checkpointer=checkpoint)
