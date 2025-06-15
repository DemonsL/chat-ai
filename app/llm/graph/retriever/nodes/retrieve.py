from state import AgentState
from core.message_types import LiberalToolMessage

async def retrieve(state: AgentState, retriever):
    """
    检索执行节点。
    """
    try:
        messages = state["messages"]
        params = messages[-1].tool_calls[0]
        query = params["args"]["query"]
        response = await retriever.ainvoke(query)
        # 检查检索结果
        if not response:
            content = []
        else:
            content = [doc.model_dump() for doc in response]
        msg = LiberalToolMessage(
            name="retrieval", content=content, tool_call_id=params["id"]
        )
        return {
            "messages": [msg],
            "msg_count": state.get("msg_count", 0) + 1
        }
    except Exception as e:
        raise RuntimeError(f"retrieve 节点异常: {e}")
