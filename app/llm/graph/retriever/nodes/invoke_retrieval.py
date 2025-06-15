from langchain_core.runnables import chain
from typing import Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from uuid import uuid4
from state import AgentState

search_prompt = PromptTemplate.from_template(
    """Given the conversation below, come up with a search query to look up.

This search query can be either a few words or question

Return ONLY this search query, nothing more.

>>> Conversation:
{conversation}
>>> END OF CONVERSATION

Remember, return ONLY the search query that will help you when formulating a response to the above conversation."""
)

@chain
async def get_search_query(messages: Sequence[BaseMessage], llm):
    convo = []
    for m in messages:
        if isinstance(m, AIMessage):
            if "function_call" not in getattr(m, "additional_kwargs", {}):
                convo.append(f"AI: {m.content}")
        if isinstance(m, HumanMessage):
            convo.append(f"Human: {m.content}")
    conversation = "\n".join(convo)
    prompt = await search_prompt.ainvoke({"conversation": conversation})
    response = await llm.ainvoke(prompt, {"tags": ["nostream"]})
    return response

async def invoke_retrieval(state: AgentState, llm):
    """
    检索意图生成节点。
    """
    try:
        messages = state["messages"]
        if len(messages) == 1:
            human_input = messages[-1].content
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": uuid4().hex,
                                "name": "retrieval",
                                "args": {"query": human_input},
                            }
                        ],
                    )
                ],
                "msg_count": state.get("msg_count", 0) + 1
            }
        else:
            search_query = await get_search_query.ainvoke(messages, llm)
            return {
                "messages": [
                    AIMessage(
                        id=getattr(search_query, "id", uuid4().hex),
                        content="",
                        tool_calls=[
                            {
                                "id": uuid4().hex,
                                "name": "retrieval",
                                "args": {"query": getattr(search_query, "content", "")},
                            }
                        ],
                    )
                ],
                "msg_count": state.get("msg_count", 0) + 1
            }
    except Exception as e:
        raise RuntimeError(f"invoke_retrieval 节点异常: {e}")
