from core.message_types import LiberalToolMessage
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import create_react_agent


async def _get_messages(messages):
    msgs = []
    for m in messages:
        if isinstance(m, LiberalToolMessage):
            _dict = m.model_dump()
            _dict["content"] = str(_dict["content"])
            m_c = ToolMessage(**_dict)
            msgs.append(m_c)
        elif isinstance(m, FunctionMessage):
            # anthropic doesn't like function messages
            msgs.append(HumanMessage(content=str(m.content)))
        else:
            msgs.append(m)

    return [SystemMessage(content=system_message)] + msgs

if tools:
    llm_with_tools = llm.bind_tools(tools)
else:
    llm_with_tools = llm
agent = _get_messages | llm_with_tools