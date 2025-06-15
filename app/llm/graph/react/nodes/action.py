from langgraph.prebuilt import ToolExecutor, ToolInvocation
from typing import cast
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from core.message_types import LiberalToolMessage



tool_executor = ToolExecutor(tools)


# Define the function to execute tools
async def call_tool(messages):
    actions: list[ToolInvocation] = []
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = cast(AIMessage, messages[-1])
    for tool_call in last_message.tool_calls:
        # We construct a ToolInvocation from the function_call
        actions.append(
            ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"],
            )
        )
    # We call the tool_executor and get back a response
    responses = await tool_executor.abatch(actions)
    # We use the response to create a ToolMessage
    tool_messages = [
        LiberalToolMessage(
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            content=response,
        )
        for tool_call, response in zip(last_message.tool_calls, responses)
    ]
    return tool_messages
