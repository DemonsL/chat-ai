import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.tools.base import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.exceptions import LLMAPIException, NotFoundException
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.llm.core.base import Message


class AgentCallbackHandler(BaseCallbackHandler):
    """Agent回调处理器，用于将Agent执行过程转换为流式输出"""

    def __init__(self):
        self.tokens = []
        self.done = False
        self.intermediate_steps = []

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """工具开始执行时的回调"""
        tool_name = serialized.get("name", "unknown_tool")
        self.intermediate_steps.append(f"🔧 使用工具: {tool_name}\n输入: {input_str}\n")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具执行结束时的回调"""
        self.intermediate_steps.append(f"🔧 工具输出: {output}\n\n")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """LLM开始生成时的回调"""
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """LLM生成新token时的回调"""
        self.tokens.append(token)

    def on_llm_end(self, response, **kwargs) -> None:
        """LLM生成结束时的回调"""
        self.done = True

    def on_agent_action(self, action, **kwargs) -> None:
        """Agent执行动作时的回调"""
        self.intermediate_steps.append(f"🤖 思考: 我需要使用 {action.tool} 工具\n\n")

    def on_agent_finish(self, finish, **kwargs) -> None:
        """Agent结束时的回调"""
        self.intermediate_steps.append(
            f"✅ 完成: {finish.return_values.get('output')}\n"
        )
        self.done = True


class AgentService:
    """
    Agent服务，提供深度研究能力
    """

    def __init__(
        self,
        message_repo: MessageRepository,
        conversation_repo: ConversationRepository,
        model_repo: ModelConfigRepository,
    ):
        self.message_repo = message_repo
        self.conversation_repo = conversation_repo
        self.model_repo = model_repo

    async def process_message(
        self, conversation_id: UUID, content: str, metadata: Optional[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        处理用户消息并执行深度研究

        参数:
            conversation_id: 会话ID
            content: 用户消息内容
            metadata: 可选的元数据

        返回:
            执行结果的流式生成器
        """
        # 获取会话信息
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")

        # 获取模型配置
        model_config = await self.model_repo.get_by_model_id(conversation.model_id)
        if not model_config or not model_config.is_active:
            raise NotFoundException(detail="所选模型不可用")

        # 检查模型是否支持Agent能力
        if "agent" not in model_config.capabilities:
            raise LLMAPIException(detail="所选模型不支持深度研究功能")

        # 获取历史消息
        history = await self.message_repo.get_conversation_history(conversation_id)

        # 创建LLM
        llm = ChatOpenAI(
            model=model_config.model_id,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7,
            streaming=True,
        )

        # 创建工具
        tools = self._get_tools()

        # 创建回调处理器
        callback_handler = AgentCallbackHandler()

        # 创建记忆
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # 添加历史消息到记忆
        for msg in history:
            role = "human" if msg.role == "user" else "ai"
            memory.chat_memory.add_message({"role": role, "content": msg.content})

        # 创建Agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
        )

        # 创建系统消息
        system_message = (
            conversation.system_prompt
            or "你是一个有帮助的AI助手，可以使用各种工具来回答问题和解决问题。"
        )

        # 启动Agent执行
        try:
            # 使用异步运行
            # 注意：LangChain目前的Agent执行不是完全异步的，这是一个简化处理
            # 为了真正的流式处理，可能需要自定义实现
            agent.run(
                input={
                    "input": content,
                    "chat_history": memory.chat_memory.messages,
                },
                callbacks=[callback_handler],
            )

            # 生成流式响应
            while (
                not callback_handler.done
                or callback_handler.tokens
                or callback_handler.intermediate_steps
            ):
                # 输出中间步骤
                if callback_handler.intermediate_steps:
                    step = callback_handler.intermediate_steps.pop(0)
                    yield json.dumps(
                        {"content": step, "done": False, "is_tool_use": True}
                    )

                # 输出LLM token
                if callback_handler.tokens:
                    token = callback_handler.tokens.pop(0)
                    yield json.dumps({"content": token, "done": False})

                # 如果没有更多内容但还没完成，等待一下
                if (
                    not callback_handler.intermediate_steps
                    and not callback_handler.tokens
                    and not callback_handler.done
                ):
                    import asyncio

                    await asyncio.sleep(0.1)

            # 发送完成标记
            yield json.dumps({"content": "", "done": True})

        except Exception as e:
            yield json.dumps(
                {
                    "content": f"执行深度研究时出错: {str(e)}",
                    "done": True,
                    "error": True,
                }
            )

    def _get_tools(self) -> List[BaseTool]:
        """获取Agent可用的工具"""

        @tool("search", return_direct=False)
        def search(query: str) -> str:
            """搜索互联网以查找有关特定主题的信息。"""
            # 这是一个模拟实现，实际应使用真实的搜索API
            return f"模拟搜索结果: {query}"

        @tool("calculator", return_direct=False)
        def calculator(expression: str) -> str:
            """进行数学计算。"""
            try:
                return str(eval(expression))
            except Exception as e:
                return f"计算错误: {str(e)}"

        # 返回工具列表
        return [search, calculator]
