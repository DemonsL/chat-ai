import json
from typing import AsyncGenerator, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.conversation import Conversation
from app.db.models.message import Message
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.repositories.user_file_repository import UserFileRepository
from app.schemas.message import MessageRole
from app.llm.core.manage import LLMOrchestratorService
from app.services.retrieval_service import RetrievalService


class MessageOrchestrator:
    """
    消息协调器，负责处理不同类型的消息并路由到正确的处理服务
    """

    def __init__(self, db_session: AsyncSession):
        """初始化协调器"""
        self.db_session = db_session
        self.conversation_repo = ConversationRepository(db_session)
        self.message_repo = MessageRepository(db_session)
        self.model_repo = ModelConfigRepository(db_session)
        self.file_repo = UserFileRepository(db_session)

        # 创建LLM编排服务（无数据库依赖）
        self.llm_orchestrator = LLMOrchestratorService()
        
        # 创建检索服务
        self.retrieval_service = RetrievalService(
            file_repo=self.file_repo
        )

    async def handle_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        处理新消息

        参数:
            conversation_id: 会话ID
            user_id: 用户ID
            content: 消息内容
            metadata: 可选的元数据

        返回:
            流式响应生成器
        """
        # 检查权限
        conversation = await self.conversation_repo.get_by_id_for_user(
            id=conversation_id, user_id=user_id
        )
        if not conversation:
            raise PermissionDeniedException(detail="没有权限访问此会话或会话不存在")

        # 存储用户消息
        await self._store_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            metadata=metadata,
        )

        try:
            # 获取模型配置
            model_config = await self.model_repo.get_by_model_id(conversation.model_id)
            if not model_config or not model_config.is_active:
                raise NotFoundException(detail="所选模型不可用")

            # 获取历史消息
            history = await self.message_repo.get_conversation_history(conversation_id)
            
            # 转换消息格式
            messages = []
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            
            # 添加当前消息
            messages.append({"role": "user", "content": content})
            
            # 准备模型配置，确保必要的参数存在
            extra_params = model_config.config or {}
            
            # 确保API密钥存在
            if not extra_params.get("api_key") and not extra_params.get("google_api_key"):
                # 根据提供商添加默认的API密钥
                import os
                if model_config.provider == "openai":
                    extra_params["api_key"] = os.getenv("OPENAI_API_KEY")
                elif model_config.provider == "anthropic":
                    extra_params["api_key"] = os.getenv("ANTHROPIC_API_KEY")
                elif model_config.provider == "google-genai":
                    extra_params["google_api_key"] = os.getenv("GOOGLE_API_KEY")
                elif model_config.provider == "deepseek":
                    extra_params["api_key"] = os.getenv("DEEPSEEK_API_KEY")
            
            model_config_dict = {
                "provider": model_config.provider,
                "model_id": model_config.model_id,
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.config.get("temperature") if model_config.config else 0.7,
                "extra_params": extra_params
            }
            
            # 根据会话模式选择处理方式
            if conversation.mode == "chat":
                # 使用基础聊天
                service_stream = self.llm_orchestrator.process_chat(
                    messages=messages,
                    model_config=model_config_dict,
                    system_prompt=conversation.system_prompt,
                )
            elif conversation.mode == "rag":
                # 使用RAG模式
                # 获取会话关联的文件
                file_ids = []
                if conversation.files:
                    file_ids = [file.id for file in conversation.files]
                
                # 检索相关文档
                retrieved_docs = await self.retrieval_service.retrieve_documents(
                    query=content,
                    file_ids=file_ids,
                    top_k=5,
                    similarity_threshold=0.8
                )
                
                service_stream = self.llm_orchestrator.process_rag(
                    messages=messages,
                    model_config=model_config_dict,
                    retrieved_documents=retrieved_docs,
                    system_prompt=conversation.system_prompt,
                )
            elif conversation.mode == "deepresearch":
                # 使用Agent模式
                available_tools = ["search", "analysis", "research"]  # 这里可以根据配置动态获取
                
                service_stream = self.llm_orchestrator.process_agent(
                    messages=messages,
                    model_config=model_config_dict,
                    available_tools=available_tools,
                    system_prompt=conversation.system_prompt,
                )
            else:
                # 默认使用聊天模式
                service_stream = self.llm_orchestrator.process_chat(
                    messages=messages,
                    model_config=model_config_dict,
                    system_prompt=conversation.system_prompt,
                )

            # 用于收集完整的助手回复
            full_response = ""
            full_metadata = {}

            # 转发流式响应
            async for chunk in service_stream:
                # 解析JSON块
                chunk_data = json.loads(chunk)
                yield chunk

                # 累积响应内容
                if not chunk_data.get("error", False):
                    # 如果是工具使用步骤，不累积到最终响应
                    if not chunk_data.get("is_tool_use", False):
                        full_response += chunk_data.get("content", "")

                    # 保存来源引用和其他元数据
                    if "sources" in chunk_data and chunk_data.get("done", False):
                        full_metadata["sources"] = chunk_data["sources"]

            # 存储完整的助手回复
            if full_response:
                await self._store_message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=full_response,
                    metadata=full_metadata or metadata,
                )

        except Exception as e:
            # 记录错误消息
            error_msg = f"处理消息时出错: {str(e)}"
            await self._store_message(
                conversation_id=conversation_id,
                role=MessageRole.SYSTEM,
                content=error_msg,
                metadata={"error": True},
            )
            # 返回错误信息
            yield json.dumps({"content": error_msg, "done": True, "error": True})
            raise

    async def _store_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """存储消息到数据库"""
        message_data = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": metadata,
        }
        return await self.message_repo.create(obj_in=message_data)

    async def get_conversation_messages(
        self, conversation_id: UUID, user_id: UUID, skip: int = 0, limit: int = 50
    ) -> List[Message]:
        """获取会话的消息列表"""
        # 检查用户是否有权限访问该会话
        conversation = await self.conversation_repo.get_by_id_for_user(
            id=conversation_id, user_id=user_id
        )
        if not conversation:
            raise PermissionDeniedException(detail="没有权限访问此会话或会话不存在")

        return await self.message_repo.get_by_conversation_id(
            conversation_id=conversation_id, skip=skip, limit=limit
        )
