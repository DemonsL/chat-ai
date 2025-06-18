import json
from typing import AsyncGenerator, Dict, List, Optional
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.conversation import Conversation
from app.db.models.message import Message
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.repositories.user_file_repository import UserFileRepository
from app.schemas.message import MessageRole
from app.llm.manage import LLMManager

logger = logging.getLogger(__name__)

class MessageService:
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

        # 创建LLM编排服务，传入检索服务
        self.llm_mgr = LLMManager()

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
        
        # 调试：输出对话的基本信息和文件关联情况
        logger.info(f"获取对话 {conversation_id}: 模式={conversation.mode}, 关联文件数={len(conversation.files) if conversation.files else 0}")
        if conversation.files:
            for file in conversation.files:
                logger.info(f"关联文件: {file.id}, 状态={getattr(file, 'status', 'unknown')}, 文件名={getattr(file, 'original_filename', 'unknown')}")

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
            
            # 动态选择处理模式
            processing_mode = "chat"  # 默认模式
            
            # 准备基础元数据，确保所有模式都包含用户ID
            processing_metadata = {
                "user_id": str(user_id),
                "conversation_id": str(conversation_id)
            }
            
            # 1. 检查元数据中是否明确指定了模式
            if metadata and metadata.get("mode"):
                processing_mode = metadata["mode"]
                logger.info(f"元数据指定处理模式: {processing_mode}")
                
                # 即使元数据指定了模式，如果是RAG模式，仍需要获取可用文件
                if processing_mode == "rag":
                    # 获取对话中的可用文件ID
                    file_ids = []
                    
                    # 首先检查元数据是否提供了文件ID
                    if metadata.get("file_ids"):
                        file_ids = [UUID(fid) for fid in metadata["file_ids"]]
                        logger.info(f"从元数据获取文件ID: {file_ids}")
                    # 如果元数据没有提供，从对话关联中获取
                    elif hasattr(conversation, 'files') and conversation.files:
                        indexed_files = [
                            file for file in conversation.files 
                            if hasattr(file, 'status') and file.status == 'indexed'
                        ]
                        file_ids = [file.id for file in indexed_files]
                        logger.info(f"从对话关联获取已索引文件ID: {file_ids}")
                        
                        # 调试：显示所有关联文件的状态
                        for file in conversation.files:
                            logger.info(f"对话关联文件: {file.id}, 状态: {getattr(file, 'status', 'unknown')}")
                    
                    # 更新处理元数据，确保文件信息传递到LangGraph
                    if file_ids:
                        processing_metadata.update({
                            "processing_mode": processing_mode,
                            "available_file_ids": [str(fid) for fid in file_ids],
                            "file_count": len(file_ids)
                        })
                        logger.info(f"RAG模式检测到 {len(file_ids)} 个可用文件")
                    else:
                        logger.warning("RAG模式但没有找到可用文件，将提醒用户上传文档")
                        # 保持RAG模式，但标记没有文件，让统一响应节点处理
                        processing_metadata.update({
                            "processing_mode": processing_mode,
                            "available_file_ids": [],
                            "file_count": 0
                        })
                        
            # 2. 检查是否有可用的文件，如果有则使用RAG模式
            elif (metadata and (metadata.get("files") or metadata.get("file_ids"))) or \
                 (hasattr(conversation, 'files') and conversation.files and len(conversation.files) > 0):
                
                # 获取文件ID列表
                file_ids = []
                logger.info(f"检查文件来源: metadata文件={metadata.get('file_ids') if metadata else None}, 对话文件数={len(conversation.files) if hasattr(conversation, 'files') and conversation.files else 0}")
                
                if metadata and metadata.get("file_ids"):
                    file_ids = [UUID(fid) for fid in metadata["file_ids"]]
                    logger.info(f"从元数据获取文件ID: {file_ids}")
                elif hasattr(conversation, 'files') and conversation.files:
                    indexed_files = [
                        file for file in conversation.files 
                        if hasattr(file, 'status') and file.status == 'indexed'
                    ]
                    file_ids = [file.id for file in indexed_files]
                    logger.info(f"从对话关联获取已索引文件ID: {file_ids}")
                    
                    # 调试：显示所有关联文件的状态
                    for file in conversation.files:
                        logger.info(f"对话关联文件: {file.id}, 状态: {getattr(file, 'status', 'unknown')}")
                
                if file_ids:
                    # 有可用文件，使用RAG模式，让graph来决定是否检索
                    processing_mode = "rag"
                    logger.info(f"检测到 {len(file_ids)} 个可用文件，使用RAG模式")
                    
                    # 添加文件相关的元数据
                    processing_metadata.update({
                        "processing_mode": processing_mode,
                        "available_file_ids": [str(fid) for fid in file_ids],
                        "file_count": len(file_ids)
                    })
                else:
                    logger.info("没有可用的已索引文件，使用聊天模式")
                    processing_mode = "chat"
            # 3. 检查是否需要联网搜索（从元数据或模式）
            elif metadata and (metadata.get("tools") and any(tool in ["search", "web_search"] for tool in metadata.get("tools", []))):
                processing_mode = "search"
            elif metadata and metadata.get("mode") == "search":
                processing_mode = "search"
            # 4. 检查是否需要深度研究（从元数据或模式）
            elif metadata and metadata.get("mode") == "deepresearch":
                processing_mode = "deepresearch"
            # 5. 如果会话有固定模式且不是默认聊天模式，使用会话模式
            elif conversation.mode and conversation.mode != "chat":
                processing_mode = conversation.mode
            
            # 添加处理模式到元数据
            processing_metadata["processing_mode"] = processing_mode
            
            # 准备可用工具（如果是Agent模式）
            available_tools = []
            if processing_mode == "agent":
                available_tools = ["search", "analysis", "research"]  # 默认工具
                # 如果元数据中指定了具体工具，使用指定的工具
                if metadata and metadata.get("tools"):
                    available_tools = metadata["tools"]
            elif processing_mode in ["search", "deepresearch"]:
                # 搜索和深度研究模式不需要传递额外工具，由内部处理
                available_tools = None
            
            # 准备当前消息（不包含历史消息，由checkpointer处理）
            current_messages = [{"role": "user", "content": content}]
            
            # 使用 LangGraph 的新架构处理对话，checkpointer会自动管理历史消息
            service_stream = self.llm_mgr.process_conversation(
                messages=current_messages,  # 只传入当前消息，历史由checkpointer管理
                model_config=model_config_dict,
                mode=processing_mode,
                system_prompt=conversation.system_prompt,
                retrieved_documents=None,
                available_tools=available_tools if processing_mode == "agent" else None,
                metadata=processing_metadata,
                conversation_id=conversation_id,  # 传递对话ID用于checkpointer
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
                    if chunk_data.get("done", False):
                        # 保存来源信息
                        if "sources" in chunk_data:
                            full_metadata["sources"] = chunk_data["sources"]
                        
                        # 保存处理策略相关的元数据
                        if "processing_strategy" in chunk_data:
                            full_metadata["processing_strategy"] = chunk_data["processing_strategy"]
                        
                        # 保存其他graph节点生成的元数据
                        for key in ["processing_type", "question_type", "analysis_completed"]:
                            if key in chunk_data:
                                full_metadata[key] = chunk_data[key]

            # 存储完整的助手回复
            if full_response:
                # 在助手回复的元数据中记录使用的处理模式
                assistant_metadata = full_metadata or {}
                assistant_metadata["processing_mode"] = processing_mode
                
                # 如果有处理元数据（来自检索），添加到助手元数据中
                if processing_metadata:
                    # 合并处理元数据，但避免覆盖已有的重要信息
                    for key, value in processing_metadata.items():
                        if key not in assistant_metadata:
                            assistant_metadata[key] = value
                
                # 添加基础统计信息
                assistant_metadata["processing_stats"] = {
                    "mode": processing_mode,
                    "document_count": 0,
                    "response_length": len(full_response),
                    "has_sources": bool(full_metadata.get("sources"))
                }
                
                await self._store_message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=full_response,
                    metadata=assistant_metadata,
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
            "msg_metadata": metadata,
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
