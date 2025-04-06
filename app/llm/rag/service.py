import json
from typing import AsyncGenerator, Dict, List, Optional
from uuid import UUID

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.core.config import settings
from app.core.exceptions import NotFoundException
from app.db.models.message import Message as DBMessage
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.message_repository import MessageRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.repositories.user_file_repository import UserFileRepository
from app.llm.core.base import LLMFactory, Message, StreamingChunk


class RAGService:
    """
    检索增强生成(RAG)服务
    """

    def __init__(
        self,
        message_repo: MessageRepository,
        conversation_repo: ConversationRepository,
        model_repo: ModelConfigRepository,
        file_repo: UserFileRepository,
    ):
        self.message_repo = message_repo
        self.conversation_repo = conversation_repo
        self.model_repo = model_repo
        self.file_repo = file_repo

        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
        )

        # 初始化向量存储
        self.vector_store = None
        if settings.VECTOR_DB_TYPE == "chroma":
            self.vector_store = Chroma(
                persist_directory=settings.CHROMA_DB_DIR,
                embedding_function=self.embeddings,
            )

    async def process_message(
        self, conversation_id: UUID, content: str, metadata: Optional[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        处理用户消息并返回基于检索的AI响应

        参数:
            conversation_id: 会话ID
            content: 用户消息内容
            metadata: 可选的元数据

        返回:
            AI响应的流式生成器
        """
        # 获取会话信息
        conversation = await self.conversation_repo.get_by_id_with_messages(
            conversation_id
        )
        if not conversation:
            raise NotFoundException(detail="会话不存在")

        # 获取模型配置
        model_config = await self.model_repo.get_by_model_id(conversation.model_id)
        if not model_config or not model_config.is_active:
            raise NotFoundException(detail="所选模型不可用")

        # 获取LLM实例
        llm = LLMFactory.create(model_config.provider)

        # 获取历史消息
        history = await self.message_repo.get_conversation_history(conversation_id)

        # 构建基本消息列表
        messages = []

        # 添加系统提示
        system_prompt = conversation.system_prompt
        if not system_prompt:
            system_prompt = "你是一个有帮助的AI助手，会基于提供的文档回答问题。如果你在提供的文档中找不到答案，请说明无法回答，不要编造信息。"

        messages.append(Message(role="system", content=system_prompt))

        # 执行文档检索
        retrieved_docs = []
        try:
            # 检查会话是否有关联文件
            if conversation.files:
                file_ids = [file.id for file in conversation.files]

                # 从向量存储中检索相关内容
                if self.vector_store:
                    # 对用户查询执行相似性搜索
                    search_results = self.vector_store.similarity_search_with_score(
                        query=content,
                        k=5,  # 返回前5个最相关结果
                        filter={
                            "file_id": {"$in": [str(file_id) for file_id in file_ids]}
                        },
                    )

                    # 提取检索到的文档内容
                    for doc, score in search_results:
                        if score < 0.8:  # 只使用相关性较高的文档
                            retrieved_docs.append(doc.page_content)
        except Exception as e:
            # 记录错误但继续，使用无增强的对话作为回退
            logger.error(f"检索失败: {str(e)}")

        # 如果有检索结果，添加到系统提示中
        if retrieved_docs:
            context = "\n\n".join(retrieved_docs)
            context_message = (
                f"以下是与用户问题相关的文档内容，请基于这些内容回答问题:\n\n{context}"
            )
            messages.append(Message(role="system", content=context_message))

        # 添加历史消息
        for msg in history:
            messages.append(Message(role=msg.role, content=msg.content))

        # 添加当前用户消息
        messages.append(Message(role="user", content=content))

        # 生成响应
        model_params = {}
        if model_config.config:
            # 添加模型特定配置
            model_params = model_config.config

        # 获取流式响应
        response_stream = llm.generate_stream(
            messages=messages,
            model_id=model_config.model_id,
            max_tokens=model_config.max_tokens,
            **model_params,
        )

        # 返回响应
        async for chunk in response_stream:
            # 转换为JSON字符串以便于前端解析
            chunk_dict = {"content": chunk.content, "done": chunk.done}

            # 如果出错，添加错误信息
            if chunk.error:
                chunk_dict["error"] = True
                chunk_dict["message"] = chunk.message

            # 添加RAG特定信息
            if chunk.done and retrieved_docs:
                chunk_dict["sources"] = [
                    {"content": doc[:200] + "..."} for doc in retrieved_docs
                ]

            yield json.dumps(chunk_dict)
