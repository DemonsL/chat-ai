import json
from typing import AsyncGenerator, Dict, List, Optional, Any, Annotated, Literal, Tuple
from uuid import UUID
from enum import Enum
from loguru import logger

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.llm.core.base import (
    StreamingChunk,
    create_chat_model,
    convert_messages_to_langchain,
    stream_chat_model_response
)
from app.llm.core.prompts import prompt_manager
from app.llm.core.checkpointer import get_checkpointer, get_conversation_config
from app.llm.rag.retrieval_service import LLMRetrievalService
from app.llm.rag.file_processor import LLMFileProcessor


from app.core.exceptions import (FileProcessingException,
                                 InvalidFileTypeException)
from langchain_core.documents import Document


class ConversationMode(str, Enum):
    """会话模式枚举"""
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"
    SEARCH = "search"
    DEEPRESEARCH = "deepresearch"


class DocQARouterResult(BaseModel):
    """DocQA路由器结构化输出模型"""
    question_category: Literal["document_related", "non_document"] = Field(
        description="问题大类：document_related(文档相关)、non_document(非文档相关)"
    )
    analysis_type: Literal["full_document", "keyword_search", "general_chat"] = Field(
        description="具体分析类型：full_document(全文档分析)、keyword_search(关键词检索)、general_chat(一般对话)"
    )
    reasoning: str = Field(
        description="分析推理过程和依据"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="分析结果的置信度，范围0.0-1.0"
    )


class ConversationState(TypedDict):
    """LangGraph 对话状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]
    model_config: Dict[str, Any]
    system_prompt: Optional[str]
    mode: str
    retrieved_documents: Optional[List[str]]
    available_tools: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    user_query: Optional[str]
    final_response: Optional[str]
    conversation_id: Optional[str]  # 添加对话ID用于checkpointer
    # 深度研究相关状态
    research_iterations: Optional[int]
    search_history: Optional[List[Dict[str, Any]]]
    current_findings: Optional[List[str]]
    research_plan: Optional[str]


class LLMManager:
    """
    基于 LangGraph 的 LLM编排服务
    使用状态图管理多轮对话流程，支持 PostgresSaver checkpointer
    """
    
    def __init__(self):
        self._model_cache = {}  # 缓存已创建的模型实例
        self._graphs = {}  # 缓存不同模式的图
        self.retrieval_service = LLMRetrievalService()  # 检索服务依赖
        self.file_mgr = LLMFileProcessor()
        
    def _get_model(self, model_config: Dict[str, Any]) -> BaseChatModel:
        """获取或创建模型实例（带缓存）"""
        cache_key = f"{model_config['provider']}-{model_config['model_id']}"
        
        if cache_key not in self._model_cache:
            try:
                # 准备模型参数
                model_params = {
                    "temperature": model_config.get("temperature", 0.7),
                    "max_tokens": model_config.get("max_tokens"),
                }
                
                # 添加额外参数
                if model_config.get("extra_params"):
                    model_params.update(model_config["extra_params"])
                
                # 过滤None值
                model_params = {k: v for k, v in model_params.items() if v is not None}
                
                # 创建模型
                self._model_cache[cache_key] = create_chat_model(
                    provider=model_config["provider"],
                    model=model_config["model_id"],
                    **model_params
                )
            except Exception as e:
                raise ValueError(f"创建模型失败 {model_config['provider']}/{model_config['model_id']}: {str(e)}")
        
        return self._model_cache[cache_key]
    
    def _build_chat_graph(self) -> StateGraph:
        """构建聊天模式的状态图"""
        def chat_node(state: ConversationState) -> Dict[str, Any]:
            """聊天节点处理函数"""
            model = self._get_model(state["model_config"])
            
            # 准备消息列表
            final_messages = []
            
            # 添加系统提示 - 使用提示词管理器
            system_prompt = state.get("system_prompt")
            if not system_prompt:
                system_prompt = prompt_manager.get_chat_prompt()
            final_messages.append(SystemMessage(content=system_prompt))
            
            # 安全地添加用户消息
            user_messages = state.get("messages") or []
            final_messages.extend(user_messages)
            
            # 调用模型
            response = model.invoke(final_messages)
            
            return {
                "messages": [response],
                "final_response": response.content,
                "metadata": {"mode": "chat"}
            }
        
        # 构建图但不编译
        graph_builder = StateGraph(ConversationState)
        graph_builder.add_node("chat", chat_node)
        graph_builder.add_edge(START, "chat")
        graph_builder.add_edge("chat", END)
        
        return graph_builder  # 返回未编译的图构建器
    
    def _build_rag_graph(self) -> StateGraph:
        """构建RAG模式的状态图 - 基于DocQA Router设计"""
        
        def docqa_router_node(state: ConversationState) -> Dict[str, Any]:
            """DocQA路由节点 - 智能判断问题类型"""
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            model = self._get_model(state["model_config"])
            
            # 使用ChatPromptTemplate构建提示词
            router_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个智能问题分析助手，需要分析用户问题并分类处理。

请分析用户问题，判断其属于以下哪种类型：

**文档相关类型：**
1. **全文档分析** (full_document)：
   - 要求对文档进行总结、概括、综述
   - 需要分析整个文档的内容结构  
   - 要求提取文档的主要观点、结论
   - 需要对文档进行翻译、转换
   - 关键词：总结、概括、分析全文、整体内容、主要观点、文档概述

2. **关键词检索** (keyword_search)：
   - 询问特定的事实、数据、概念
   - 查找文档中的特定信息点
   - 回答具体问题，不需要完整文档
   - 关键词：什么是、如何、为什么、具体数据、特定概念

**非文档相关类型：**
3. **一般对话** (general_chat)：
   - 问候、寒暄、感谢等社交对话
   - 关于系统功能的询问
   - 不需要文档内容的常识性问题
   - 闲聊、娱乐性对话
   - 关键词：你好、谢谢、再见、怎么样、聊天

请准确分析并返回结构化结果。"""),
                ("user", "用户问题：{query}")
            ])
            
            try:
                # 使用结构化输出
                structured_model = model.with_structured_output(DocQARouterResult)
                
                # 调用模型进行路由判断
                result: DocQARouterResult = structured_model.invoke(
                    router_prompt.format_messages(query=user_query)
                )
                
                # 判断具体类型
                question_category = result.question_category
                analysis_type = result.analysis_type
                is_full_document_analysis = analysis_type == "full_document"
                is_non_document = question_category == "non_document"
                
                logger.info(f"DocQA路由判断: {question_category} -> {analysis_type} (置信度: {result.confidence})")
                logger.info(f"判断理由: {result.reasoning}")
                
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "question_category": question_category,
                        "analysis_type": analysis_type,
                        "is_full_document_analysis": is_full_document_analysis,
                        "is_non_document": is_non_document,
                        "routing_confidence": result.confidence,
                        "routing_reasoning": result.reasoning,
                        "routing_completed": True
                    }
                }
                
            except Exception as e:
                logger.error(f"DocQA路由判断失败: {str(e)}")
                # 路由失败，默认使用关键词检索
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "question_category": "document_related",
                        "analysis_type": "keyword_search",
                        "is_full_document_analysis": False,
                        "is_non_document": False,
                        "routing_error": str(e),
                        "routing_completed": True
                    }
                }
        
        async def sim_search_node(state: ConversationState) -> Dict[str, Any]:
            """相似度搜索节点 - 使用优化后的检索服务进行异步检索"""
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            metadata = state.get("metadata", {})
            model = self._get_model(state["model_config"])
            
            retrieved_docs = []
            retrieval_info = {}
            no_sim_results = False
            
            # 检查检索服务是否可用
            if not self.retrieval_service or not self.retrieval_service.is_ready:
                no_sim_results = True
                retrieval_info = {
                    "method": "langchain_similarity_search",
                    "status": "检索服务不可用",
                    "query": user_query,
                    "no_sim_results": True
                }
                logger.warning("相似度搜索：检索服务不可用")
                
                return {
                    "retrieved_documents": retrieved_docs,
                    "metadata": {
                        **state.get("metadata", {}),
                        "sim_search_completed": True,
                        "no_sim_results": no_sim_results,
                        "retrieval_info": retrieval_info,
                        "document_count": 0
                    }
                }
            
            # 权限验证：确保必要的安全参数存在
            user_id = metadata.get("user_id")
            if not user_id:
                logger.error("相似度搜索：缺少用户ID，存在安全风险")
                no_sim_results = True
                retrieval_info = {
                    "method": "langchain_similarity_search",
                    "status": "权限验证失败：缺少用户ID",
                    "query": user_query,
                    "no_sim_results": True,
                    "security_error": True
                }
                
                return {
                    "retrieved_documents": retrieved_docs,
                    "metadata": {
                        **state.get("metadata", {}),
                        "sim_search_completed": True,
                        "no_sim_results": no_sim_results,
                        "retrieval_info": retrieval_info,
                        "document_count": 0
                    }
                }
            
            try:
                # 第一步：使用模型优化查询
                query_optimization_prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个专业的查询优化助手。请将用户的原始问题转换为更适合向量检索的查询。

优化原则：
1. 提取核心关键词和概念
2. 去除冗余的语言表达
3. 增加相关的同义词和概念
4. 保持查询的语义完整性
5. 针对文档检索进行优化

请返回优化后的查询，保持简洁但信息丰富。"""),
                    ("user", "原始问题: {original_query}")
                ])
                
                logger.info(f"开始优化查询: '{user_query}'")
                
                # 调用模型优化查询
                optimized_response = model.invoke(
                    query_optimization_prompt.format_messages(original_query=user_query)
                )
                optimized_query = optimized_response.content.strip()
                
                logger.info(f"查询优化完成: '{user_query}' -> '{optimized_query}'")
                
                # 第二步：使用检索服务进行相似度搜索
                # 从元数据中获取用户和文件信息
                available_file_ids = metadata.get("available_file_ids", [])
                conversation_id = state.get("conversation_id")
                
                # 验证文件权限：确保用户有权访问指定的文件
                if available_file_ids:
                    logger.info(f"权限验证：用户 {user_id} 尝试访问文件 {available_file_ids}")
                    # 这里可以添加额外的权限检查逻辑
                    # 例如：查询数据库验证用户对这些文件的访问权限
                
                search_results = await self.retrieval_service.similarity_search_with_score(
                    query=optimized_query,
                    k=5,
                    user_id=user_id,
                    file_ids=available_file_ids if available_file_ids else None,
                    conversation_id=conversation_id
                )
                
                # 处理检索结果
                similarity_threshold = 0.3  # 相似度阈值
                filtered_results = [
                    (doc, score) for doc, score in search_results 
                    if score >= similarity_threshold
                ]
                
                if filtered_results:
                    retrieved_docs = [doc.page_content for doc, score in filtered_results]
                    scores = [score for doc, score in filtered_results]
                    
                    # 安全检查：验证返回的文档确实属于当前用户
                    security_validated = True
                    for doc, _ in filtered_results:
                        doc_user_id = doc.metadata.get("user_id")
                        if doc_user_id and doc_user_id != user_id:
                            logger.error(f"安全警告：检索到其他用户的文档！doc_user_id={doc_user_id}, current_user_id={user_id}")
                            security_validated = False
                            break
                    
                    if not security_validated:
                        # 发现安全问题，清空结果
                        retrieved_docs = []
                        no_sim_results = True
                        retrieval_info = {
                            "method": "langchain_similarity_search",
                            "status": "安全验证失败：检测到跨用户访问",
                            "security_error": True,
                            "no_sim_results": True
                        }
                    else:
                        retrieval_info = {
                            "method": "langchain_similarity_search",
                            "original_query": user_query,
                            "optimized_query": optimized_query,
                            "document_count": len(retrieved_docs),
                            "similarity_scores": scores,
                            "similarity_threshold": similarity_threshold,
                            "total_candidates": len(search_results),
                            "status": "相似度搜索成功",
                            "security_validated": True
                        }
                        
                        logger.info(f"相似度搜索成功: 找到 {len(retrieved_docs)} 个相关文档片段")
                    
                else:
                    # 没有满足阈值的结果
                    no_sim_results = True
                    all_scores = [score for doc, score in search_results] if search_results else []
                    
                    retrieval_info = {
                        "method": "langchain_similarity_search", 
                        "original_query": user_query,
                        "optimized_query": optimized_query,
                        "document_count": 0,
                        "similarity_threshold": similarity_threshold,
                        "total_candidates": len(search_results),
                        "max_score": max(all_scores) if all_scores else 0,
                        "status": "无满足阈值的相似文档",
                        "no_sim_results": True
                    }
                    
                    logger.info(f"相似度搜索: 无满足阈值 {similarity_threshold} 的文档")
                
            except Exception as e:
                logger.error(f"相似度搜索失败: {str(e)}")
                no_sim_results = True
                retrieval_info = {
                    "method": "langchain_similarity_search",
                    "original_query": user_query,
                    "optimized_query": optimized_query if 'optimized_query' in locals() else user_query,
                    "error": str(e),
                    "status": "相似度搜索失败",
                    "no_sim_results": True
                }
            
            return {
                "retrieved_documents": retrieved_docs,
                "metadata": {
                    **state.get("metadata", {}),
                    "sim_search_completed": True,
                    "no_sim_results": no_sim_results,
                    "retrieval_info": retrieval_info,
                    "document_count": len(retrieved_docs)
                }
            }
        
        async def full_doc_qa_node(state: ConversationState) -> Dict[str, Any]:
            """全文档QA节点 - 获取更多文档内容进行全文分析"""
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            metadata = state.get("metadata", {})
            
            full_documents = []
            retrieval_info = {}
            
            # 检查检索服务是否可用
            if not self.retrieval_service or not self.retrieval_service.is_ready:
                retrieval_info = {
                    "method": "full_document_retrieval",
                    "status": "检索服务不可用",
                    "query": user_query
                }
                logger.warning("全文档QA：检索服务不可用")
            else:
                try:
                    logger.info("开始全文档内容获取")
                    
                    # 使用更大的k值和更低的阈值获取更多文档内容
                    # 从元数据中获取用户和文件信息
                    user_id = metadata.get("user_id")
                    available_file_ids = metadata.get("available_file_ids", [])
                    conversation_id = state.get("conversation_id")
                    
                    search_results = await self.retrieval_service.similarity_search_with_score(
                        query=user_query,
                        k=20,  # 获取更多文档片段
                        user_id=user_id,
                        file_ids=available_file_ids if available_file_ids else None,
                        conversation_id=conversation_id
                    )
                    
                    # 对于全文档分析，我们使用更宽松的阈值
                    full_doc_threshold = 0.1  # 很低的阈值，获取更多内容
                    filtered_results = [
                        (doc, score) for doc, score in search_results 
                        if score >= full_doc_threshold
                    ]
                    
                    if filtered_results:
                        full_documents = [doc.page_content for doc, score in filtered_results]
                        scores = [score for doc, score in filtered_results]
                        
                        retrieval_info = {
                            "method": "full_document_retrieval",
                            "query": user_query,
                            "document_count": len(full_documents),
                            "similarity_scores": scores,
                            "similarity_threshold": full_doc_threshold,
                            "total_candidates": len(search_results),
                            "status": "全文档获取成功"
                        }
                        
                        logger.info(f"全文档获取成功: 获得 {len(full_documents)} 个文档片段")
                    else:
                        # 如果连低阈值都没有结果，则获取所有候选结果
                        if search_results:
                            full_documents = [doc.page_content for doc, score in search_results]
                            scores = [score for doc, score in search_results]
                            
                            retrieval_info = {
                                "method": "full_document_retrieval",
                                "query": user_query,
                                "document_count": len(full_documents),
                                "similarity_scores": scores,
                                "similarity_threshold": "all_candidates",
                                "total_candidates": len(search_results),
                                "status": "全文档获取成功（所有候选）"
                            }
                            
                            logger.info(f"全文档获取成功（所有候选）: 获得 {len(full_documents)} 个文档片段")
                        else:
                            retrieval_info = {
                                "method": "full_document_retrieval",
                                "query": user_query,
                                "document_count": 0,
                                "status": "未找到任何文档内容"
                            }
                            logger.warning("全文档获取: 未找到任何文档内容")
                    
                except Exception as e:
                    logger.error(f"全文档获取失败: {str(e)}")
                    retrieval_info = {
                        "method": "full_document_retrieval",
                        "error": str(e),
                        "status": "全文档获取失败",
                        "query": user_query
                    }
            
            return {
                "retrieved_documents": full_documents,
                "metadata": {
                    **state.get("metadata", {}),
                    "full_doc_completed": True,
                    "retrieval_info": retrieval_info,
                    "document_count": len(full_documents),
                    "processing_type": "全文档分析"
                }
            }
        
        def unified_response_node(state: ConversationState) -> Dict[str, Any]:
            """统一响应节点 - 处理所有类型的问题回答"""
            model = self._get_model(state["model_config"])
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            retrieved_docs = state.get("retrieved_documents") or []
            metadata = state.get("metadata", {})
            
            analysis_type = metadata.get("analysis_type", "keyword_search")
            question_category = metadata.get("question_category", "document_related")
            is_full_document_analysis = metadata.get("is_full_document_analysis", False)
            is_non_document = metadata.get("is_non_document", False)
            processing_type = metadata.get("processing_type", "知识检索")
            
            # 根据问题类型构建不同的提示词
            available_file_ids = metadata.get("available_file_ids", [])
            
            if is_non_document:
                # 非文档相关问题 - 直接对话
                response_prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个友好、专业的AI助手。用户的问题不需要文档内容支持，请直接基于你的知识库回答。

对于以下类型的问题，请提供相应的回答：
- 问候和寒暄：友好回应
- 功能询问：简要说明系统功能  
- 常识问题：基于通用知识回答
- 闲聊对话：自然互动

请保持回答简洁、准确、友好。"""),
                    ("placeholder", "{messages}")
                ])
                context_instruction = None
                
            elif not available_file_ids and not retrieved_docs:
                # 没有可用文件且问题是文档相关的
                response_prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个智能文档助手。用户提出了与文档相关的问题，但当前对话中没有可用的文档。

请友好地提醒用户：
1. 需要先上传相关文档才能进行文档分析
2. 支持的文件格式包括：PDF、DOCX、TXT等
3. 上传文档后，您就可以帮助用户分析、总结和回答文档相关的问题

请用温和、有帮助的语气回应。"""),
                    ("placeholder", "{messages}")
                ])
                context_instruction = None
                
            elif is_full_document_analysis:
                # 全文档分析
                response_prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个专业的文档分析助手。当前任务类型：全文档分析

请基于提供的完整文档内容进行深度分析、总结或处理。
注意整体性和全面性，提供结构化的分析结果。

{context_instruction}"""),
                    ("placeholder", "{messages}")
                ])
                
                # 准备全文档上下文
                if retrieved_docs:
                    context = "\n\n=== 文档分隔符 ===\n\n".join(retrieved_docs)
                    context_instruction = f"完整文档内容：\n\n{context}"
                else:
                    context_instruction = "没有找到相关文档内容，请基于常识回答用户问题。"
                    
            else:
                # 关键词检索
                response_prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个专业的知识问答助手。当前任务类型：关键词检索

请基于检索到的相关文档片段回答用户的具体问题。
重点关注问题的准确回答，引用相关片段支持答案。

{context_instruction}"""),
                    ("placeholder", "{messages}")
                ])
                
                # 准备关键词检索上下文
                if retrieved_docs:
                    context = "\n\n".join([f"相关片段 {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
                    context_instruction = f"检索到的相关内容：\n\n{context}"
                else:
                    context_instruction = "没有找到相关文档内容，请基于常识回答用户问题。"
            
            # 调用模型生成回答
            try:
                if is_non_document:
                    # 非文档相关问题直接调用
                    response = model.invoke(response_prompt.format_messages(messages=messages))
                    processing_type = "直接对话"
                    retrieval_method = "none"
                else:
                    # 文档相关问题需要上下文
                    response = model.invoke(response_prompt.format_messages(
                        context_instruction=context_instruction,
                        messages=messages
                    ))
                    retrieval_method = "full_document" if is_full_document_analysis else "similarity_search"
                
                # 准备来源信息
                sources = []
                if retrieved_docs and not is_non_document:
                    sources = [
                        {
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                            "index": i,
                            "type": "full_document" if is_full_document_analysis else "similarity_chunk"
                        } 
                        for i, doc in enumerate(retrieved_docs)
                    ]
                
                # 准备响应元数据
                response_metadata = {
                    "mode": "rag",
                    "question_category": question_category,
                    "analysis_type": analysis_type,
                    "is_full_document_analysis": is_full_document_analysis,
                    "is_non_document": is_non_document,
                    "processing_type": processing_type,
                    "document_count": len(retrieved_docs) if not is_non_document else 0,
                    "sources": sources,
                    "retrieval_method": retrieval_method
                }
                
                # 添加检索信息
                if "retrieval_info" in metadata and not is_non_document:
                    response_metadata["retrieval_info"] = metadata["retrieval_info"]
                
                logger.info(f"统一响应生成成功，类型: {question_category} -> {analysis_type}, 文档数: {len(retrieved_docs) if not is_non_document else 0}")
                
                return {
                    "messages": [response],
                    "final_response": response.content,
                    "metadata": response_metadata
                }
                
            except Exception as e:
                logger.error(f"统一响应生成失败: {str(e)}")
                return {
                    "final_response": f"生成回答时出错: {str(e)}",
                    "metadata": {
                        "mode": "rag",
                        "error": True,
                        "error_type": "unified_response_failed",
                        "error_message": str(e),
                        "analysis_type": analysis_type,
                        "question_category": question_category
                    }
                }
        
        def route_after_router(state: ConversationState) -> str:
            """路由器后的条件路由 - 根据问题类型选择处理流程"""
            metadata = state.get("metadata", {})
            is_non_document = metadata.get("is_non_document", False)
            is_full_document_analysis = metadata.get("is_full_document_analysis", False)
            available_file_ids = metadata.get("available_file_ids", [])
            
            # 添加调试日志
            logger.info(f"路由决策: is_non_document={is_non_document}, is_full_document_analysis={is_full_document_analysis}, available_file_ids={available_file_ids}")
            
            # 如果是非文档相关问题，直接响应
            if is_non_document:
                logger.info("判断为非文档相关问题，进入统一响应流程")
                return "unified_response"
            
            # 如果没有可用文件，直接生成答案（提醒用户上传文件）
            if not available_file_ids:
                logger.info("没有可用文件，进入统一响应流程")
                return "unified_response"
            
            # 如果是全文档分析，直接进入全文档QA节点
            if is_full_document_analysis:
                logger.info("判断为全文档分析，进入全文档QA流程")
                return "full_doc_qa"
            
            # 否则进入相似度搜索
            logger.info("判断为关键词检索，进入相似度搜索流程")
            return "sim_search"
        
        def route_after_sim_search(state: ConversationState) -> str:
            """相似度搜索后的条件路由 - 判断NoSim"""
            metadata = state.get("metadata", {})
            no_sim_results = metadata.get("no_sim_results", False)
            
            # 如果没有相似结果，进入全文档QA节点
            if no_sim_results:
                logger.info("相似度搜索无结果，转入全文档QA流程")
                return "full_doc_qa"
            
            # 有相似结果，进入统一响应生成
            logger.info("相似度搜索有结果，进入统一响应生成")
            return "unified_response"
        
        # 构建图但不编译
        graph_builder = StateGraph(ConversationState)
        
        # 添加节点
        graph_builder.add_node("docqa_router", docqa_router_node)
        graph_builder.add_node("sim_search", sim_search_node)  # 异步节点
        graph_builder.add_node("full_doc_qa", full_doc_qa_node)
        graph_builder.add_node("unified_response", unified_response_node)
        
        # 添加边和条件路由
        graph_builder.add_edge(START, "docqa_router")
        
        # DocQA Router 后的条件路由
        graph_builder.add_conditional_edges(
            "docqa_router",
            route_after_router,
            {
                "unified_response": "unified_response",
                "sim_search": "sim_search", 
                "full_doc_qa": "full_doc_qa"
            }
        )
        
        # 相似度搜索后的条件路由 (判断NoSim)
        graph_builder.add_conditional_edges(
            "sim_search",
            route_after_sim_search,
            {
                "full_doc_qa": "full_doc_qa",
                "unified_response": "unified_response"
            }
        )
        
        # 全文档QA后进入统一响应
        graph_builder.add_edge("full_doc_qa", "unified_response")
        
        # 统一响应完成
        graph_builder.add_edge("unified_response", END)
        
        return graph_builder  # 返回未编译的图构建器
    
    def _build_agent_graph(self) -> StateGraph:
        """构建Agent模式的状态图"""
        def agent_planning_node(state: ConversationState) -> Dict[str, Any]:
            """Agent规划节点"""
            tools = state.get("available_tools", [])
            
            # 添加系统提示 - 使用Agent专用提示词
            system_prompt = state.get("system_prompt")
            if not system_prompt:
                system_prompt = prompt_manager.get_agent_prompt(available_tools=tools)
            
            planning_messages = [SystemMessage(content=system_prompt)]
            
            # 安全地添加用户消息
            user_messages = state.get("messages") or []
            planning_messages.extend(user_messages)
            
            return {
                "messages": planning_messages,
                "metadata": {"planning_completed": True, "available_tools": tools}
            }
        
        def agent_response_node(state: ConversationState) -> Dict[str, Any]:
            """Agent响应节点"""
            model = self._get_model(state["model_config"])
            
            # 安全地获取消息
            messages = state.get("messages") or []
            if not messages:
                # 如果没有消息，返回错误
                return {
                    "final_response": "没有可处理的消息",
                    "metadata": {
                        "mode": "agent",
                        "error": True,
                        "error_type": "no_messages",
                        "error_message": "没有可处理的消息"
                    }
                }
            
            # 调用模型
            response = model.invoke(messages)
            
            return {
                "messages": [response],
                "final_response": response.content,
                "metadata": {"mode": "agent", "is_tool_use": False}
            }
        
        # 构建图但不编译
        graph_builder = StateGraph(ConversationState)
        graph_builder.add_node("agent_planning", agent_planning_node)
        graph_builder.add_node("agent_response", agent_response_node)
        
        graph_builder.add_edge(START, "agent_planning")
        graph_builder.add_edge("agent_planning", "agent_response")
        graph_builder.add_edge("agent_response", END)
        
        return graph_builder  # 返回未编译的图构建器
    
    def _build_search_graph(self) -> StateGraph:
        """构建搜索模式的状态图"""
        
        def search_planning_node(state: ConversationState) -> Dict[str, Any]:
            """搜索规划节点 - 分析用户问题并生成搜索查询"""
            model = self._get_model(state["model_config"])
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            
            # 构建搜索规划提示
            planning_prompt = f"""
请分析以下用户问题，并生成1-3个相关但不重复的搜索查询来获取全面信息。
每个搜索查询应该从不同角度或方面来探索这个问题。

用户问题：{user_query}

请只返回搜索查询，每行一个，不需要其他解释：
"""
            
            planning_messages = [SystemMessage(content=planning_prompt)]
            
            try:
                response = model.invoke(planning_messages)
                search_queries_text = response.content.strip()
                
                # 解析搜索查询
                search_queries = []
                for line in search_queries_text.split('\n'):
                    query = line.strip()
                    if query and not query.startswith('#') and not query.startswith('搜索查询'):
                        # 清理可能的序号或标点
                        import re
                        query = re.sub(r'^\d+[\.、]\s*', '', query)
                        query = query.strip('- ')
                        if query:
                            search_queries.append(query)
                
                # 限制搜索查询数量
                search_queries = search_queries[:3]
                
                if not search_queries:
                    search_queries = [user_query]  # 如果解析失败，使用原始查询
                
                logger.info(f"搜索规划完成，生成 {len(search_queries)} 个查询: {search_queries}")
                
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "search_queries": search_queries,
                        "planning_completed": True
                    }
                }
                
            except Exception as e:
                logger.error(f"搜索规划失败: {str(e)}")
                # 规划失败，使用原始查询
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "search_queries": [user_query],
                        "planning_completed": True,
                        "planning_error": str(e)
                    }
                }
        
        def execute_search_node(state: ConversationState) -> Dict[str, Any]:
            """执行搜索节点 - 使用DuckDuckGo进行实际搜索"""
            import asyncio
            from app.llm.tools.duckduckgo_search import duckduckgo_search_tool
            
            metadata = state.get("metadata", {})
            search_queries = metadata.get("search_queries", [])
            
            search_results = []
            search_info = {}
            
            if not search_queries:
                search_info = {
                    "status": "搜索失败",
                    "error": "没有生成搜索查询",
                    "query_count": 0,
                    "result_count": 0
                }
            else:
                try:
                    logger.info(f"开始执行 {len(search_queries)} 个搜索查询")
                    
                    # 使用DuckDuckGo工具进行搜索
                    for i, query in enumerate(search_queries):
                        try:
                            # 调用DuckDuckGo搜索工具
                            result = duckduckgo_search_tool.invoke({"query": query})
                            
                            if result and isinstance(result, str):
                                # 添加搜索结果，标明来源查询
                                search_results.append({
                                    "query": query,
                                    "content": result,
                                    "source": f"搜索查询 {i+1}",
                                    "tool": "duckduckgo"
                                })
                                logger.info(f"查询 '{query}' 搜索成功，结果长度: {len(result)}")
                            else:
                                logger.warning(f"查询 '{query}' 没有返回有效结果")
                        
                        except Exception as e:
                            logger.error(f"查询 '{query}' 搜索失败: {str(e)}")
                            search_results.append({
                                "query": query,
                                "content": f"搜索失败: {str(e)}",
                                "source": f"搜索查询 {i+1}",
                                "tool": "duckduckgo",
                                "error": True
                            })
                    
                    search_info = {
                        "status": "搜索完成",
                        "query_count": len(search_queries),
                        "result_count": len([r for r in search_results if not r.get("error", False)]),
                        "error_count": len([r for r in search_results if r.get("error", False)]),
                        "queries": search_queries
                    }
                    
                    logger.info(f"搜索执行完成: 成功 {search_info['result_count']} 个，失败 {search_info['error_count']} 个")
                    
                except Exception as e:
                    logger.error(f"搜索执行失败: {str(e)}")
                    search_info = {
                        "status": "搜索失败",
                        "error": str(e),
                        "query_count": len(search_queries),
                        "result_count": 0
                    }
            
            return {
                "retrieved_documents": [r["content"] for r in search_results if not r.get("error", False)],
                "metadata": {
                    **state.get("metadata", {}),
                    "search_completed": True,
                    "search_results": search_results,
                    "search_info": search_info
                }
            }
        
        def search_response_node(state: ConversationState) -> Dict[str, Any]:
            """搜索响应节点 - 基于搜索结果生成回答"""
            model = self._get_model(state["model_config"])
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            retrieved_docs = state.get("retrieved_documents") or []
            metadata = state.get("metadata", {})
            
            # 获取搜索相关信息
            search_info = metadata.get("search_info", {})
            search_results = metadata.get("search_results", [])
            
            # 构建系统提示
            system_prompt = state.get("system_prompt")
            if not system_prompt:
                system_prompt = prompt_manager.get_search_prompt()
            
            # 构建消息列表
            final_messages = [SystemMessage(content=system_prompt)]
            
            # 如果有搜索结果，添加搜索上下文
            if retrieved_docs:
                context_parts = []
                for i, (doc, result) in enumerate(zip(retrieved_docs, search_results)):
                    if not result.get("error", False):
                        context_parts.append(f"## 搜索结果 {i+1}: {result['query']}\n\n{doc}")
                
                if context_parts:
                    search_context = "\n\n---\n\n".join(context_parts)
                    context_message = f"基于以下搜索结果回答用户问题：\n\n{search_context}"
                    final_messages.append(SystemMessage(content=context_message))
            else:
                # 没有搜索结果的情况
                no_result_message = "搜索没有返回有效结果，请基于常识回答用户问题，并说明可能需要更具体的搜索词。"
                final_messages.append(SystemMessage(content=no_result_message))
            
            # 添加用户消息
            final_messages.extend(messages)
            
            # 调用模型生成回答
            try:
                response = model.invoke(final_messages)
                
                # 准备来源信息
                sources = []
                if search_results:
                    for result in search_results:
                        if not result.get("error", False):
                            sources.append({
                                "query": result["query"],
                                "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                                "tool": result.get("tool", "unknown")
                            })
                
                # 准备响应元数据
                response_metadata = {
                    "mode": "search",
                    "search_info": search_info,
                    "source_count": len(sources),
                    "sources": sources,
                    "processing_type": "联网搜索"
                }
                
                return {
                    "messages": [response],
                    "final_response": response.content,
                    "metadata": response_metadata
                }
                
            except Exception as e:
                logger.error(f"搜索响应生成失败: {str(e)}")
                return {
                    "final_response": f"生成搜索回答时出错: {str(e)}",
                    "metadata": {
                        "mode": "search",
                        "error": True,
                        "error_type": "response_generation_failed",
                        "error_message": str(e),
                        "search_info": search_info
                    }
                }
        
        # 构建图但不编译
        graph_builder = StateGraph(ConversationState)
        
        # 添加节点
        graph_builder.add_node("search_planning", search_planning_node)
        graph_builder.add_node("execute_search", execute_search_node)
        graph_builder.add_node("search_response", search_response_node)
        
        # 添加边：规划 -> 搜索 -> 响应
        graph_builder.add_edge(START, "search_planning")
        graph_builder.add_edge("search_planning", "execute_search")
        graph_builder.add_edge("execute_search", "search_response")
        graph_builder.add_edge("search_response", END)
        
        return graph_builder  # 返回未编译的图构建器
    
    def _build_deepresearch_graph(self) -> StateGraph:
        """构建深度研究模式的状态图 - 基于ReAct模式的多轮搜索"""
        
        def research_planning_node(state: ConversationState) -> Dict[str, Any]:
            """研究规划节点 - 分析问题并制定研究计划"""
            model = self._get_model(state["model_config"])
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            
            # 构建研究规划提示
            planning_prompt = f"""
🤔 深度研究规划

作为专业研究分析师，请为以下研究主题制定详细的研究计划：

研究主题：{user_query}

请分析并制定研究计划，包括：

1. **研究目标分解**
   - 主要研究目标
   - 关键研究问题 
   - 需要收集的信息类型

2. **搜索策略规划**
   - 第一轮搜索：基础信息和背景
   - 第二轮搜索：深度分析和专业观点
   - 第三轮搜索：最新发展和趋势

3. **预期产出**
   - 最终报告应包含的核心内容
   - 重点关注的分析角度

请提供一个结构化的研究计划，指导后续的多轮搜索和分析。
"""
            
            planning_messages = [SystemMessage(content=planning_prompt)]
            
            try:
                response = model.invoke(planning_messages)
                research_plan = response.content.strip()
                
                # 根据计划生成第一轮搜索查询
                query_prompt = f"""
基于以下研究计划，生成3个第一轮搜索查询，用于收集基础信息和背景：

研究计划：
{research_plan}

原始问题：{user_query}

请只返回搜索查询，每行一个：
"""
                
                query_response = model.invoke([SystemMessage(content=query_prompt)])
                search_queries_text = query_response.content.strip()
                
                # 解析搜索查询
                initial_queries = []
                for line in search_queries_text.split('\n'):
                    query = line.strip()
                    if query and not query.startswith('#'):
                        import re
                        query = re.sub(r'^\d+[\.、]\s*', '', query)
                        query = query.strip('- ')
                        if query:
                            initial_queries.append(query)
                
                initial_queries = initial_queries[:3]  # 限制为3个查询
                
                if not initial_queries:
                    initial_queries = [user_query]  # 回退
                
                logger.info(f"研究规划完成，生成初始查询: {initial_queries}")
                
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "research_phase": "planning",
                        "current_iteration": 1,
                        "max_iterations": 3
                    },
                    "research_plan": research_plan,
                    "research_iterations": 1,
                    "search_history": [{
                        "iteration": 1,
                        "phase": "initial_exploration",
                        "queries": initial_queries,
                        "purpose": "收集基础信息和背景"
                    }],
                    "current_findings": []
                }
                
            except Exception as e:
                logger.error(f"研究规划失败: {str(e)}")
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "research_phase": "planning",
                        "planning_error": str(e)
                    },
                    "research_plan": f"由于规划失败，将直接搜索用户问题: {user_query}",
                    "research_iterations": 1,
                    "search_history": [{
                        "iteration": 1,
                        "phase": "fallback",
                        "queries": [user_query],
                        "purpose": "直接搜索用户问题"
                    }],
                    "current_findings": []
                }
        
        def execute_research_search_node(state: ConversationState) -> Dict[str, Any]:
            """执行研究搜索节点 - 执行当前迭代的搜索"""
            from app.llm.tools.duckduckgo_search import duckduckgo_search_tool
            
            search_history = state.get("search_history", [])
            current_iteration = state.get("research_iterations", 1)
            
            if not search_history:
                logger.error("没有搜索历史记录")
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "search_error": "没有搜索历史记录"
                    }
                }
            
            # 获取当前迭代的搜索信息
            current_search = None
            for search in search_history:
                if search.get("iteration") == current_iteration:
                    current_search = search
                    break
            
            if not current_search:
                logger.error(f"找不到第 {current_iteration} 次迭代的搜索信息")
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "search_error": f"找不到第 {current_iteration} 次迭代的搜索信息"
                    }
                }
            
            search_queries = current_search.get("queries", [])
            search_results = []
            
            logger.info(f"开始第 {current_iteration} 轮搜索，查询数量: {len(search_queries)}")
            
            # 执行当前迭代的所有搜索查询
            for i, query in enumerate(search_queries):
                try:
                    result = duckduckgo_search_tool.invoke({"query": query})
                    if result and isinstance(result, str):
                        search_results.append({
                            "iteration": current_iteration,
                            "query": query,
                            "content": result,
                            "index": i + 1,
                            "success": True
                        })
                        logger.info(f"查询 '{query}' 搜索成功，结果长度: {len(result)}")
                    else:
                        logger.warning(f"查询 '{query}' 没有返回有效结果")
                        search_results.append({
                            "iteration": current_iteration,
                            "query": query,
                            "content": "搜索未返回有效结果",
                            "index": i + 1,
                            "success": False
                        })
                except Exception as e:
                    logger.error(f"查询 '{query}' 搜索失败: {str(e)}")
                    search_results.append({
                        "iteration": current_iteration,
                        "query": query,
                        "content": f"搜索失败: {str(e)}",
                        "index": i + 1,
                        "success": False,
                        "error": str(e)
                    })
            
            # 更新搜索历史，添加结果
            updated_search_history = search_history.copy()
            for search in updated_search_history:
                if search.get("iteration") == current_iteration:
                    search["results"] = search_results
                    search["completed"] = True
                    break
            
            # 收集当前发现
            current_findings = state.get("current_findings", [])
            new_findings = []
            for result in search_results:
                if result.get("success", False):
                    new_findings.append(f"【第{current_iteration}轮-查询{result['index']}】{result['query']}: {result['content'][:300]}...")
            
            updated_findings = current_findings + new_findings
            
            logger.info(f"第 {current_iteration} 轮搜索完成，成功: {len([r for r in search_results if r.get('success')])}, 失败: {len([r for r in search_results if not r.get('success')])}")
            
            return {
                "search_history": updated_search_history,
                "current_findings": updated_findings,
                "retrieved_documents": [r["content"] for r in search_results if r.get("success", False)],
                "metadata": {
                    **state.get("metadata", {}),
                    "current_iteration": current_iteration,
                    "search_completed": True,
                    "search_results_count": len([r for r in search_results if r.get("success")])
                }
            }
        
        def research_analysis_node(state: ConversationState) -> Dict[str, Any]:
            """研究分析节点 - 分析当前结果并决定是否继续"""
            model = self._get_model(state["model_config"])
            current_iteration = state.get("research_iterations", 1)
            max_iterations = state.get("metadata", {}).get("max_iterations", 3)
            current_findings = state.get("current_findings", [])
            research_plan = state.get("research_plan", "")
            user_query = state.get("user_query", "")
            
            # 如果已达到最大迭代次数，标记完成
            if current_iteration >= max_iterations:
                logger.info(f"已达到最大迭代次数 {max_iterations}，准备生成最终报告")
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "research_phase": "final_report",
                        "analysis_complete": True,
                        "continue_research": False
                    }
                }
            
            # 分析当前收集的信息
            findings_text = "\n\n".join(current_findings) if current_findings else "暂无有效发现"
            
            analysis_prompt = f"""
🤔 研究进度分析

原始研究问题：{user_query}

研究计划：
{research_plan}

当前迭代：{current_iteration}/{max_iterations}

已收集的信息：
{findings_text}

请分析当前研究进度：

1. **信息完整性评估**
   - 当前信息是否足够回答原始问题？
   - 还有哪些关键信息缺失？

2. **下一轮搜索建议**
   - 如果需要继续研究，应该搜索什么？
   - 建议3个具体的搜索查询

3. **研究决策**
   - 是否应该继续下一轮搜索？
   - 还是可以开始生成最终报告？

请最后明确回答：CONTINUE（继续研究）或 COMPLETE（完成研究）
"""
            
            try:
                response = model.invoke([SystemMessage(content=analysis_prompt)])
                analysis_result = response.content.strip()
                
                # 判断是否继续研究
                continue_research = "CONTINUE" in analysis_result.upper() and "COMPLETE" not in analysis_result.upper()
                
                if continue_research and current_iteration < max_iterations:
                    # 提取下一轮搜索查询
                    next_queries = []
                    lines = analysis_result.split('\n')
                    capture_queries = False
                    for line in lines:
                        line = line.strip()
                        if '搜索查询' in line or 'queries' in line.lower():
                            capture_queries = True
                            continue
                        if capture_queries and line:
                            if line.startswith(('1.', '2.', '3.', '-', '•')):
                                import re
                                query = re.sub(r'^[\d\.\-\•\s]+', '', line).strip()
                                if query:
                                    next_queries.append(query)
                    
                    # 如果没有提取到查询，生成默认查询
                    if not next_queries:
                        next_queries = [f"{user_query} 最新发展", f"{user_query} 专家观点", f"{user_query} 案例分析"]
                    
                    next_queries = next_queries[:3]  # 限制为3个
                    
                    # 更新搜索历史，添加下一轮
                    search_history = state.get("search_history", [])
                    next_iteration = current_iteration + 1
                    search_history.append({
                        "iteration": next_iteration,
                        "phase": f"deep_dive_{next_iteration}",
                        "queries": next_queries,
                        "purpose": f"第{next_iteration}轮深度研究"
                    })
                    
                    logger.info(f"决定继续第 {next_iteration} 轮研究，查询: {next_queries}")
                    
                    return {
                        "search_history": search_history,
                        "research_iterations": next_iteration,
                        "metadata": {
                            **state.get("metadata", {}),
                            "research_phase": f"iteration_{next_iteration}",
                            "continue_research": True,
                            "analysis_result": analysis_result[:500] + "..."  # 截断以节省空间
                        }
                    }
                else:
                    logger.info("分析决定完成研究，准备生成最终报告")
                    return {
                        "metadata": {
                            **state.get("metadata", {}),
                            "research_phase": "final_report",
                            "continue_research": False,
                            "analysis_result": analysis_result[:500] + "..."
                        }
                    }
                    
            except Exception as e:
                logger.error(f"研究分析失败: {str(e)}")
                # 分析失败，默认完成研究
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "research_phase": "final_report",
                        "continue_research": False,
                        "analysis_error": str(e)
                    }
                }
        
        def generate_research_report_node(state: ConversationState) -> Dict[str, Any]:
            """生成研究报告节点 - 基于所有收集的信息生成最终报告"""
            model = self._get_model(state["model_config"])
            user_query = state.get("user_query", "")
            research_plan = state.get("research_plan", "")
            current_findings = state.get("current_findings", [])
            search_history = state.get("search_history", [])
            messages = state.get("messages") or []
            
            # 构建系统提示
            system_prompt = state.get("system_prompt")
            if not system_prompt:
                system_prompt = prompt_manager.get_deepresearch_prompt()
            
            # 准备研究过程总结
            research_summary = []
            for search in search_history:
                iteration = search.get("iteration", 0)
                phase = search.get("phase", "unknown")
                queries = search.get("queries", [])
                purpose = search.get("purpose", "")
                research_summary.append(f"第{iteration}轮 ({phase}): {purpose} - 查询: {', '.join(queries)}")
            
            research_process = "\n".join(research_summary)
            
            # 整理所有发现
            all_findings = "\n\n".join(current_findings) if current_findings else "未收集到有效信息"
            
            # 构建最终报告生成提示
            report_prompt = f"""
基于深度研究结果，请生成一份全面的研究报告：

## 研究背景
原始问题：{user_query}

研究计划：
{research_plan}

## 研究过程
{research_process}

## 收集的信息
{all_findings}

## 要求
请生成一份结构化的研究报告，包括：

1. **执行摘要** - 核心发现和关键结论
2. **详细分析** - 分主题的深入分析
3. **关键发现** - 重要数据和洞察
4. **多元视角** - 不同角度的观点
5. **结论与建议** - 总结和建议
6. **研究局限** - 承认信息的限制

请确保报告：
- 结构清晰，逻辑严密
- 基于实际收集的信息
- 提供有洞察力的分析
- 使用中文撰写
"""
            
            # 构建消息列表
            final_messages = [SystemMessage(content=system_prompt)]
            final_messages.append(SystemMessage(content=report_prompt))
            final_messages.extend(messages)
            
            try:
                response = model.invoke(final_messages)
                
                # 准备响应元数据
                response_metadata = {
                    "mode": "deepresearch",
                    "total_iterations": len(search_history),
                    "total_findings": len(current_findings),
                    "research_process": research_process,
                    "processing_type": "深度研究报告"
                }
                
                # 添加搜索来源信息
                sources = []
                for finding in current_findings:
                    if "】" in finding:
                        source_info = finding.split("】")[0] + "】"
                        content_preview = finding.split("】")[1][:200] if "】" in finding else finding[:200]
                        sources.append({
                            "source": source_info,
                            "content": content_preview + "..." if len(content_preview) == 200 else content_preview
                        })
                
                response_metadata["sources"] = sources[:10]  # 限制显示前10个来源
                
                logger.info(f"深度研究报告生成成功，总迭代次数: {len(search_history)}, 发现数量: {len(current_findings)}")
                
                return {
                    "messages": [response],
                    "final_response": response.content,
                    "metadata": response_metadata
                }
                
            except Exception as e:
                logger.error(f"研究报告生成失败: {str(e)}")
                return {
                    "final_response": f"生成研究报告时出错: {str(e)}",
                    "metadata": {
                        "mode": "deepresearch",
                        "error": True,
                        "error_type": "report_generation_failed",
                        "error_message": str(e),
                        "total_iterations": len(search_history),
                        "total_findings": len(current_findings)
                    }
                }
        
        def route_research_flow(state: ConversationState) -> str:
            """路由研究流程 - 决定下一步动作"""
            metadata = state.get("metadata", {})
            research_phase = metadata.get("research_phase", "planning")
            continue_research = metadata.get("continue_research", True)
            
            if research_phase == "planning":
                return "execute_search"
            elif research_phase.startswith("iteration_") or metadata.get("search_completed", False):
                if continue_research:
                    return "execute_search"
                else:
                    return "generate_report"
            elif research_phase == "final_report":
                return "generate_report"
            else:
                # 默认分析当前状态
                return "analyze_progress"
        
        # 构建图但不编译
        graph_builder = StateGraph(ConversationState)
        
        # 添加节点
        graph_builder.add_node("research_planning", research_planning_node)
        graph_builder.add_node("execute_search", execute_research_search_node)
        graph_builder.add_node("analyze_progress", research_analysis_node)
        graph_builder.add_node("generate_report", generate_research_report_node)
        
        # 添加边和条件路由
        graph_builder.add_edge(START, "research_planning")
        
        # 从规划到执行搜索
        graph_builder.add_edge("research_planning", "execute_search")
        
        # 从搜索执行到分析进度
        graph_builder.add_edge("execute_search", "analyze_progress")
        
        # 从分析进度的条件路由
        graph_builder.add_conditional_edges(
            "analyze_progress",
            route_research_flow,
            {
                "execute_search": "execute_search",
                "generate_report": "generate_report"
            }
        )
        
        # 生成报告到结束
        graph_builder.add_edge("generate_report", END)
        
        return graph_builder  # 返回未编译的图构建器
    
    async def _get_graph(self, mode: str, conversation_id: Optional[UUID] = None):
        """获取对应模式的图（每次为特定conversation_id创建独立实例）"""
        # 构建基础图（未编译）
        if mode == "chat":
            graph_builder = self._build_chat_graph()
        elif mode == "rag":
            graph_builder = self._build_rag_graph()
        elif mode == "agent":
            graph_builder = self._build_agent_graph()
        elif mode == "search":
            graph_builder = self._build_search_graph()
        elif mode == "deepresearch":
            graph_builder = self._build_deepresearch_graph()
        else:
            # 默认使用聊天模式
            graph_builder = self._build_chat_graph()
        
        # 每次都重新编译图以确保checkpointer正确绑定
        try:
            checkpointer = await get_checkpointer(conversation_id)
            # 编译图并添加checkpointer
            compiled_graph = graph_builder.compile(checkpointer=checkpointer)
            logger.debug(f"为对话 {conversation_id} 创建了带checkpointer的图")
            return compiled_graph
        except Exception as e:
            # 如果checkpointer失败，编译不带checkpointer的图
            logger.warning(f"Warning: 无法创建checkpointer，使用基础图: {e}")
            return graph_builder.compile()
    
    async def process_conversation(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        mode: str = "chat",
        system_prompt: Optional[str] = None,
        retrieved_documents: Optional[List[str]] = None,
        available_tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[UUID] = None,
    ) -> AsyncGenerator[str, None]:
        """
        使用 LangGraph 处理对话，支持 checkpointer 状态持久化
        
        Args:
            messages: 消息历史
            model_config: 模型配置
            mode: 处理模式 (chat/rag/agent)
            system_prompt: 系统提示（如果不提供，将使用对应模式的默认提示词）
            retrieved_documents: 检索到的文档（RAG模式）
            available_tools: 可用工具（Agent模式）
            metadata: 额外元数据
            conversation_id: 对话ID（用于checkpointer）
        """
        try:
            # 获取对应模式的图
            graph = await self._get_graph(mode, conversation_id)
            
            # 转换消息格式
            langchain_messages = convert_messages_to_langchain(messages)
            
            # 构建初始状态
            initial_state: ConversationState = {
                "messages": langchain_messages,
                "model_config": model_config,
                "system_prompt": system_prompt,
                "mode": mode,
                "retrieved_documents": retrieved_documents,
                "available_tools": available_tools,
                "metadata": metadata or {},
                "user_query": messages[-1]["content"] if messages else None,
                "final_response": None,
                "conversation_id": str(conversation_id) if conversation_id else None,
            }
            
            # 准备图执行配置
            config = {}
            if conversation_id:
                config = get_conversation_config(conversation_id)
            
            # 执行图
            try:
                result = await graph.ainvoke(initial_state, config=config)
                
                # 提取最终响应
                final_response = result.get("final_response", "")
                result_metadata = result.get("metadata", {})
                
                if final_response:
                    # 生成流式响应
                    chunk_dict = {
                        "content": final_response,
                        "done": True,
                        "error": False,
                        "message": None,
                        "metadata": result_metadata
                    }
                    
                    # 添加RAG模式的来源信息
                    if mode == "rag" and result_metadata.get("sources"):
                        chunk_dict["sources"] = result_metadata["sources"]
                    
                    # 添加Agent模式的工具使用信息
                    if mode == "agent":
                        chunk_dict["is_tool_use"] = result_metadata.get("is_tool_use", False)
                    
                    yield json.dumps(chunk_dict)
                else:
                    # 如果没有最终响应，返回错误
                    yield json.dumps({
                        "content": "生成响应时发生错误",
                        "done": True,
                        "error": True,
                        "message": "No response generated",
                        "metadata": {}
                    })
                    
            except Exception as e:
                # 图执行错误
                yield json.dumps({
                    "content": f"处理对话时发生错误: {str(e)}",
                    "done": True,
                    "error": True,
                    "message": str(e),
                    "metadata": {}
                })
                
        except Exception as e:
            # 整体错误处理
            yield json.dumps({
                "content": f"初始化对话处理时发生错误: {str(e)}",
                "done": True,
                "error": True,
                "message": str(e),
                "metadata": {}
            })
    
    # 保持向后兼容的方法
    async def process_chat(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """向后兼容的聊天处理方法"""
        async for chunk in self.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="chat",
            system_prompt=system_prompt
        ):
            yield chunk
    
    async def process_rag(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        retrieved_documents: List[str],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """向后兼容的RAG处理方法"""
        async for chunk in self.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="rag",
            system_prompt=system_prompt,
            retrieved_documents=retrieved_documents
        ):
            yield chunk
    
    async def process_agent(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        available_tools: List[str],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """向后兼容的Agent处理方法"""
        async for chunk in self.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="agent",
            system_prompt=system_prompt,
            available_tools=available_tools
        ):
            yield chunk
    
    async def process_search(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """向后兼容的搜索处理方法"""
        async for chunk in self.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="search",
            system_prompt=system_prompt
        ):
            yield chunk
    
    async def process_deepresearch(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """向后兼容的深度研究处理方法"""
        async for chunk in self.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="deepresearch",
            system_prompt=system_prompt
        ):
            yield chunk
    
    def clear_model_cache(self):
        """清除模型缓存"""
        self._model_cache.clear()
        self._graphs.clear()
    
    async def clear_conversation_state(self, conversation_id: UUID):
        """清除特定对话的状态"""
        from app.llm.core.checkpointer import clear_conversation_checkpoint
        await clear_conversation_checkpoint(conversation_id)
    
    def get_cached_models(self) -> List[str]:
        """获取已缓存的模型列表"""
        return list(self._model_cache.keys())
    
    def get_cached_graphs(self) -> List[str]:
        """获取已缓存的图列表"""
        return list(self._graphs.keys())

    async def estimate_tokens(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        估算token使用量
        
        Args:
            messages: 消息历史
            model_config: 模型配置
            system_prompt: 系统提示
            
        Returns:
            token使用情况
        """
        # 获取模型实例
        model = self._get_model(model_config)
        
        # 构建消息列表
        final_messages = []
        if system_prompt:
            final_messages.append(SystemMessage(content=system_prompt))
        
        # 转换并添加历史消息
        langchain_messages = convert_messages_to_langchain(messages)
        final_messages.extend(langchain_messages)
        
        # 使用LangChain的token计算功能
        try:
            # 大多数LangChain模型都有get_num_tokens方法
            if hasattr(model, 'get_num_tokens_from_messages'):
                token_count = model.get_num_tokens_from_messages(final_messages)
            elif hasattr(model, 'get_num_tokens'):
                # 如果没有专门的消息token计算方法，尝试转换为文本
                text = "\n".join([msg.content for msg in final_messages])
                token_count = model.get_num_tokens(text)
            else:
                # 简单估算：每4个字符约1个token
                text = "\n".join([msg.content for msg in final_messages])
                token_count = len(text) // 4
        except Exception:
            # 回退到简单估算
            text = "\n".join([msg.content for msg in final_messages])
            token_count = len(text) // 4
        
        return {
            "prompt_tokens": token_count,
            "max_tokens": model_config.get("max_tokens", 4000),
            "available_tokens": max(0, model_config.get("max_tokens", 4000) - token_count),
        } 
    
    
    async def process_file_to_documents(
        self, 
        file_path: str, 
        file_type: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> Tuple[List[Document], Dict]:
        """
        处理文件内容并返回分割后的Document对象列表
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            user_id: 用户ID（用于元数据）
            file_id: 文件ID（用于元数据）
            file_name: 原始文件名（用于元数据）
            
        Returns:
            (分割后的Document对象列表, 文件元数据)
        """
        try:
            # 验证文件类型
            if not self.file_mgr.validate_file_type(file_type):
                raise FileProcessingException(detail=f"不支持的文件类型: {file_type}")
            
            # 加载文档
            documents = await self.file_mgr.load_documents(file_path, file_type)
            
            if not documents:
                raise FileProcessingException(detail="无法从文件中提取内容")
            
            # 收集原始文档元数据
            metadata = self.file_mgr.collect_metadata(documents, file_type)
            
            # 分割文档
            split_documents = self.file_mgr.split_documents(documents, chunk_size, chunk_overlap)
            
            # 为每个分割的文档添加隔离和追踪元数据
            enhanced_documents = []
            for i, doc in enumerate(split_documents):
                enhanced_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(split_documents),
                }
                
                # 添加隔离相关的元数据
                if user_id:
                    enhanced_metadata["user_id"] = user_id
                if file_id:
                    enhanced_metadata["file_id"] = file_id
                if file_name:
                    enhanced_metadata["original_filename"] = file_name
                    enhanced_metadata["source"] = file_name
                
                # 添加处理时间戳
                import datetime
                enhanced_metadata["processed_at"] = datetime.datetime.now().isoformat()
                enhanced_metadata["file_type"] = file_type
                
                # 过滤复杂的元数据类型，只保留向量数据库支持的基础类型
                filtered_metadata = {
                    k: v for k, v in enhanced_metadata.items() 
                    if isinstance(v, (str, bool, int, float)) and v is not None
                }
                
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                )
                enhanced_documents.append(enhanced_doc)
            
            # 计算统计信息
            total_content = "\n\n".join([doc.page_content for doc in documents])
            metadata.update({
                "original_document_count": len(documents),
                "split_document_count": len(split_documents),
                "total_character_count": len(total_content),
                "chunk_size": chunk_size or getattr(self, 'default_chunk_size', 1000),
                "chunk_overlap": chunk_overlap or getattr(self, 'default_chunk_overlap', 200),
                "user_id": user_id,
                "file_id": file_id,
                "file_name": file_name,
                "file_type": file_type
            })
            
            logger.info(f"文件处理成功: {file_name or file_path}, 分割为 {len(enhanced_documents)} 个文档块")
            
            return enhanced_documents, metadata
            
        except Exception as e:
            logger.error(f"文件处理失败: {str(e)}")
            raise FileProcessingException(detail=f"文件处理失败: {str(e)}")
    