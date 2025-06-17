import json
from typing import AsyncGenerator, Dict, List, Optional, Any, Annotated, Literal
from uuid import UUID
from enum import Enum
from loguru import logger

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
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


class ConversationMode(str, Enum):
    """会话模式枚举"""
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"
    SEARCH = "search"
    DEEPRESEARCH = "deepresearch"


class QuestionAnalysisResult(BaseModel):
    """问题分析结果的结构化输出模型"""
    question_type: Literal["knowledge_retrieval", "document_processing", "general_chat"] = Field(
        description="问题类型：knowledge_retrieval(知识检索)、document_processing(文档处理)、general_chat(一般对话)"
    )
    processing_strategy: Literal["standard_rag", "summarization", "analysis", "translation", "direct_answer"] = Field(
        description="处理策略：standard_rag(标准RAG)、summarization(总结)、analysis(分析)、translation(翻译)、direct_answer(直接回答)"
    )
    needs_retrieval: bool = Field(
        description="是否需要检索文档内容"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="分析结果的置信度，范围0.0-1.0"
    )
    reasoning: str = Field(
        description="分析推理过程和依据"
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
    
    def __init__(self, retrieval_service=None):
        self._model_cache = {}  # 缓存已创建的模型实例
        self._graphs = {}  # 缓存不同模式的图
        self.retrieval_service = retrieval_service  # 检索服务依赖
        
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
        """构建RAG模式的状态图"""
        
        def analyze_question_node(state: ConversationState) -> Dict[str, Any]:
            """问题分析节点 - 使用LLM进行智能问题分析"""
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            model = self._get_model(state["model_config"])
            
            # 使用提示词管理器获取问题分析提示词
            analysis_prompt = prompt_manager.get_question_analysis_prompt(user_query=user_query)

            try:
                # 首先尝试使用with_structured_output进行结构化输出
                try:
                    structured_model = model.with_structured_output(QuestionAnalysisResult)
                    analysis_messages = [SystemMessage(content=analysis_prompt)]
                    analysis_result: QuestionAnalysisResult = structured_model.invoke(analysis_messages)
                    
                    logger.info(f"LLM结构化分析成功: {analysis_result.question_type} -> {analysis_result.processing_strategy} (需要检索: {analysis_result.needs_retrieval}, 置信度: {analysis_result.confidence})")
                    logger.info(f"分析推理: {analysis_result.reasoning}")
                    
                    return {
                        "metadata": {
                            **state.get("metadata", {}),
                            "question_type": analysis_result.question_type,
                            "processing_strategy": analysis_result.processing_strategy,
                            "needs_retrieval": analysis_result.needs_retrieval,
                            "analysis_confidence": analysis_result.confidence,
                            "analysis_reasoning": analysis_result.reasoning,
                            "analysis_method": "llm_structured",
                            "analysis_completed": True
                        }
                    }
                    
                except Exception as structured_error:
                    # 如果结构化输出失败，回退到JSON解析方式
                    logger.warning(f"结构化输出失败，回退到JSON解析: {str(structured_error)}")
                    
                    # 调用LLM进行问题分析（传统方式）
                    analysis_messages = [SystemMessage(content=analysis_prompt)]
                    response = model.invoke(analysis_messages)
                    analysis_text = response.content.strip()
                    
                    # 解析LLM返回的JSON结果
                    import json
                    import re
                    
                    # 尝试提取JSON部分
                    json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        analysis_data = json.loads(json_str)
                    else:
                        # 如果没有找到JSON，尝试直接解析
                        analysis_data = json.loads(analysis_text)
                    
                    # 使用Pydantic模型验证数据
                    analysis_result = QuestionAnalysisResult(**analysis_data)
                    
                    logger.info(f"LLM JSON解析成功: {analysis_result.question_type} -> {analysis_result.processing_strategy} (需要检索: {analysis_result.needs_retrieval}, 置信度: {analysis_result.confidence})")
                    logger.info(f"分析推理: {analysis_result.reasoning}")
                    
                    return {
                        "metadata": {
                            **state.get("metadata", {}),
                            "question_type": analysis_result.question_type,
                            "processing_strategy": analysis_result.processing_strategy,
                            "needs_retrieval": analysis_result.needs_retrieval,
                            "analysis_confidence": analysis_result.confidence,
                            "analysis_reasoning": analysis_result.reasoning,
                            "analysis_method": "llm_json_fallback",
                            "analysis_completed": True
                        }
                    }
                
            except Exception as e:
                logger.error(f"LLM问题分析完全失败: {str(e)}")
                
                # 分析失败，返回网络繁忙状态
                return {
                    "metadata": {
                        **state.get("metadata", {}),
                        "analysis_error": True,
                        "error_message": "网络繁忙，请稍后重试",
                        "analysis_completed": False
                    }
                }
        
        def retrieve_documents_node(state: ConversationState) -> Dict[str, Any]:
            """文档检索节点 - 执行真正的文档检索"""
            import asyncio
            
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            metadata = state.get("metadata", {})
            
            # 获取文件ID列表和用户ID
            available_file_ids = metadata.get("available_file_ids") or []
            user_id_str = metadata.get("user_id")
            
            retrieved_docs = []
            retrieval_info = {}
            
            if available_file_ids and user_id_str and self.retrieval_service:
                try:
                    from uuid import UUID
                    
                    logger.info(f"开始检索 {len(available_file_ids)} 个文件的相关内容，查询: '{user_query}'")
                    
                    # 转换字符串ID为UUID
                    file_ids = [UUID(fid) for fid in available_file_ids]
                    user_id = UUID(user_id_str)
                    
                    # 在线程池中执行异步检索操作
                    async def _async_retrieve():
                        return await self.retrieval_service.retrieve_documents(
                            query=user_query,
                            file_ids=file_ids,
                            user_id=user_id,
                            top_k=5,
                            similarity_threshold=0.3  # 使用较低的阈值以获得更多结果
                        )
                    
                    # 在新事件循环中运行异步操作
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        retrieved_docs = loop.run_until_complete(_async_retrieve())
                    finally:
                        loop.close()
                    
                    retrieval_info = {
                        "file_count": len(available_file_ids),
                        "query": user_query,
                        "document_count": len(retrieved_docs),
                        "status": "检索成功",
                        "user_id": user_id_str
                    }
                    
                    logger.info(f"文档检索成功: 找到 {len(retrieved_docs)} 个相关文档片段")
                    
                except Exception as e:
                    logger.error(f"文档检索失败: {str(e)}")
                    retrieval_info = {
                        "error": str(e),
                        "status": "检索失败",
                        "file_count": len(available_file_ids),
                        "query": user_query,
                        "user_id": user_id_str
                    }
            else:
                missing_items = []
                if not available_file_ids:
                    missing_items.append("file_ids")
                if not user_id_str:
                    missing_items.append("user_id")
                if not self.retrieval_service:
                    missing_items.append("retrieval_service")
                
                retrieval_info = {
                    "status": f"检索条件不满足，缺少: {', '.join(missing_items)}",
                    "file_count": len(available_file_ids),
                    "has_service": bool(self.retrieval_service),
                    "has_user_id": bool(user_id_str),
                    "query": user_query
                }
                logger.warning(f"检索节点：检索条件不满足，缺少: {', '.join(missing_items)}")
            
            return {
                "retrieved_documents": retrieved_docs,
                "metadata": {
                    **state.get("metadata", {}),
                    "retrieval_completed": True,
                    "retrieval_info": retrieval_info,
                    "document_count": len(retrieved_docs)
                }
            }
        
        def rag_response_node(state: ConversationState) -> Dict[str, Any]:
            """RAG响应节点 - 根据检索结果和用户问题生成回答"""
            model = self._get_model(state["model_config"])
            messages = state.get("messages") or []
            user_query = state.get("user_query") or (messages[-1].content if messages else "")
            retrieved_docs = state.get("retrieved_documents") or []
            metadata = state.get("metadata", {})
            
            processing_strategy = metadata.get("processing_strategy", "standard_rag")
            needs_retrieval = metadata.get("needs_retrieval", True)
            
            # 构建系统提示
            system_prompt = state.get("system_prompt")
            if not system_prompt:
                system_prompt = prompt_manager.get_rag_prompt()
            
            # 根据处理策略调整系统提示
            if processing_strategy == "direct_answer":
                system_prompt += "\n\n## 当前任务\n请基于常识直接回答用户问题，不需要检索文档。"
            elif retrieved_docs:
                if processing_strategy == "summarization":
                    system_prompt += "\n\n## 当前任务\n请对提供的文档内容进行总结。"
                elif processing_strategy == "analysis":
                    system_prompt += "\n\n## 当前任务\n请对提供的文档内容进行深度分析。"
                elif processing_strategy == "translation":
                    system_prompt += "\n\n## 当前任务\n请对提供的文档内容进行翻译。"
                else:
                    system_prompt += "\n\n## 当前任务\n请基于检索到的文档内容回答用户问题。"
            else:
                system_prompt += "\n\n## 当前任务\n没有找到相关文档，请基于常识回答用户问题。"
            
            # 构建消息列表
            final_messages = [SystemMessage(content=system_prompt)]
            
            # 如果有检索到的文档，添加文档上下文
            if retrieved_docs:
                if processing_strategy in ["summarization", "analysis", "translation"]:
                    # 文档处理模式：提供完整文档内容
                    context = "\n\n=== 文档分隔符 ===\n\n".join(retrieved_docs)
                    context_message = f"需要处理的文档内容：\n\n{context}"
                else:
                    # 标准RAG模式：提供相关片段
                    context = "\n\n".join([f"文档片段 {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
                    context_message = f"相关文档内容：\n\n{context}"
                
                final_messages.append(SystemMessage(content=context_message))
            
            # 添加用户消息
            final_messages.extend(messages)
            
            # 调用模型生成回答
            try:
                response = model.invoke(final_messages)
                
                # 准备来源信息
                sources = []
                if retrieved_docs:
                    sources = [
                        {
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                            "index": i
                        } 
                        for i, doc in enumerate(retrieved_docs)
                    ]
                
                # 准备响应元数据
                response_metadata = {
                    "mode": "rag",
                    "processing_strategy": processing_strategy,
                    "needs_retrieval": needs_retrieval,
                    "document_count": len(retrieved_docs),
                    "sources": sources
                }
                
                # 添加处理类型标识
                if processing_strategy == "summarization":
                    response_metadata["processing_type"] = "总结"
                elif processing_strategy == "analysis":
                    response_metadata["processing_type"] = "分析"
                elif processing_strategy == "translation":
                    response_metadata["processing_type"] = "翻译"
                elif processing_strategy == "direct_answer":
                    response_metadata["processing_type"] = "常识回答"
                else:
                    response_metadata["processing_type"] = "知识检索"
                
                return {
                    "messages": [response],
                    "final_response": response.content,
                    "metadata": response_metadata
                }
                
            except Exception as e:
                logger.error(f"RAG响应生成失败: {str(e)}")
                return {
                    "final_response": f"生成回答时出错: {str(e)}",
                    "metadata": {
                        "mode": "rag",
                        "error": True,
                        "error_type": "response_generation_failed",
                        "error_message": str(e)
                    }
                }
        
        def error_response_node(state: ConversationState) -> Dict[str, Any]:
            """错误响应节点 - 处理分析失败的情况"""
            error_message = state.get("metadata", {}).get("error_message", "网络繁忙，请稍后重试")
            
            return {
                "final_response": error_message,
                "metadata": {
                    "mode": "rag",
                    "error": True,
                    "error_type": "analysis_failed",
                    "error_message": error_message
                }
            }
        
        def route_after_analysis(state: ConversationState) -> str:
            """分析后的路由节点 - 决定是否需要检索"""
            metadata = state.get("metadata", {})
            
            # 检查分析是否失败
            if metadata.get("analysis_error", False):
                return "error_response"
            
            needs_retrieval = metadata.get("needs_retrieval", True)
            available_file_ids = metadata.get("available_file_ids", [])
            
            # 如果不需要检索或没有可用文件，直接回答
            if not needs_retrieval or not available_file_ids:
                return "rag_response"
            
            # 需要检索且有可用文件，进入检索节点
            return "retrieve_documents"
        
        # 构建图但不编译
        graph_builder = StateGraph(ConversationState)
        
        # 添加节点
        graph_builder.add_node("analyze_question", analyze_question_node)
        graph_builder.add_node("retrieve_documents", retrieve_documents_node)
        graph_builder.add_node("rag_response", rag_response_node)
        graph_builder.add_node("error_response", error_response_node)
        
        # 添加边
        graph_builder.add_edge(START, "analyze_question")
        
        # 添加条件路由：分析后决定是否检索
        graph_builder.add_conditional_edges(
            "analyze_question",
            route_after_analysis,
            {
                "retrieve_documents": "retrieve_documents",
                "rag_response": "rag_response", 
                "error_response": "error_response"
            }
        )
        
        # 检索完成后直接生成回答
        graph_builder.add_edge("retrieve_documents", "rag_response")
        graph_builder.add_edge("rag_response", END)
        graph_builder.add_edge("error_response", END)
        
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