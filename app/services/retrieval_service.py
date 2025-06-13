import json
import asyncio
from typing import List, Dict, Optional, Any
from uuid import UUID

from loguru import logger

from app.db.repositories.user_file_repository import UserFileRepository
from app.llm.rag.retrieval_service import LLMRetrievalService


class RetrievalService:
    """
    业务层检索服务
    专门处理文档检索相关的业务逻辑，协调数据库操作和LLM检索服务
    支持多种嵌入模型提供商（OpenAI、Qwen等）
    支持基于LLM的智能意图识别和检索策略
    """
    
    def __init__(self, file_repo: UserFileRepository, embedding_provider: Optional[str] = None):
        self.file_repo = file_repo
        
        # 初始化LLM检索服务，支持指定嵌入模型提供商
        self.llm_retrieval = LLMRetrievalService(embedding_provider=embedding_provider)
        
        logger.info(f"RetrievalService 初始化完成，使用嵌入模型: {self.llm_retrieval.embedding_provider}")
    
    async def _execute_retrieval_strategy(
        self,
        search_queries: List[str],
        strategy: str,
        file_ids: List[UUID],
        top_k: int,
        similarity_threshold: float
    ) -> Dict[str, Any]:
        """
        执行具体的检索策略
        """
        all_results = []
        strategy_details = []
        
        if strategy == "keyword_search":
            # 关键词检索：提高相似度阈值，专注精确匹配
            threshold = max(similarity_threshold, 0.5)
            top_per_query = max(2, top_k // len(search_queries))
        elif strategy == "semantic_search":
            # 语义检索：使用标准阈值
            threshold = similarity_threshold
            top_per_query = max(3, top_k // len(search_queries))
        elif strategy == "multi_query":
            # 多查询检索：降低阈值，增加召回
            threshold = max(0.2, similarity_threshold - 0.1)
            top_per_query = max(2, top_k // len(search_queries))
        else:  # topic_search
            # 主题检索：平衡精度和召回
            threshold = similarity_threshold
            top_per_query = max(3, top_k // len(search_queries))
        
        for query in search_queries:
            logger.info(f"执行检索 [{strategy}]: {query}")
            
            search_results = await self.llm_retrieval.search_documents(
                query=query,
                file_ids=[str(file_id) for file_id in file_ids],
                top_k=top_per_query,
                similarity_threshold=threshold
            )
            
            strategy_details.append({
                "query": query,
                "strategy": strategy,
                "threshold": threshold,
                "results_count": len(search_results),
                "avg_similarity": sum(r.get("similarity_score", 0) for r in search_results) / max(1, len(search_results))
            })
            
            all_results.extend(search_results)
        
        # 去重和排序
        unique_results = []
        seen_content = set()
        for result in all_results:
            content = result["content"]
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append(result)
        
        # 根据策略调整排序方式
        if strategy == "keyword_search":
            # 关键词检索优先考虑相似度
            unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        elif strategy == "topic_search":
            # 主题检索考虑相似度和多样性
            unique_results.sort(key=lambda x: (x.get("similarity_score", 0), -x.get("distance", 0)), reverse=True)
        else:
            # 默认按相似度排序
            unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        final_results = unique_results[:top_k]
        
        return {
            "documents": [r["content"] for r in final_results],
            "detailed_results": final_results,
            "info": {
                "strategy": strategy,
                "queries_used": len(search_queries),
                "total_results_found": len(all_results),
                "unique_results": len(unique_results),
                "final_count": len(final_results),
                "strategy_details": strategy_details
            }
        }
    
    async def retrieve_documents(
        self,
        query: str,
        file_ids: List[UUID],
        user_id: UUID,
        top_k: int = 5,
        similarity_threshold: float = 0.3,  # 降低默认阈值
        enable_query_rewrite: bool = True  # 启用查询改写
    ) -> List[str]:
        """
        根据查询检索相关文档
        
        Args:
            query: 查询文本
            file_ids: 文件ID列表
            user_id: 用户ID（用于权限验证）
            top_k: 返回文档数量
            similarity_threshold: 相似度阈值
            enable_query_rewrite: 是否启用智能查询改写
            
        Returns:
            相关文档内容列表
        """
        try:
            # 验证用户对文件的访问权限
            valid_file_ids = await self._validate_file_access(file_ids, user_id)
            
            if not valid_file_ids:
                logger.warning(f"用户 {user_id} 没有权限访问任何指定的文件")
                return []
            
            # 获取文件上下文信息用于查询改写
            file_context = await self._get_file_context(valid_file_ids, user_id)
            
            # 智能查询改写
            retrieval_queries = [query]  # 默认使用原始查询
            if enable_query_rewrite:
                rewrite_result = await self.query_rewriter.rewrite_query(
                    original_query=query,
                    use_llm=True,
                    file_context=file_context
                )
                
                if rewrite_result["query_type"] == "instruction":
                    retrieval_queries = rewrite_result["rewritten_queries"]
                    logger.info(f"检测到指令性查询，改写为: {retrieval_queries}")
                else:
                    logger.info(f"检测到具体查询，直接使用原查询: {query}")
            
            # 对每个改写后的查询进行检索
            all_results = []
            seen_content = set()  # 去重
            
            for search_query in retrieval_queries:
                logger.info(f"使用查询进行检索: {search_query}")
                
                search_results = await self.llm_retrieval.search_documents(
                    query=search_query,
                    file_ids=[str(file_id) for file_id in valid_file_ids],
                    top_k=max(3, top_k // len(retrieval_queries)),  # 分配检索数量
                    similarity_threshold=similarity_threshold
                )
                
                # 去重并添加到结果
                for result in search_results:
                    content = result["content"]
                    if content not in seen_content:
                        seen_content.add(content)
                        all_results.append(result)
            
            # 按相似度分数排序并截取前top_k个
            all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            final_results = all_results[:top_k]
            
            # 提取文档内容
            retrieved_docs = [result["content"] for result in final_results]
            
            logger.info(f"为用户 {user_id} 检索到 {len(retrieved_docs)} 个相关文档")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return []
    
    async def retrieve_documents_with_enhanced_strategy(
        self,
        query: str,
        file_ids: List[UUID],
        user_id: UUID,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        增强的文档检索，返回详细的检索策略信息
        
        Returns:
            包含documents, strategy_info, query_analysis等的详细结果
        """
        try:
            # 验证用户对文件的访问权限
            valid_file_ids = await self._validate_file_access(file_ids, user_id)
            
            if not valid_file_ids:
                return {
                    "documents": [],
                    "strategy_info": {"error": "无权限访问文件"},
                    "query_analysis": {}
                }
            
            # 获取文件上下文
            file_context = await self._get_file_context(valid_file_ids, user_id)
            
            # 查询分析和改写
            rewrite_result = await self.query_rewriter.rewrite_query(
                original_query=query,
                use_llm=True,
                file_context=file_context
            )
            
            # 执行检索
            all_results = []
            strategy_details = []
            
            for i, search_query in enumerate(rewrite_result["rewritten_queries"]):
                search_results = await self.llm_retrieval.search_documents(
                    query=search_query,
                    file_ids=[str(file_id) for file_id in valid_file_ids],
                    top_k=3,
                    similarity_threshold=similarity_threshold
                )
                
                strategy_details.append({
                    "query": search_query,
                    "results_count": len(search_results),
                    "avg_similarity": sum(r.get("similarity_score", 0) for r in search_results) / max(1, len(search_results))
                })
                
                all_results.extend(search_results)
            
            # 去重和排序
            unique_results = []
            seen_content = set()
            for result in all_results:
                if result["content"] not in seen_content:
                    seen_content.add(result["content"])
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            final_results = unique_results[:top_k]
            
            return {
                "documents": [r["content"] for r in final_results],
                "detailed_results": final_results,
                "strategy_info": {
                    "query_type": rewrite_result["query_type"],
                    "rewrite_strategy": rewrite_result["strategy"],
                    "confidence": rewrite_result["confidence"],
                    "queries_used": len(rewrite_result["rewritten_queries"]),
                    "total_results_found": len(all_results),
                    "unique_results": len(unique_results),
                    "strategy_details": strategy_details
                },
                "query_analysis": rewrite_result,
                "file_context": file_context
            }
            
        except Exception as e:
            logger.error(f"增强文档检索失败: {str(e)}")
            return {
                "documents": [],
                "strategy_info": {"error": str(e)},
                "query_analysis": {}
            }
    
    async def _get_file_context(self, file_ids: List[UUID], user_id: UUID) -> Dict[str, Any]:
        """获取文件上下文信息"""
        file_context = {
            "file_names": [],
            "file_types": [],
            "file_count": len(file_ids)
        }
        
        for file_id in file_ids:
            try:
                file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
                if file_record:
                    file_context["file_names"].append(file_record.original_filename)
                    file_context["file_types"].append(file_record.file_type)
            except Exception as e:
                logger.warning(f"获取文件 {file_id} 上下文信息失败: {str(e)}")
        
        return file_context
    
    async def retrieve_documents_with_metadata(
        self,
        query: str,
        file_ids: List[UUID],
        user_id: UUID,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        检索文档并返回完整的元数据信息
        
        Args:
            query: 查询文本
            file_ids: 文件ID列表
            user_id: 用户ID（用于权限验证）
            top_k: 返回文档数量
            similarity_threshold: 相似度阈值
            
        Returns:
            包含文档内容、元数据和相似度分数的字典列表
        """
        try:
            # 使用增强策略进行检索
            enhanced_result = await self.retrieve_documents_with_enhanced_strategy(
                query=query,
                file_ids=file_ids,
                user_id=user_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # 增强元数据信息（添加文件信息等）
            if enhanced_result.get("detailed_results"):
                enhanced_results = await self._enhance_search_results(
                    enhanced_result["detailed_results"], user_id
                )
                logger.info(f"为用户 {user_id} 检索到 {len(enhanced_results)} 个相关文档（包含元数据）")
                return enhanced_results
            else:
                return []
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return []
    
    async def add_documents(
        self,
        file_id: UUID,
        user_id: UUID,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        添加文档到向量存储
        
        Args:
            file_id: 文件ID
            user_id: 用户ID（用于权限验证）
            documents: 文档内容列表
            metadata_list: 文档元数据列表
            
        Returns:
            是否成功添加
        """
        try:
            # 验证用户对文件的访问权限
            file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
            if not file_record:
                logger.error(f"用户 {user_id} 没有权限访问文件 {file_id}")
                return False
            
            # 构建元数据
            if not metadata_list:
                metadata_list = []
            
            # 确保每个元数据都包含必要的信息
            enhanced_metadata_list = []
            for i, metadata in enumerate(metadata_list if metadata_list else [{}] * len(documents)):
                enhanced_metadata = {
                    "file_id": str(file_id),
                    "user_id": str(user_id),
                    "file_name": file_record.original_name,
                    "file_type": file_record.file_type,
                    "chunk_index": i,
                    **metadata
                }
                enhanced_metadata_list.append(enhanced_metadata)
            
            # 调用LLM检索服务添加文档
            success = await self.llm_retrieval.add_documents_to_vector_store(
                documents=documents,
                metadatas=enhanced_metadata_list
            )
            
            if success:
                logger.info(f"成功为用户 {user_id} 添加文件 {file_id} 的 {len(documents)} 个文档块到向量存储")
            
            return success
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    async def remove_documents(self, file_id: UUID, user_id: UUID) -> bool:
        """
        从向量存储中移除指定文件的文档
        
        Args:
            file_id: 文件ID
            user_id: 用户ID（用于权限验证）
            
        Returns:
            是否成功移除
        """
        try:
            # 验证用户对文件的访问权限
            file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
            if not file_record:
                logger.error(f"用户 {user_id} 没有权限访问文件 {file_id}")
                return False
            
            # 调用LLM检索服务删除文档
            success = await self.llm_retrieval.remove_documents_from_vector_store(str(file_id))
            
            if success:
                logger.info(f"成功为用户 {user_id} 从向量存储删除文件 {file_id} 的文档")
            
            return success
            
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    
    async def split_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        文本分块（业务层包装）
        
        Args:
            text: 原始文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块后的文本列表
        """
        try:
            return await self.llm_retrieval.split_text_into_chunks(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except Exception as e:
            logger.error(f"文本分块失败: {str(e)}")
            return [text]
    
    async def _validate_file_access(self, file_ids: List[UUID], user_id: UUID) -> List[UUID]:
        """
        验证用户对文件的访问权限
        
        Args:
            file_ids: 文件ID列表
            user_id: 用户ID
            
        Returns:
            用户有权限访问的文件ID列表
        """
        valid_file_ids = []
        
        for file_id in file_ids:
            try:
                file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
                if file_record:
                    valid_file_ids.append(file_id)
                else:
                    logger.warning(f"用户 {user_id} 无权访问文件 {file_id}")
            except Exception as e:
                logger.error(f"验证文件 {file_id} 访问权限时出错: {str(e)}")
        
        return valid_file_ids
    
    async def _enhance_search_results(
        self, 
        search_results: List[Dict[str, Any]], 
        user_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        增强搜索结果，添加额外的文件信息
        
        Args:
            search_results: LLM检索服务返回的搜索结果
            user_id: 用户ID
            
        Returns:
            增强后的搜索结果
        """
        enhanced_results = []
        
        for result in search_results:
            try:
                # 从元数据中获取文件ID
                file_id_str = result.get("metadata", {}).get("file_id")
                if file_id_str:
                    file_id = UUID(file_id_str)
                    
                    # 获取文件详细信息
                    file_record = await self.file_repo.get_by_id_for_user(file_id, user_id)
                    if file_record:
                        # 添加文件详细信息到结果中
                        enhanced_result = {
                            **result,
                            "file_info": {
                                "id": str(file_record.id),
                                "name": file_record.original_name,
                                "type": file_record.file_type,
                                "size": file_record.file_size,
                                "upload_time": file_record.created_at.isoformat() if file_record.created_at else None
                            }
                        }
                        enhanced_results.append(enhanced_result)
                    else:
                        # 如果无法获取文件信息，仍然保留原始结果
                        enhanced_results.append(result)
                else:
                    # 如果没有文件ID信息，保留原始结果
                    enhanced_results.append(result)
                    
            except Exception as e:
                logger.error(f"增强搜索结果时出错: {str(e)}")
                # 出错时保留原始结果
                enhanced_results.append(result)
        
        return enhanced_results
    
    async def get_embedding_info(self) -> Dict[str, Any]:
        """
        获取当前嵌入模型信息
        
        Returns:
            嵌入模型信息
        """
        try:
            return self.llm_retrieval.get_embedding_info()
        except Exception as e:
            logger.error(f"获取嵌入模型信息失败: {str(e)}")
            return {"error": str(e)}
    
    async def test_embedding_connection(self) -> Dict[str, Any]:
        """
        测试嵌入模型连接状态
        
        Returns:
            连接测试结果
        """
        try:
            return await self.llm_retrieval.test_embedding_connection()
        except Exception as e:
            logger.error(f"测试嵌入模型连接失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "嵌入模型连接测试失败"
            }
    
    @classmethod
    def create_with_provider(
        cls, 
        file_repo: UserFileRepository, 
        embedding_provider: str
    ) -> "RetrievalService":
        """
        使用指定的嵌入模型提供商创建检索服务实例
        
        Args:
            file_repo: 文件仓库实例
            embedding_provider: 嵌入模型提供商
            
        Returns:
            检索服务实例
        """
        return cls(file_repo=file_repo, embedding_provider=embedding_provider)
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """
        获取支持的嵌入模型提供商列表
        
        Returns:
            支持的提供商列表
        """
        return LLMRetrievalService.get_supported_providers() 