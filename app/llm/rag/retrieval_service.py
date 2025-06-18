import os
from typing import Any, Dict, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.core.config import settings


class LLMRetrievalService:
    """
    LLM检索服务
    使用标准化的langchain_chroma接口实现向量存储和文档检索
    支持多种嵌入模型提供商：OpenAI、Qwen等
    """
    
    def __init__(self, embedding_provider: str = None):
        # 确定使用的嵌入模型提供商
        self.embedding_provider = embedding_provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
        
        # 初始化嵌入模型
        self.embeddings = self._initialize_embedding_model()
        
        # 初始化向量存储
        self.vector_store = None
        self._initialize_vector_store()
    
    @property
    def is_ready(self) -> bool:
        """检查向量存储是否准备就绪"""
        return self.vector_store is not None
    
    def _initialize_embedding_model(self) -> Embeddings:
        """
        根据配置初始化嵌入模型
        
        Returns:
            嵌入模型实例
        """
        try:
            if self.embedding_provider.lower() == 'openai':
                logger.info("使用 OpenAI 嵌入模型")
                return OpenAIEmbeddings(
                    model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                    openai_api_key=settings.OPENAI_API_KEY
                )
                
            elif self.embedding_provider.lower() == 'qwen':
                logger.info("使用 Qwen (DashScope) 嵌入模型")
                try:
                    from langchain_community.embeddings import DashScopeEmbeddings
                    
                    return DashScopeEmbeddings(
                        model=getattr(settings, 'QWEN_EMBEDDING_MODEL', 'text-embedding-v1'),
                        dashscope_api_key=getattr(settings, 'QWEN_API_KEY', None) or getattr(settings, 'DASHSCOPE_API_KEY', None)
                    )
                except ImportError as e:
                    logger.error(f"无法导入 DashScopeEmbeddings: {str(e)}")
                    logger.error("请确保安装了 langchain-community 和 dashscope 库")
                    raise ImportError(
                        "DashScope 嵌入需要安装以下依赖:\n"
                        "pip install langchain-community dashscope"
                    )
                
            else:
                logger.warning(f"不支持的嵌入模型提供商: {self.embedding_provider}，回退到 OpenAI")
                return OpenAIEmbeddings(
                    model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                    openai_api_key=settings.OPENAI_API_KEY
                )
                
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            # 回退到 OpenAI 作为默认选项
            return OpenAIEmbeddings(
                model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                openai_api_key=settings.OPENAI_API_KEY
            )
    
    def _initialize_vector_store(self):
        """
        初始化向量存储
        """
        try:
            if settings.VECTOR_DB_TYPE == "chroma":
                # 确保存储目录存在
                os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
                
                # 使用标准化的langchain_chroma接口
                self.vector_store = Chroma(
                    persist_directory=settings.CHROMA_DB_DIR,
                    embedding_function=self.embeddings,
                    collection_name="documents"  # 指定集合名称
                )
                logger.info(f"向量存储初始化成功，使用 {self.embedding_provider} 嵌入模型")
            else:
                logger.error(f"不支持的向量数据库类型: {settings.VECTOR_DB_TYPE}")
                
        except Exception as e:
            logger.error(f"向量存储初始化失败: {str(e)}")
            self.vector_store = None
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        使用LangChain标准接口进行相似度搜索（带分数）
        
        Args:
            query: 查询文本
            k: 返回文档数量
            filter_dict: 过滤条件字典
            user_id: 用户ID，用于过滤用户文档
            file_ids: 文件ID列表，用于过滤特定文件
            conversation_id: 对话ID，用于过滤对话文档
            **kwargs: 其他参数
            
        Returns:
            (Document, score)元组列表
        """
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
            
            # 构建过滤条件
            final_filter = self._build_filter_condition(
                filter_dict, user_id, file_ids, conversation_id
            )
            
            # 使用LangChain标准的异步相似度搜索
            results = await self.vector_store.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=final_filter,
                **kwargs
            )
            
            logger.info(f"相似度搜索完成，返回 {len(results)} 个结果，过滤条件: {final_filter}")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        使用LangChain标准接口进行相似度搜索
        
        Args:
            query: 查询文本  
            k: 返回文档数量
            filter_dict: 过滤条件字典
            user_id: 用户ID，用于过滤用户文档
            file_ids: 文件ID列表，用于过滤特定文件
            conversation_id: 对话ID，用于过滤对话文档
            **kwargs: 其他参数
            
        Returns:
            Document列表
        """
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
            
            # 构建过滤条件
            final_filter = self._build_filter_condition(
                filter_dict, user_id, file_ids, conversation_id
            )
            
            # 使用LangChain标准的异步相似度搜索
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                filter=final_filter,
                **kwargs
            )
            
            logger.info(f"相似度搜索完成，返回 {len(results)} 个文档，过滤条件: {final_filter}")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []
    
    def _build_filter_condition(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        构建过滤条件 - 兼容Chroma数据库的where语法
        
        Args:
            filter_dict: 基础过滤条件
            user_id: 用户ID
            file_ids: 文件ID列表
            conversation_id: 对话ID
            
        Returns:
            合并后的过滤条件
        """
        conditions = []
        
        # 处理基础过滤条件
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, dict):
                    conditions.append({key: value})
                else:
                    conditions.append({key: {"$eq": value}})
        
        # 用户ID过滤（安全级别最高）
        if user_id:
            conditions.append({"user_id": {"$eq": user_id}})
        
        # 文件ID过滤（优先级：特定文件 > 对话文件 > 用户所有文件）
        if file_ids:
            if len(file_ids) == 1:
                conditions.append({"file_id": {"$eq": file_ids[0]}})
            else:
                conditions.append({"file_id": {"$in": file_ids}})
        elif conversation_id:
            # 如果没有指定特定文件，但有对话ID，需要查询该对话关联的文件
            conditions.append({"conversation_id": {"$eq": conversation_id}})
        
        # 构建最终的过滤条件
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            # 多个条件使用$and操作符
            return {"$and": conditions}
    
    async def add_documents(
        self,
        documents: List[Document],
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        使用LangChain标准接口添加文档到向量存储
        
        Args:
            documents: Document对象列表
            user_id: 用户ID
            file_id: 文件ID
            conversation_id: 对话ID
            **kwargs: 其他参数
            
        Returns:
            是否成功添加
        """
        try:
            if not self.vector_store or not documents:
                logger.warning("向量存储未初始化或文档列表为空")
                return False
            
            # 过滤和增强元数据
            enhanced_documents = []
            for doc in documents:
                enhanced_metadata = {
                    **doc.metadata,
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": getattr(self.embeddings, 'model', 'unknown')
                }
                
                # 添加隔离相关的元数据
                if user_id:
                    enhanced_metadata["user_id"] = user_id
                if file_id:
                    enhanced_metadata["file_id"] = file_id
                if conversation_id:
                    enhanced_metadata["conversation_id"] = conversation_id
                
                # 添加时间戳用于清理和审计
                import datetime
                enhanced_metadata["indexed_at"] = datetime.datetime.now().isoformat()
                
                # 过滤复杂的元数据类型（Chroma只支持基础类型）
                filtered_metadata = {
                    k: v for k, v in enhanced_metadata.items() 
                    if isinstance(v, (str, bool, int, float))
                }
                
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                )
                enhanced_documents.append(enhanced_doc)
            
            # 使用LangChain标准的异步添加方法
            ids = await self.vector_store.aadd_documents(enhanced_documents)
            
            logger.info(f"成功添加 {len(enhanced_documents)} 个文档到向量存储，用户: {user_id}, 文件: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            return False
    
    async def delete_documents_by_filter(
        self,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """
        根据过滤条件删除文档
        
        Args:
            user_id: 用户ID
            file_id: 文件ID  
            conversation_id: 对话ID
            
        Returns:
            是否成功删除
        """
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return False
            
            # 构建过滤条件
            filter_condition = self._build_filter_condition(
                filter_dict=None, 
                user_id=user_id, 
                file_ids=[file_id] if file_id else None, 
                conversation_id=conversation_id
            )
            
            if not filter_condition:
                logger.warning("删除操作需要提供过滤条件")
                return False
            
            # 注意：不是所有向量数据库都支持按过滤条件删除
            # 这里需要根据具体的向量数据库实现
            if hasattr(self.vector_store, 'delete'):
                # 先搜索匹配的文档ID
                matching_docs = await self.vector_store.asimilarity_search(
                    query="",  # 空查询，只用于获取匹配过滤条件的文档
                    k=10000,  # 大数值确保获取所有匹配文档
                    filter=filter_condition
                )
                
                if matching_docs:
                    # 提取文档ID并删除（假设文档有ID）
                    doc_ids = [doc.metadata.get("id") for doc in matching_docs if doc.metadata.get("id")]
                    if doc_ids:
                        self.vector_store.delete(ids=doc_ids)
                        logger.info(f"成功删除 {len(doc_ids)} 个文档，过滤条件: {filter_condition}")
                        return True
            
            logger.warning("向量存储不支持按过滤条件删除或没有找到匹配文档")
            return False
            
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    

class MultiTenantRetrievalService:
    """
    多租户检索服务 - 每个用户使用独立的向量库
    适用于高安全性要求的场景
    """
    
    def __init__(self, embedding_provider: str = None):
        self.embedding_provider = embedding_provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
        self.embeddings = self._initialize_embedding_model()
        self.vector_stores = {}  # 缓存不同用户的向量库
        
    def _get_user_vector_store(self, user_id: str):
        """获取用户专属的向量库"""
        if user_id not in self.vector_stores:
            try:
                # 用户专属的数据库目录
                user_db_dir = os.path.join(settings.CHROMA_DB_DIR, f"user_{user_id}")
                os.makedirs(user_db_dir, exist_ok=True)
                
                # 创建用户专属的向量库
                self.vector_stores[user_id] = Chroma(
                    persist_directory=user_db_dir,
                    embedding_function=self.embeddings,
                    collection_name=f"documents_user_{user_id}"
                )
                logger.info(f"创建用户 {user_id} 的专属向量库")
                
            except Exception as e:
                logger.error(f"创建用户向量库失败: {str(e)}")
                return None
                
        return self.vector_stores[user_id]
    
    async def similarity_search_with_score(
        self,
        query: str,
        user_id: str,  # 必需参数
        k: int = 5,
        file_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """多租户相似度搜索"""
        try:
            vector_store = self._get_user_vector_store(user_id)
            if not vector_store:
                return []
            
            # 可选的文件ID过滤
            filter_dict = None
            if file_ids:
                filter_dict = {"file_id": {"$in": file_ids}} if len(file_ids) > 1 else {"file_id": file_ids[0]}
            
            results = await vector_store.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict,
                **kwargs
            )
            
            logger.info(f"用户 {user_id} 相似度搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"多租户相似度搜索失败: {str(e)}")
            return []
    

class NamespaceRetrievalService:
    """
    命名空间检索服务 - 使用向量数据库的命名空间功能
    适用于支持命名空间的向量数据库（如Pinecone）
    """
    
    def __init__(self, embedding_provider: str = None):
        self.embedding_provider = embedding_provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
        self.embeddings = self._initialize_embedding_model()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _get_namespace(self, user_id: str, conversation_id: Optional[str] = None) -> str:
        """
        生成命名空间名称
        
        策略：
        - 用户级别：user_{user_id}
        - 对话级别：user_{user_id}_conv_{conversation_id}
        """
        if conversation_id:
            return f"user_{user_id}_conv_{conversation_id}"
        return f"user_{user_id}"
    
    async def similarity_search_with_score(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        conversation_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """命名空间相似度搜索"""
        try:
            if not self.vector_store:
                return []
            
            # 生成命名空间
            namespace = self._get_namespace(user_id, conversation_id)
            
            # 可选的文件ID过滤
            filter_dict = None
            if file_ids:
                filter_dict = {"file_id": {"$in": file_ids}} if len(file_ids) > 1 else {"file_id": file_ids[0]}
            
            # 根据不同的向量数据库实现命名空间搜索
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                # 对于支持命名空间的向量数据库
                results = await self.vector_store.asimilarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict,
                    namespace=namespace,  # 关键：使用命名空间
                    **kwargs
                )
            else:
                # 回退到普通过滤
                filter_dict = filter_dict or {}
                filter_dict["user_id"] = user_id
                if conversation_id:
                    filter_dict["conversation_id"] = conversation_id
                
                results = await self.vector_store.asimilarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict,
                    **kwargs
                )
            
            logger.info(f"命名空间 {namespace} 相似度搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"命名空间相似度搜索失败: {str(e)}")
            return []
    
    async def add_documents(
        self,
        documents: List[Document],
        user_id: str,
        conversation_id: Optional[str] = None,
        file_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """添加文档到指定命名空间"""
        try:
            if not self.vector_store or not documents:
                return False
            
            # 生成命名空间
            namespace = self._get_namespace(user_id, conversation_id)
            
            # 增强元数据
            enhanced_documents = []
            for doc in documents:
                enhanced_metadata = {
                    **doc.metadata,
                    "user_id": user_id,
                    "namespace": namespace,
                    "embedding_provider": self.embedding_provider,
                }
                
                if conversation_id:
                    enhanced_metadata["conversation_id"] = conversation_id
                if file_id:
                    enhanced_metadata["file_id"] = file_id
                
                # 过滤元数据类型
                filtered_metadata = {
                    k: v for k, v in enhanced_metadata.items() 
                    if isinstance(v, (str, bool, int, float))
                }
                
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                )
                enhanced_documents.append(enhanced_doc)
            
            # 添加到指定命名空间
            if hasattr(self.vector_store, 'add_documents'):
                ids = await self.vector_store.aadd_documents(
                    enhanced_documents,
                    namespace=namespace,  # 关键：指定命名空间
                    **kwargs
                )
            else:
                # 回退实现
                ids = await self.vector_store.aadd_documents(enhanced_documents)
            
            logger.info(f"成功添加 {len(enhanced_documents)} 个文档到命名空间 {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到命名空间失败: {str(e)}")
            return False
    