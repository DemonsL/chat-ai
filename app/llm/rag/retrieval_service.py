import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStore

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
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        使用LangChain标准接口进行相似度搜索（带分数）
        
        Args:
            query: 查询文本
            k: 返回文档数量
            filter_dict: 过滤条件字典
            **kwargs: 其他参数
            
        Returns:
            (Document, score)元组列表
        """
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
            
            # 使用LangChain标准的异步相似度搜索
            results = await self.vector_store.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict,
                **kwargs
            )
            
            logger.info(f"相似度搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        使用LangChain标准接口进行相似度搜索
        
        Args:
            query: 查询文本  
            k: 返回文档数量
            filter_dict: 过滤条件字典
            **kwargs: 其他参数
            
        Returns:
            Document列表
        """
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
            
            # 使用LangChain标准的异步相似度搜索
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                filter=filter_dict,
                **kwargs
            )
            
            logger.info(f"相似度搜索完成，返回 {len(results)} 个文档")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []
    
    async def add_documents(
        self,
        documents: List[Document]
    ) -> bool:
        """
        使用LangChain标准接口添加文档到向量存储
        
        Args:
            documents: Document对象列表
            
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
            
            logger.info(f"成功添加 {len(enhanced_documents)} 个文档到向量存储")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            return False
    

    
    async def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """
        使用LangChain标准接口按过滤条件删除文档
        
        Args:
            filter_dict: 过滤条件字典
            
        Returns:
            是否成功删除
        """
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return False
            
            # 使用标准的删除方法
            success = await self.vector_store.adelete(filter=filter_dict)
            
            logger.info(f"成功删除匹配过滤条件的文档: {filter_dict}")
            return success
            
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    
    def get_text_splitter(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> RecursiveCharacterTextSplitter:
        """
        获取文本分割器
        
        Args:
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            文本分割器实例
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
    
    @property
    def is_ready(self) -> bool:
        """检查向量存储是否准备就绪"""
        return self.vector_store is not None
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        获取嵌入模型信息
        
        Returns:
            包含嵌入模型信息的字典
        """
        return {
            "provider": self.embedding_provider,
            "model": getattr(self.embeddings, 'model', 'unknown'),
            "vector_store_type": settings.VECTOR_DB_TYPE,
            "vector_store_ready": self.is_ready
        } 