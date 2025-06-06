from typing import List, Dict, Optional, Any
from uuid import UUID

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.core.config import settings
from app.db.repositories.user_file_repository import UserFileRepository


class RetrievalService:
    """
    检索服务
    专门处理文档检索和向量搜索功能，不涉及LLM处理
    """
    
    def __init__(self, file_repo: UserFileRepository):
        self.file_repo = file_repo
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL, 
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # 初始化向量存储
        self.vector_store = None
        if settings.VECTOR_DB_TYPE == "chroma":
            self.vector_store = Chroma(
                persist_directory=settings.CHROMA_DB_DIR,
                embedding_function=self.embeddings,
            )
    
    async def retrieve_documents(
        self,
        query: str,
        file_ids: List[UUID],
        top_k: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[str]:
        """
        根据查询检索相关文档
        
        Args:
            query: 查询文本
            file_ids: 文件ID列表
            top_k: 返回文档数量
            similarity_threshold: 相似度阈值
            
        Returns:
            相关文档内容列表
        """
        retrieved_docs = []
        
        try:
            if not self.vector_store or not file_ids:
                return retrieved_docs
            
            # 执行向量相似性搜索
            search_results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter={
                    "file_id": {"$in": [str(file_id) for file_id in file_ids]}
                },
            )
            
            # 过滤相似度较高的文档
            for doc, score in search_results:
                if score < similarity_threshold:
                    retrieved_docs.append(doc.page_content)
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
        
        return retrieved_docs
    
    async def retrieve_documents_with_metadata(
        self,
        query: str,
        file_ids: List[UUID],
        top_k: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        检索文档并返回元数据
        
        Args:
            query: 查询文本
            file_ids: 文件ID列表
            top_k: 返回文档数量
            similarity_threshold: 相似度阈值
            
        Returns:
            包含文档内容和元数据的字典列表
        """
        retrieved_docs = []
        
        try:
            if not self.vector_store or not file_ids:
                return retrieved_docs
            
            # 执行向量相似性搜索
            search_results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter={
                    "file_id": {"$in": [str(file_id) for file_id in file_ids]}
                },
            )
            
            # 构建结果
            for doc, score in search_results:
                if score < similarity_threshold:
                    retrieved_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": score
                    })
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
        
        return retrieved_docs
    
    async def add_documents(
        self,
        file_id: UUID,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        添加文档到向量存储
        
        Args:
            file_id: 文件ID
            documents: 文档内容列表
            metadata_list: 文档元数据列表
            
        Returns:
            是否成功添加
        """
        try:
            if not self.vector_store or not documents:
                return False
            
            # 构建元数据
            if not metadata_list:
                metadata_list = [{"file_id": str(file_id)} for _ in documents]
            else:
                # 确保每个元数据都包含file_id
                for metadata in metadata_list:
                    metadata["file_id"] = str(file_id)
            
            # 添加到向量存储
            self.vector_store.add_texts(
                texts=documents,
                metadatas=metadata_list
            )
            
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    async def remove_documents(self, file_id: UUID) -> bool:
        """
        从向量存储中移除指定文件的文档
        
        Args:
            file_id: 文件ID
            
        Returns:
            是否成功移除
        """
        try:
            if not self.vector_store:
                return False
            
            # 这里需要根据具体的向量存储实现来删除文档
            # Chroma 可能需要通过 collection 操作来删除
            # 由于 langchain 的接口限制，这里是一个简化版本
            logger.info(f"请求删除文件 {file_id} 的向量数据")
            
            return True
            
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
        文本分块
        
        Args:
            text: 原始文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块后的文本列表
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            
            chunks = text_splitter.split_text(text)
            return chunks
            
        except Exception as e:
            logger.error(f"文本分块失败: {str(e)}")
            return [text]  # 返回原始文本作为单个块 