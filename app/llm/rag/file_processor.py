import os
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader, 
    TextLoader,
    UnstructuredImageLoader
)
from langchain.docstore.document import Document
from loguru import logger

from app.core.exceptions import (FileProcessingException,
                                 InvalidFileTypeException)
from app.llm.rag.retrieval_service import LLMRetrievalService


class LLMFileProcessor:
    """
    LLM层文件处理器
    专门负责文件内容提取、文本分块和向量化等纯LLM功能
    不涉及业务逻辑和数据库操作
    
    使用 LangChain 标准文档加载器，确保代码标准性和统一性
    """

    def __init__(self):
        # 使用LLM检索服务
        self.llm_retrieval = LLMRetrievalService()
        
        # 支持的文件类型和对应的加载器映射
        self.loader_mapping = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "txt": TextLoader,
            "image": UnstructuredImageLoader,
        }

    async def process_file_content(
        self, 
        file_path: str, 
        file_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Tuple[List[str], Dict]:
        """
        处理文件内容并分块
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            (文本块列表, 文件元数据)
        """
        try:
            # 使用 LangChain 标准加载器提取文件内容
            documents = await self._load_documents_with_langchain(file_path, file_type)
            
            if not documents:
                raise FileProcessingException(detail="无法从文件中提取内容")
            
            # 合并所有文档内容
            full_content = "\n\n".join([doc.page_content for doc in documents])
            
            # 收集元数据
            metadata = self._collect_metadata(documents, file_type)
            
            # 分割文本为块
            chunks = await self.llm_retrieval.split_text_into_chunks(
                text=full_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 添加处理统计信息
            metadata.update({
                "chunk_count": len(chunks),
                "character_count": len(full_content),
                "document_count": len(documents)
            })
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"文件内容处理失败: {str(e)}")
            raise FileProcessingException(detail=f"文件内容处理失败: {str(e)}")
    
    async def _load_documents_with_langchain(
        self, 
        file_path: str, 
        file_type: str
    ) -> List[Document]:
        """
        使用 LangChain 标准加载器加载文档
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            
        Returns:
            文档列表
        """
        if not os.path.exists(file_path):
            raise FileProcessingException(detail=f"文件不存在: {file_path}")
        
        if file_type not in self.loader_mapping:
            raise InvalidFileTypeException(detail=f"不支持的文件类型: {file_type}")
        
        try:
            # 获取对应的加载器类
            loader_class = self.loader_mapping[file_type]
            
            # 创建加载器实例
            if file_type == "txt":
                # TextLoader 需要指定编码
                loader = loader_class(file_path, encoding="utf-8")
            elif file_type == "image":
                # UnstructuredImageLoader 可能需要特殊处理
                loader = loader_class(file_path)
            else:
                loader = loader_class(file_path)
            
            # 加载文档
            documents = loader.load()
            
            # 为每个文档添加文件信息到元数据
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata.update({
                    "source_file": file_path,
                    "file_type": file_type,
                    "loader_type": loader_class.__name__
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"使用 LangChain 加载器加载文件失败: {str(e)}")
            raise FileProcessingException(detail=f"文档加载失败: {str(e)}")
    
    def _collect_metadata(self, documents: List[Document], file_type: str) -> Dict:
        """
        收集文档元数据
        
        Args:
            documents: 文档列表
            file_type: 文件类型
            
        Returns:
            合并的元数据
        """
        metadata = {
            "source_type": file_type,
            "loader_type": "langchain_standard",
            "document_count": len(documents)
        }
        
        # 根据文件类型收集特定元数据
        if file_type == "pdf" and documents:
            # PDF 特定元数据
            first_doc_meta = documents[0].metadata or {}
            if "page" in first_doc_meta:
                metadata["page_count"] = len(documents)
            
        elif file_type == "docx" and documents:
            # DOCX 特定元数据
            metadata["section_count"] = len(documents)
            
        elif file_type == "image" and documents:
            # 图片特定元数据
            first_doc_meta = documents[0].metadata or {}
            metadata.update({k: v for k, v in first_doc_meta.items() 
                           if k.startswith(('image_', 'ocr_'))})
        
        # 收集所有文档的通用元数据
        all_sources = []
        for doc in documents:
            if doc.metadata and "source" in doc.metadata:
                all_sources.append(doc.metadata["source"])
        
        if all_sources:
            # 将sources列表转换为字符串，避免Chroma元数据类型错误
            unique_sources = list(set(all_sources))
            metadata["sources_count"] = len(unique_sources)
            metadata["primary_source"] = unique_sources[0] if unique_sources else ""
        
        return metadata
    
    async def add_chunks_to_vector_store(
        self,
        chunks: List[str],
        metadatas: List[Dict]
    ) -> bool:
        """
        将文本块添加到向量存储
        
        Args:
            chunks: 文本块列表
            metadatas: 元数据列表
            
        Returns:
            是否成功添加
        """
        try:
            return await self.llm_retrieval.add_documents_to_vector_store(
                documents=chunks,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"添加向量存储失败: {str(e)}")
            return False
    
    async def remove_from_vector_store(self, file_id: str) -> bool:
        """
        从向量存储中删除文件相关数据
        
        Args:
            file_id: 文件ID字符串
            
        Returns:
            是否成功删除
        """
        try:
            return await self.llm_retrieval.remove_documents_from_vector_store(file_id)
        except Exception as e:
            logger.error(f"从向量存储删除失败: {str(e)}")
            return False
    
    async def process_file_to_documents(
        self, 
        file_path: str, 
        file_type: str,
        file_id: str,
        user_id: str,
        file_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Tuple[List[Document], Dict]:
        """
        处理文件内容并返回Document对象列表
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            file_id: 文件ID
            user_id: 用户ID
            file_name: 文件名
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            (Document对象列表, 文件元数据)
        """
        try:
            # 使用 LangChain 标准加载器提取文件内容
            documents = await self._load_documents_with_langchain(file_path, file_type)
            
            if not documents:
                raise FileProcessingException(detail="无法从文件中提取内容")
            
            # 合并所有文档内容
            full_content = "\n\n".join([doc.page_content for doc in documents])
            
            # 收集元数据
            metadata = self._collect_metadata(documents, file_type)
            
            # 分割文本为块
            chunks = await self.llm_retrieval.split_text_into_chunks(
                text=full_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 创建Document对象列表
            document_objects = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "file_id": file_id,
                    "user_id": user_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_index": i,
                    "chunk_id": f"{file_id}_{i}",
                    # 只添加简单类型的元数据
                    "source_type": metadata.get("source_type", ""),
                    "loader_type": metadata.get("loader_type", ""),
                    "document_count": metadata.get("document_count", 0),
                    "sources_count": metadata.get("sources_count", 0),
                    "primary_source": metadata.get("primary_source", ""),
                }
                
                doc = Document(
                    page_content=chunk,
                    metadata=doc_metadata
                )
                document_objects.append(doc)
            
            # 添加处理统计信息
            metadata.update({
                "chunk_count": len(chunks),
                "character_count": len(full_content),
                "document_count": len(documents)
            })
            
            return document_objects, metadata
            
        except Exception as e:
            logger.error(f"文件内容处理失败: {str(e)}")
            raise FileProcessingException(detail=f"文件内容处理失败: {str(e)}")
    
    async def add_documents_to_vector_store(
        self,
        documents: List[Document]
    ) -> bool:
        """
        将Document对象添加到向量存储
        
        Args:
            documents: Document对象列表
            
        Returns:
            是否成功添加
        """
        try:
            return await self.llm_retrieval.add_document_objects_to_vector_store(documents)
        except Exception as e:
            logger.error(f"添加Document对象到向量存储失败: {str(e)}")
            return False

    def get_supported_file_types(self) -> List[str]:
        """
        获取支持的文件类型列表
        
        Returns:
            支持的文件类型列表
        """
        return list(self.loader_mapping.keys())
    
    def validate_file_type(self, file_type: str) -> bool:
        """
        验证文件类型是否支持
        
        Args:
            file_type: 文件类型
            
        Returns:
            是否支持该文件类型
        """
        return file_type in self.loader_mapping
    
    def get_loader_info(self) -> Dict[str, str]:
        """
        获取支持的加载器信息
        
        Returns:
            文件类型到加载器的映射信息
        """
        return {
            file_type: loader_class.__name__ 
            for file_type, loader_class in self.loader_mapping.items()
        }
    
    async def validate_loader_availability(self) -> Dict[str, bool]:
        """
        验证所有加载器的可用性
        
        Returns:
            加载器可用性状态
        """
        availability = {}
        
        for file_type, loader_class in self.loader_mapping.items():
            try:
                # 尝试导入和实例化加载器来检查可用性
                if file_type == "image":
                    # UnstructuredImageLoader 可能需要额外的依赖
                    try:
                        from unstructured.partition.image import partition_image
                        availability[file_type] = True
                    except ImportError:
                        availability[file_type] = False
                        logger.warning(f"图片处理加载器不可用，缺少 unstructured 依赖")
                else:
                    availability[file_type] = True
                    
            except Exception as e:
                availability[file_type] = False
                logger.error(f"加载器 {loader_class.__name__} 不可用: {str(e)}")
        
        return availability
