import os
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader, 
    TextLoader,
    UnstructuredImageLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from app.core.exceptions import (FileProcessingException,
                                 InvalidFileTypeException)


class LLMFileProcessor:
    """
    LLM层文件处理器
    专门负责文件内容提取和文本分块等纯LLM功能
    不涉及业务逻辑、数据库操作和向量存储
    
    使用 LangChain 标准文档加载器和文本分割器，确保代码标准性和统一性
    """

    def __init__(self):
        # 支持的文件类型和对应的加载器映射
        self.loader_mapping = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "txt": TextLoader,
            "image": UnstructuredImageLoader,
        }
        
        # 默认文本分割器配置
        self.default_chunk_size = 1000
        self.default_chunk_overlap = 200

    async def load_documents(
        self, 
        file_path: str, 
        file_type: str
    ) -> List[Document]:
        """
        使用LangChain标准加载器加载文档
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            
        Returns:
            Document对象列表
        """
        return await self._load_documents_with_langchain(file_path, file_type)
    
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        使用LangChain标准文本分割器分割文档
        
        Args:
            documents: 文档列表
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            分割后的Document对象列表
        """
        if not documents:
            return []
        
        # 使用传入的参数或默认值
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        
        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        # 分割文档
        split_docs = text_splitter.split_documents(documents)
        
        logger.info(f"文档分割完成：{len(documents)} 个原始文档 -> {len(split_docs)} 个文档块")
        return split_docs
    
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
    
    def collect_metadata(self, documents: List[Document], file_type: str) -> Dict:
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
    