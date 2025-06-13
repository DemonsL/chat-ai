"""
测试LLM文件处理器 - 使用 LangChain 标准加载器
"""

import os
from typing import List
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from langchain.docstore.document import Document

from app.core.exceptions import (FileProcessingException,
                                 InvalidFileTypeException)
from app.llm.rag.file_processor import LLMFileProcessor


class TestLLMFileProcessor:
    """测试LLM文件处理器"""

    def setup_method(self):
        """测试前准备"""
        # 创建文件处理器实例
        self.file_processor = LLMFileProcessor()

    @pytest.mark.asyncio
    async def test_process_pdf_file_content(self):
        """测试处理PDF文件内容 - 使用LangChain标准加载器"""
        file_path = "/tmp/test.pdf"
        file_type = "pdf"

        # 创建模拟的文档
        mock_documents = [
            Document(
                page_content="第一页内容",
                metadata={"page": 1, "source": file_path}
            ),
            Document(
                page_content="第二页内容",
                metadata={"page": 2, "source": file_path}
            )
        ]

        # 模拟文件存在
        with patch("os.path.exists", return_value=True):
            # 模拟LangChain文档加载
            with patch.object(
                self.file_processor,
                "_load_documents_with_langchain",
                return_value=mock_documents
            ):
                # 模拟LLM检索服务的文本分割
                with patch.object(
                    self.file_processor.llm_retrieval,
                    "split_text_into_chunks",
                    return_value=["第一页内容", "第二页内容"],
                ):
                    # 执行测试
                    chunks, metadata = await self.file_processor.process_file_content(
                        file_path, file_type
                    )

                    # 验证结果
                    assert len(chunks) == 2
                    assert chunks[0] == "第一页内容"
                    assert chunks[1] == "第二页内容"
                    assert metadata["source_type"] == "pdf"
                    assert metadata["chunk_count"] == 2
                    assert metadata["document_count"] == 2
                    assert metadata["character_count"] == len("第一页内容\n\n第二页内容")

    @pytest.mark.asyncio
    async def test_process_docx_file_content(self):
        """测试处理DOCX文件内容 - 使用LangChain标准加载器"""
        file_path = "/tmp/test.docx"
        file_type = "docx"

        # 创建模拟的文档
        mock_documents = [
            Document(
                page_content="这是DOCX文件内容",
                metadata={"source": file_path}
            )
        ]

        # 模拟文件存在
        with patch("os.path.exists", return_value=True):
            # 模拟LangChain文档加载
            with patch.object(
                self.file_processor,
                "_load_documents_with_langchain",
                return_value=mock_documents
            ):
                # 模拟LLM检索服务的文本分割
                with patch.object(
                    self.file_processor.llm_retrieval,
                    "split_text_into_chunks",
                    return_value=["这是DOCX文件内容"],
                ):
                    # 执行测试
                    chunks, metadata = await self.file_processor.process_file_content(
                        file_path, file_type
                    )

                    # 验证结果
                    assert len(chunks) == 1
                    assert chunks[0] == "这是DOCX文件内容"
                    assert metadata["source_type"] == "docx"
                    assert metadata["chunk_count"] == 1
                    assert metadata["document_count"] == 1

    @pytest.mark.asyncio
    async def test_load_documents_with_langchain_pdf(self):
        """测试使用LangChain加载PDF文档"""
        file_path = "/tmp/test.pdf"
        file_type = "pdf"

        # 创建模拟的PDF加载器
        mock_loader = MagicMock()
        mock_documents = [
            Document(page_content="PDF内容", metadata={"page": 1})
        ]
        mock_loader.load.return_value = mock_documents

        # 模拟文件存在
        with patch("os.path.exists", return_value=True):
            # 模拟PyPDFLoader
            with patch("app.llm.rag.file_processor.PyPDFLoader", return_value=mock_loader):
                # 执行测试
                documents = await self.file_processor._load_documents_with_langchain(
                    file_path, file_type
                )

                # 验证结果
                assert len(documents) == 1
                assert documents[0].page_content == "PDF内容"
                assert documents[0].metadata["source_file"] == file_path
                assert documents[0].metadata["file_type"] == "pdf"
                assert documents[0].metadata["loader_type"] == "PyPDFLoader"

    @pytest.mark.asyncio
    async def test_load_documents_with_langchain_txt(self):
        """测试使用LangChain加载TXT文档"""
        file_path = "/tmp/test.txt"
        file_type = "txt"

        # 创建模拟的文本加载器
        mock_loader = MagicMock()
        mock_documents = [
            Document(page_content="文本内容", metadata={"source": file_path})
        ]
        mock_loader.load.return_value = mock_documents

        # 模拟文件存在
        with patch("os.path.exists", return_value=True):
            # 模拟TextLoader
            with patch("app.llm.rag.file_processor.TextLoader", return_value=mock_loader):
                # 执行测试
                documents = await self.file_processor._load_documents_with_langchain(
                    file_path, file_type
                )

                # 验证结果
                assert len(documents) == 1
                assert documents[0].page_content == "文本内容"
                assert documents[0].metadata["source_file"] == file_path
                assert documents[0].metadata["file_type"] == "txt"
                assert documents[0].metadata["loader_type"] == "TextLoader"

    @pytest.mark.asyncio
    async def test_process_file_not_found(self):
        """测试处理不存在的文件"""
        file_path = "/tmp/nonexistent.pdf"
        file_type = "pdf"

        # 模拟文件不存在
        with patch("os.path.exists", return_value=False):
            # 执行测试并期望异常
            with pytest.raises(FileProcessingException):
                await self.file_processor.process_file_content(file_path, file_type)

    @pytest.mark.asyncio
    async def test_load_documents_unsupported_type(self):
        """测试加载不支持的文件类型"""
        file_path = "/tmp/test.xyz"
        file_type = "xyz"

        # 模拟文件存在
        with patch("os.path.exists", return_value=True):
            # 执行测试并期望异常
            with pytest.raises(InvalidFileTypeException):
                await self.file_processor._load_documents_with_langchain(file_path, file_type)

    def test_collect_metadata_pdf(self):
        """测试收集PDF元数据"""
        documents = [
            Document(page_content="页面1", metadata={"page": 1}),
            Document(page_content="页面2", metadata={"page": 2}),
        ]
        
        metadata = self.file_processor._collect_metadata(documents, "pdf")
        
        assert metadata["source_type"] == "pdf"
        assert metadata["loader_type"] == "langchain_standard"
        assert metadata["document_count"] == 2
        assert metadata["page_count"] == 2

    def test_collect_metadata_docx(self):
        """测试收集DOCX元数据"""
        documents = [
            Document(page_content="内容", metadata={"source": "/tmp/test.docx"}),
        ]
        
        metadata = self.file_processor._collect_metadata(documents, "docx")
        
        assert metadata["source_type"] == "docx"
        assert metadata["document_count"] == 1
        assert metadata["section_count"] == 1

    def test_collect_metadata_image(self):
        """测试收集图片元数据"""
        documents = [
            Document(
                page_content="OCR文本", 
                metadata={
                    "image_width": 800,
                    "image_height": 600,
                    "ocr_confidence": 0.95
                }
            ),
        ]
        
        metadata = self.file_processor._collect_metadata(documents, "image")
        
        assert metadata["source_type"] == "image"
        assert metadata["document_count"] == 1
        assert metadata["image_width"] == 800
        assert metadata["ocr_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_add_chunks_to_vector_store(self):
        """测试添加文本块到向量存储"""
        chunks = ["文本块1", "文本块2"]
        metadatas = [{"chunk_id": "1"}, {"chunk_id": "2"}]

        # 模拟LLM检索服务的添加方法
        with patch.object(
            self.file_processor.llm_retrieval,
            "add_documents_to_vector_store",
            return_value=True,
        ):
            # 执行测试
            result = await self.file_processor.add_chunks_to_vector_store(chunks, metadatas)

            # 验证结果
            assert result is True

    @pytest.mark.asyncio
    async def test_remove_from_vector_store(self):
        """测试从向量存储删除"""
        file_id = "test-file-id"

        # 模拟LLM检索服务的删除方法
        with patch.object(
            self.file_processor.llm_retrieval,
            "remove_documents_from_vector_store",
            return_value=True,
        ):
            # 执行测试
            result = await self.file_processor.remove_from_vector_store(file_id)

            # 验证结果
            assert result is True

    def test_get_supported_file_types(self):
        """测试获取支持的文件类型"""
        supported_types = self.file_processor.get_supported_file_types()
        
        # 验证结果
        assert isinstance(supported_types, list)
        assert "pdf" in supported_types
        assert "docx" in supported_types
        assert "txt" in supported_types
        assert "image" in supported_types

    def test_validate_file_type(self):
        """测试文件类型验证"""
        # 测试支持的文件类型
        assert self.file_processor.validate_file_type("pdf") is True
        assert self.file_processor.validate_file_type("docx") is True
        assert self.file_processor.validate_file_type("txt") is True
        assert self.file_processor.validate_file_type("image") is True
        
        # 测试不支持的文件类型
        assert self.file_processor.validate_file_type("xyz") is False
        assert self.file_processor.validate_file_type("") is False

    def test_get_loader_info(self):
        """测试获取加载器信息"""
        loader_info = self.file_processor.get_loader_info()
        
        # 验证结果
        assert isinstance(loader_info, dict)
        assert loader_info["pdf"] == "PyPDFLoader"
        assert loader_info["docx"] == "Docx2txtLoader"
        assert loader_info["txt"] == "TextLoader"
        assert loader_info["image"] == "UnstructuredImageLoader"

    @pytest.mark.asyncio
    async def test_validate_loader_availability(self):
        """测试验证加载器可用性"""
        # 模拟所有加载器都可用
        with patch("builtins.__import__"):
            availability = await self.file_processor.validate_loader_availability()
            
            # 验证结果
            assert isinstance(availability, dict)
            assert all(isinstance(v, bool) for v in availability.values())

    @pytest.mark.asyncio
    async def test_process_empty_documents(self):
        """测试处理空文档列表"""
        file_path = "/tmp/empty.pdf"
        file_type = "pdf"

        # 模拟返回空文档列表
        with patch("os.path.exists", return_value=True):
            with patch.object(
                self.file_processor,
                "_load_documents_with_langchain",
                return_value=[]
            ):
                # 执行测试并期望异常
                with pytest.raises(FileProcessingException, match="无法从文件中提取内容"):
                    await self.file_processor.process_file_content(file_path, file_type)

    @pytest.mark.asyncio
    async def test_loader_error_handling(self):
        """测试加载器错误处理"""
        file_path = "/tmp/corrupt.pdf"
        file_type = "pdf"

        # 模拟文件存在但加载失败
        with patch("os.path.exists", return_value=True):
            with patch.object(
                self.file_processor,
                "_load_documents_with_langchain",
                side_effect=Exception("加载失败")
            ):
                # 执行测试并期望异常
                with pytest.raises(FileProcessingException, match="文件内容处理失败"):
                    await self.file_processor.process_file_content(file_path, file_type)
