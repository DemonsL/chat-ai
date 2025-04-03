"""
测试文件处理器
"""

import os
import uuid
from pathlib import Path
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import docx
import pypdf
import pytest
from langchain_community.vectorstores import Chroma

from app.core.exceptions import (FileProcessingException,
                                 InvalidFileTypeException)
from app.db.models.user_file import UserFile
from app.db.repositories.user_file_repository import UserFileRepository
from app.llm.rag.file_processor import FileProcessor
from app.schemas.file import FileStatus


class TestFileProcessor:
    """测试文件处理器"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟的仓库对象
        self.file_repo = MagicMock(spec=UserFileRepository)

        # 创建文件处理器实例
        self.file_processor = FileProcessor(file_repo=self.file_repo)

        # 创建测试数据
        self.file_id = uuid.uuid4()
        self.user_id = uuid.uuid4()

        # 设置模拟文件
        self.file = MagicMock(spec=UserFile)
        self.file.id = self.file_id
        self.file.user_id = self.user_id
        self.file.storage_path = "/tmp/test.pdf"
        self.file.file_type = "pdf"
        self.file.original_filename = "test.pdf"
        self.file.status = "pending"

    @pytest.mark.asyncio
    async def test_process_pdf_file(self):
        """测试处理PDF文件"""
        # 设置获取文件的模拟
        self.file_repo.get_by_id.return_value = self.file
        self.file.file_type = "pdf"

        # 模拟文件存在
        with patch("os.path.exists", return_value=True):
            # 模拟PDF内容提取
            with patch.object(
                self.file_processor,
                "_extract_pdf",
                return_value=("这是PDF文件内容", {"page_count": 1}),
            ):
                # 模拟文本分割
                with patch.object(
                    self.file_processor,
                    "_split_text",
                    return_value=[MagicMock(), MagicMock()],
                ):
                    # 模拟向量存储
                    with patch(
                        "app.llm.rag.file_processor.Chroma.from_documents",
                        return_value=MagicMock(),
                    ):
                        # 模拟更新文件状态
                        self.file_repo.update_status.return_value = self.file

                        # 执行测试
                        result = await self.file_processor.process_file(self.file_id)

                        # 验证结果
                        assert result["status"] == "indexed"
                        assert "metadata" in result

                        # 验证调用
                        self.file_repo.get_by_id.assert_called_once_with(self.file_id)
                        self.file_repo.update_status.assert_called()

    @pytest.mark.asyncio
    async def test_process_docx_file(self):
        """测试处理DOCX文件"""
        # 设置获取文件的模拟
        self.file_repo.get_by_id.return_value = self.file
        self.file.file_type = "docx"
        self.file.storage_path = "/tmp/test.docx"

        # 模拟文件存在
        with patch("os.path.exists", return_value=True):
            # 模拟DOCX内容提取
            with patch.object(
                self.file_processor,
                "_extract_docx",
                return_value=("这是DOCX文件内容", {"page_count": 1}),
            ):
                # 模拟文本分割
                with patch.object(
                    self.file_processor,
                    "_split_text",
                    return_value=[MagicMock(), MagicMock()],
                ):
                    # 模拟向量存储
                    with patch(
                        "app.llm.rag.file_processor.Chroma.from_documents",
                        return_value=MagicMock(),
                    ):
                        # 模拟更新文件状态
                        self.file_repo.update_status.return_value = self.file

                        # 执行测试
                        result = await self.file_processor.process_file(self.file_id)

                        # 验证结果
                        assert result["status"] == "indexed"
                        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_process_file_not_found(self):
        """测试处理不存在的文件"""
        # 设置获取文件的模拟
        self.file_repo.get_by_id.return_value = self.file

        # 模拟文件不存在
        with patch("os.path.exists", return_value=False):
            # 模拟更新文件状态
            self.file_repo.update_status.return_value = self.file

            # 执行测试
            result = await self.file_processor.process_file(self.file_id)

            # 验证结果
            assert result["status"] == "error"
            assert "error_message" in result

            # 验证调用
            self.file_repo.update_status.assert_called_with(
                file_id=self.file_id,
                status="error",
                error_message="文件不存在或无法访问",
            )

    def test_extract_pdf(self):
        """测试PDF内容提取"""
        # 创建模拟PDF内容
        pdf_content = "这是PDF文件内容"
        mock_pdf_reader = MagicMock()
        mock_pdf_reader.pages = [MagicMock()]
        mock_pdf_reader.pages[0].extract_text.return_value = pdf_content

        # 模拟pypdf.PdfReader
        with patch("pypdf.PdfReader", return_value=mock_pdf_reader):
            # 执行测试
            content, metadata = self.file_processor._extract_pdf("/fake/path.pdf")

            # 验证结果
            assert content == pdf_content
            assert metadata["page_count"] == 1

    def test_extract_docx(self):
        """测试DOCX内容提取"""
        # 创建模拟DOCX内容
        docx_content = "这是DOCX文件内容"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock()]
        mock_doc.paragraphs[0].text = docx_content

        # 模拟docx.Document
        with patch("docx.Document", return_value=mock_doc):
            # 执行测试
            content, metadata = self.file_processor._extract_docx("/fake/path.docx")

            # 验证结果
            assert content == docx_content
            assert "paragraphs" in metadata

    def test_extract_txt(self):
        """测试TXT内容提取"""
        # 创建模拟TXT内容
        txt_content = "这是TXT文件内容"

        # 模拟open函数
        with patch("builtins.open", mock_open(read_data=txt_content)):
            # 执行测试
            content, metadata = self.file_processor._extract_txt("/fake/path.txt")

            # 验证结果
            assert content == txt_content
            assert "file_size" in metadata

    def test_extract_image(self):
        """测试图片内容提取"""
        # 创建模拟OCR结果
        ocr_result = "这是图片中提取的文本"

        # 模拟pytesseract和PIL
        with patch("pytesseract.image_to_string", return_value=ocr_result):
            with patch("PIL.Image.open", return_value=MagicMock()):
                # 执行测试
                content, metadata = self.file_processor._extract_image("/fake/path.jpg")

                # 验证结果
                assert content == ocr_result
                assert "ocr_engine" in metadata

    def test_extract_invalid_file(self):
        """测试无效文件类型提取"""
        # 执行测试，预期抛出异常
        with pytest.raises(InvalidFileTypeException):
            self.file_processor._extract_file_content("/fake/path.xyz", "xyz")
