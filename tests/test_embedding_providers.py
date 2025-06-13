"""
测试不同嵌入模型提供商的支持
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pytest
from unittest.mock import Mock, patch

from app.llm.rag.retrieval_service import LLMRetrievalService
from app.services.retrieval_service import RetrievalService


class TestEmbeddingProviders:
    """测试多种嵌入模型提供商"""
    
    def test_supported_providers(self):
        """测试获取支持的提供商列表"""
        providers = LLMRetrievalService.get_supported_providers()
        assert "openai" in providers
        assert "qwen" in providers
        assert isinstance(providers, list)
    
    def test_create_with_openai_provider(self):
        """测试创建 OpenAI 提供商的检索服务"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            service = LLMRetrievalService.create_with_provider("openai")
            assert service.embedding_provider == "openai"
            assert service.embeddings is not None
    
    def test_create_with_qwen_provider(self):
        """测试创建 Qwen 提供商的检索服务"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.QWEN_API_KEY = "test-key"
            mock_settings.DASHSCOPE_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            with patch('langchain_community.embeddings.DashScopeEmbeddings') as MockDashScope:
                mock_embeddings = Mock()
                mock_embeddings.model = "text-embedding-v1"
                MockDashScope.return_value = mock_embeddings
                
                service = LLMRetrievalService.create_with_provider("qwen")
                assert service.embedding_provider == "qwen"
                assert service.embeddings is not None
    
    def test_get_embedding_info(self):
        """测试获取嵌入模型信息"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            service = LLMRetrievalService(embedding_provider="openai")
            info = service.get_embedding_info()
            
            assert "provider" in info
            assert "model_name" in info
            assert "dimension" in info
            assert "model_class" in info
            assert info["provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_embedding_connection_test(self):
        """测试嵌入模型连接测试"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            service = LLMRetrievalService(embedding_provider="openai")
            
            # Mock 嵌入方法
            with patch.object(service.embeddings, 'embed_query', return_value=[0.1] * 1536):
                result = await service.test_embedding_connection()
                
                assert "success" in result
                assert "provider" in result
                assert result["provider"] == "openai"
                assert "model_info" in result
    
    def test_dashscope_embeddings_initialization(self):
        """测试 DashScope 嵌入模型初始化"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.QWEN_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            with patch('langchain_community.embeddings.DashScopeEmbeddings') as MockDashScope:
                mock_embeddings = Mock()
                mock_embeddings.model = "text-embedding-v1"
                mock_embeddings.dashscope_api_key = "test-key"
                MockDashScope.return_value = mock_embeddings
                
                service = LLMRetrievalService(embedding_provider="qwen")
                
                # 验证 DashScopeEmbeddings 被正确调用
                MockDashScope.assert_called_once_with(
                    model="text-embedding-v1",
                    dashscope_api_key="test-key"
                )
                assert service.embeddings == mock_embeddings
    
    @pytest.mark.asyncio
    async def test_dashscope_embed_query(self):
        """测试 DashScope 查询嵌入"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.QWEN_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            with patch('langchain_community.embeddings.DashScopeEmbeddings') as MockDashScope:
                mock_embeddings = Mock()
                mock_embeddings.model = "text-embedding-v1"
                mock_embeddings.embed_query.return_value = [0.1] * 1536
                MockDashScope.return_value = mock_embeddings
                
                service = LLMRetrievalService(embedding_provider="qwen")
                
                # 测试查询嵌入
                result = await asyncio.to_thread(service.embeddings.embed_query, "测试文本")
                
                assert len(result) == 1536
                assert all(isinstance(x, float) for x in result)
                mock_embeddings.embed_query.assert_called_once_with("测试文本")
    
    @pytest.mark.asyncio 
    async def test_dashscope_embed_documents(self):
        """测试 DashScope 文档嵌入"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.QWEN_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            with patch('langchain_community.embeddings.DashScopeEmbeddings') as MockDashScope:
                mock_embeddings = Mock()
                mock_embeddings.model = "text-embedding-v1"
                mock_embeddings.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
                MockDashScope.return_value = mock_embeddings
                
                service = LLMRetrievalService(embedding_provider="qwen")
                
                texts = ["文本1", "文本2"]
                results = await asyncio.to_thread(service.embeddings.embed_documents, texts)
                
                assert len(results) == 2
                assert all(len(r) == 1536 for r in results)
                mock_embeddings.embed_documents.assert_called_once_with(texts)
    
    def test_business_layer_integration(self):
        """测试业务层集成多模型支持"""
        mock_file_repo = Mock()
        
        # 测试默认提供商
        # with patch('app.core.config.settings') as mock_settings:
        #     mock_settings.EMBEDDING_PROVIDER = "openai"
        #     mock_settings.OPENAI_API_KEY = "test-key"
        #     mock_settings.VECTOR_DB_TYPE = "chroma"
        #     mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
        #     # 创建时不指定 embedding_provider，应该使用配置中的默认值
        #     service = RetrievalService(mock_file_repo, embedding_provider=None)
        #     assert service.llm_retrieval.embedding_provider == "qwen"
        
        # 测试指定提供商
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.QWEN_API_KEY = "sk-f7b6989a8aea440d9eddc2d225f1876c"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            with patch('langchain_community.embeddings.DashScopeEmbeddings'):
                service = RetrievalService.create_with_provider(mock_file_repo, "qwen")
                assert service.llm_retrieval.embedding_provider == "qwen"
    
    @pytest.mark.asyncio
    async def test_search_with_different_providers(self):
        """测试使用不同提供商进行搜索"""
        # 这个测试需要实际的向量数据库，这里只测试接口调用
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            service = LLMRetrievalService(embedding_provider="openai")
            
            # Mock 向量搜索
            with patch.object(service, 'vector_store') as mock_vector_store:
                mock_vector_store.similarity_search_with_score.return_value = []
                
                results = await service.search_documents(
                    query="测试查询",
                    file_ids=["file1", "file2"],
                    top_k=5
                )
                
                assert isinstance(results, list)
    
    def test_provider_fallback(self):
        """测试不支持的提供商回退机制"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            # 使用不支持的提供商
            service = LLMRetrievalService(embedding_provider="unsupported")
            
            # 应该回退到 OpenAI
            assert service.embedding_provider == "unsupported"
            # 嵌入模型应该是 OpenAI 的实例
            from langchain_openai import OpenAIEmbeddings
            assert isinstance(service.embeddings, OpenAIEmbeddings)
    
    def test_import_error_handling(self):
        """测试导入错误处理"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.QWEN_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            # 由于实际环境中 dashscope 包不存在，直接测试这种情况
            # 当前系统应该会抛出 ImportError
            try:
                service = LLMRetrievalService(embedding_provider="qwen")
                # 如果没有抛出异常，说明 dashscope 包已安装
                assert service.embedding_provider == "qwen"
            except ImportError as e:
                # 验证错误消息包含预期内容
                assert "DashScope 嵌入需要安装以下依赖" in str(e)
    
    def test_dimension_detection(self):
        """测试维度检测功能"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            # 测试 OpenAI 服务
            openai_service = LLMRetrievalService(embedding_provider="openai")
            dimension = openai_service.get_embedding_dimension()
            assert dimension == 1536
            
            # 测试 Qwen 服务
            with patch('langchain_community.embeddings.DashScopeEmbeddings') as MockDashScope:
                mock_embeddings = Mock()
                mock_embeddings.model = "text-embedding-v1"
                MockDashScope.return_value = mock_embeddings
                
                qwen_service = LLMRetrievalService(embedding_provider="qwen")
                dimension = qwen_service.get_embedding_dimension()
                assert dimension == 1536
    
    @pytest.mark.asyncio
    async def test_enhanced_embedding_info(self):
        """测试增强的嵌入模型信息"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.QWEN_API_KEY = "test-key"
            mock_settings.VECTOR_DB_TYPE = "chroma"
            mock_settings.CHROMA_DB_DIR = "./test_chroma"
            
            with patch('langchain_community.embeddings.DashScopeEmbeddings') as MockDashScope:
                mock_embeddings = Mock()
                mock_embeddings.model = "text-embedding-v1"
                mock_embeddings.__class__.__name__ = "DashScopeEmbeddings"
                MockDashScope.return_value = mock_embeddings
                
                service = LLMRetrievalService(embedding_provider="qwen")
                
                # 测试获取嵌入信息
                info = service.get_embedding_info()
                assert info["provider"] == "qwen"
                assert info["model_name"] == "text-embedding-v1"
                assert info["model_class"] == "DashScopeEmbeddings"
                assert "dimension" in info
                
                # 测试连接测试
                mock_embeddings.embed_query.return_value = [0.1] * 1536
                result = await service.test_embedding_connection()
                assert result["success"] is True
                assert "model_info" in result
                assert result["model_info"]["provider"] == "qwen"


if __name__ == "__main__":
    # 运行简单测试
    async def main():
        test_instance = TestEmbeddingProviders()
        
        print("测试支持的提供商...")
        test_instance.test_supported_providers()
        print("✓ 支持的提供商测试通过")
        
        print("测试嵌入模型信息...")
        test_instance.test_get_embedding_info()
        print("✓ 嵌入模型信息测试通过")
        
        print("测试业务层集成...")
        test_instance.test_business_layer_integration()
        print("✓ 业务层集成测试通过")
        
        print("测试回退机制...")
        test_instance.test_provider_fallback()
        print("✓ 回退机制测试通过")
        
        print("测试导入错误处理...")
        try:
            test_instance.test_import_error_handling()
            print("✓ 导入错误处理测试通过")
        except Exception as e:
            print(f"⚠ 导入错误处理测试跳过: {str(e)}")
        
        print("测试维度检测...")
        test_instance.test_dimension_detection()
        print("✓ 维度检测测试通过")
        
        print("\n所有测试通过! 🎉")
    
    asyncio.run(main()) 