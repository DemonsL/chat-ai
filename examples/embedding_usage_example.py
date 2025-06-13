#!/usr/bin/env python3
"""
Embedding 模型使用示例

本示例展示如何在 chat-ai 项目中使用不同的嵌入模型提供商（OpenAI、Qwen）
"""

import asyncio
import os
from typing import List, Dict, Any
from unittest.mock import Mock

# 模拟导入，实际使用时请确保正确的导入路径
try:
    from app.llm.rag.retrieval_service import LLMRetrievalService
    from app.services.retrieval_service import RetrievalService
    from app.db.repositories.user_file_repository import UserFileRepository
except ImportError:
    print("注意：在实际项目环境中运行此示例")


async def example_1_basic_usage():
    """示例1：基础使用 - 创建不同的嵌入服务"""
    print("=== 示例1：基础使用 ===")
    
    # 1. 使用默认配置创建服务（基于环境变量 EMBEDDING_PROVIDER）
    print("1. 创建默认嵌入服务...")
    default_service = LLMRetrievalService()
    print(f"   默认提供商: {default_service.embedding_provider}")
    
    # 2. 显式创建 OpenAI 嵌入服务
    print("2. 创建 OpenAI 嵌入服务...")
    openai_service = LLMRetrievalService.create_with_provider("openai")
    print(f"   OpenAI 提供商: {openai_service.embedding_provider}")
    
    # 3. 显式创建 Qwen 嵌入服务
    print("3. 创建 Qwen 嵌入服务...")
    try:
        qwen_service = LLMRetrievalService.create_with_provider("qwen")
        print(f"   Qwen 提供商: {qwen_service.embedding_provider}")
    except ImportError:
        print("   注意：需要安装 langchain-community 和 dashscope 库才能使用 Qwen 嵌入")
    
    print()


async def example_2_model_info():
    """示例2：获取模型信息"""
    print("=== 示例2：获取模型信息 ===")
    
    # 创建服务实例
    service = LLMRetrievalService(embedding_provider="openai")
    
    # 获取嵌入模型信息
    info = service.get_embedding_info()
    print("嵌入模型信息:")
    print(f"  提供商: {info.get('provider')}")
    print(f"  模型名称: {info.get('model_name')}")
    print(f"  向量维度: {info.get('dimension')}")
    print(f"  模型类: {info.get('model_class')}")
    
    # 测试连接状态
    print("\n测试连接状态:")
    try:
        connection_result = await service.test_embedding_connection()
        if connection_result.get('success'):
            print(f"  ✓ 连接成功: {connection_result.get('message')}")
        else:
            print(f"  ✗ 连接失败: {connection_result.get('message')}")
    except Exception as e:
        print(f"  ✗ 连接测试出错: {str(e)}")
    
    print()


async def example_3_document_operations():
    """示例3：文档操作 - 添加和搜索"""
    print("=== 示例3：文档操作 ===")
    
    # 创建嵌入服务
    service = LLMRetrievalService(embedding_provider="openai")
    
    # 模拟文档数据
    documents = [
        "这是第一个文档，包含关于机器学习的内容。",
        "第二个文档讨论了深度学习和神经网络。",
        "第三个文档介绍了自然语言处理技术。"
    ]
    
    # 构建元数据
    metadatas = [
        {"file_id": "file_1", "source": "ml_basics.txt", "chunk_index": 0},
        {"file_id": "file_1", "source": "ml_basics.txt", "chunk_index": 1},
        {"file_id": "file_2", "source": "nlp_guide.txt", "chunk_index": 0}
    ]
    
    print("1. 添加文档到向量存储...")
    try:
        success = await service.add_documents_to_vector_store(documents, metadatas)
        if success:
            print("   ✓ 文档添加成功")
        else:
            print("   ✗ 文档添加失败")
    except Exception as e:
        print(f"   ✗ 添加文档时出错: {str(e)}")
    
    print("2. 搜索相关文档...")
    try:
        search_results = await service.search_documents(
            query="深度学习相关内容",
            file_ids=["file_1", "file_2"],
            top_k=2,
            similarity_threshold=0.8
        )
        
        print(f"   找到 {len(search_results)} 个相关文档:")
        for i, result in enumerate(search_results, 1):
            print(f"     {i}. 相似度: {result.get('similarity_score', 'N/A'):.4f}")
            print(f"        内容预览: {result.get('content', '')[:50]}...")
            print(f"        来源: {result.get('metadata', {}).get('source', 'N/A')}")
    except Exception as e:
        print(f"   ✗ 搜索文档时出错: {str(e)}")
    
    print()


async def example_4_business_layer():
    """示例4：业务层集成"""
    print("=== 示例4：业务层集成 ===")
    
    # 模拟文件仓库
    mock_file_repo = Mock(spec=UserFileRepository)
    
    print("1. 创建业务层检索服务...")
    
    # 使用默认提供商
    default_retrieval = RetrievalService(mock_file_repo)
    print(f"   默认服务提供商: {default_retrieval.llm_retrieval.embedding_provider}")
    
    # 使用指定提供商
    openai_retrieval = RetrievalService.create_with_provider(mock_file_repo, "openai")
    print(f"   OpenAI 服务提供商: {openai_retrieval.llm_retrieval.embedding_provider}")
    
    print("2. 获取支持的提供商列表...")
    providers = RetrievalService.get_supported_providers()
    print(f"   支持的提供商: {', '.join(providers)}")
    
    print("3. 测试嵌入模型信息获取...")
    try:
        info = await default_retrieval.get_embedding_info()
        if "error" not in info:
            print(f"   ✓ 模型信息获取成功: {info.get('provider')}")
        else:
            print(f"   ✗ 获取失败: {info.get('error')}")
    except Exception as e:
        print(f"   ✗ 获取模型信息时出错: {str(e)}")
    
    print()


async def example_5_provider_comparison():
    """示例5：提供商对比"""
    print("=== 示例5：提供商对比 ===")
    
    providers_info = []
    
    for provider in ["openai", "qwen"]:
        try:
            print(f"正在测试 {provider} 提供商...")
            service = LLMRetrievalService.create_with_provider(provider)
            info = service.get_embedding_info()
            
            provider_data = {
                "provider": provider,
                "model": info.get('model_name', 'N/A'),
                "dimension": info.get('dimension', 'N/A'),
                "status": "可用"
            }
            
            # 测试连接
            try:
                connection_result = await service.test_embedding_connection()
                if not connection_result.get('success'):
                    provider_data["status"] = f"连接失败: {connection_result.get('message')}"
            except Exception as e:
                provider_data["status"] = f"连接错误: {str(e)}"
            
            providers_info.append(provider_data)
            
        except Exception as e:
            providers_info.append({
                "provider": provider,
                "model": "N/A",
                "dimension": "N/A",
                "status": f"初始化失败: {str(e)}"
            })
    
    # 显示对比结果
    print("\n提供商对比结果:")
    print("-" * 60)
    print(f"{'提供商':<10} {'模型':<20} {'维度':<8} {'状态':<20}")
    print("-" * 60)
    for info in providers_info:
        print(f"{info['provider']:<10} {info['model']:<20} {info['dimension']:<8} {info['status']:<20}")
    
    print()


def example_6_configuration():
    """示例6：配置说明"""
    print("=== 示例6：配置说明 ===")
    
    print("环境变量配置示例:")
    print()
    
    print("1. 使用 OpenAI 嵌入:")
    print("   EMBEDDING_PROVIDER=openai")
    print("   EMBEDDING_MODEL=text-embedding-3-small")
    print("   OPENAI_API_KEY=your-openai-api-key")
    print()
    
    print("2. 使用 Qwen 嵌入:")
    print("   EMBEDDING_PROVIDER=qwen")
    print("   QWEN_EMBEDDING_MODEL=text-embedding-v1")
    print("   QWEN_API_KEY=your-qwen-api-key")
    print("   QWEN_BASE_URL=https://dashscope.aliyuncs.com/api/v1")
    print()
    
    print("3. 当前环境变量:")
    env_vars = [
        "EMBEDDING_PROVIDER",
        "OPENAI_API_KEY", 
        "QWEN_API_KEY",
        "EMBEDDING_MODEL",
        "QWEN_EMBEDDING_MODEL"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "未设置")
        # 隐藏API密钥的部分内容
        if "API_KEY" in var and value != "未设置":
            value = value[:8] + "..." if len(value) > 8 else "***"
        print(f"   {var}: {value}")
    
    print()


async def main():
    """主函数 - 运行所有示例"""
    print("🚀 Embedding 模型多提供商支持示例")
    print("=" * 50)
    print()
    
    try:
        await example_1_basic_usage()
        await example_2_model_info()
        await example_3_document_operations()
        await example_4_business_layer()
        await example_5_provider_comparison()
        example_6_configuration()
        
        print("✅ 所有示例运行完成!")
        print()
        print("💡 提示:")
        print("   - 确保已正确配置相应的API密钥")
        print("   - Qwen 嵌入需要安装依赖: pip install langchain-community dashscope")
        print("   - 实际使用时请在项目环境中运行")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {str(e)}")
        print("请检查配置和依赖是否正确安装")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main()) 