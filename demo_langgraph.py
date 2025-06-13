"""
LangGraph 多轮对话演示脚本
演示如何使用重构后的 LLMManager 进行不同模式的对话
包含新的提示词管理和 PostgresSaver checkpointer 功能
"""

import asyncio
import json
import os
from typing import Dict, List
from uuid import uuid4

# 设置环境变量（演示用）
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # 请替换为真实的API Key

from app.llm.manage import LLMManager
from app.llm.core.prompts import prompt_manager


async def demo_prompt_management():
    """演示提示词管理功能"""
    print("=== 提示词管理演示 ===")
    
    print("聊天模式提示词:")
    print(prompt_manager.get_chat_prompt())
    print("\n" + "-"*50 + "\n")
    
    print("RAG模式提示词:")
    print(prompt_manager.get_rag_prompt())
    print("\n" + "-"*50 + "\n")
    
    print("Agent模式提示词:")
    print(prompt_manager.get_agent_prompt(available_tools=["search", "analysis"]))
    print("\n" + "="*50 + "\n")


async def demo_checkpointer_conversation():
    """演示使用 checkpointer 的多轮对话"""
    print("=== Checkpointer 多轮对话演示 ===")
    
    llm_manager = LLMManager()
    conversation_id = uuid4()  # 生成对话ID
    
    # 模型配置
    model_config = {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "extra_params": {
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
    
    print(f"对话ID: {conversation_id}")
    print("注意：状态将通过 checkpointer 持久化\n")
    
    # 模拟多轮对话
    conversation_turns = [
        "我想学习Python编程，请给我一些建议。",
        "我应该从哪些基础概念开始？",
        "能推荐一些适合初学者的项目吗？"
    ]
    
    for i, user_input in enumerate(conversation_turns, 1):
        print(f"回合 {i}")
        print(f"用户: {user_input}")
        print("助手: ", end="", flush=True)
        
        # 构建当前对话的消息历史（模拟实际场景）
        messages = [{"role": "user", "content": user_input}]
        
        try:
            async for chunk in llm_manager.process_conversation(
                messages=messages,
                model_config=model_config,
                mode="chat",
                conversation_id=conversation_id  # 传递对话ID
            ):
                chunk_data = json.loads(chunk)
                if not chunk_data.get("error", False):
                    print(chunk_data.get("content", ""), end="", flush=True)
            
            print("\n" + "-"*40)
            
        except Exception as e:
            print(f"错误: {e}")
            break
    
    print("="*50 + "\n")


async def demo_conversation_state_management():
    """演示对话状态管理"""
    print("=== 对话状态管理演示 ===")
    
    llm_manager = LLMManager()
    conversation_id = uuid4()
    
    print(f"创建新对话: {conversation_id}")
    print("缓存状态:")
    print(f"已缓存的模型: {llm_manager.get_cached_models()}")
    print(f"已缓存的图: {llm_manager.get_cached_graphs()}")
    
    # 模拟一次对话以创建缓存
    model_config = {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 500,
        "extra_params": {
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
    
    messages = [{"role": "user", "content": "测试对话状态"}]
    
    try:
        async for chunk in llm_manager.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="chat",
            conversation_id=conversation_id
        ):
            pass  # 不输出内容，只是为了触发缓存
        
        print("\n对话后的缓存状态:")
        print(f"已缓存的模型: {llm_manager.get_cached_models()}")
        print(f"已缓存的图: {llm_manager.get_cached_graphs()}")
        
        # 清除特定对话的状态
        print(f"\n清除对话 {conversation_id} 的状态...")
        llm_manager.clear_conversation_state(conversation_id)
        
        # 清除所有缓存
        print("清除所有缓存...")
        llm_manager.clear_model_cache()
        
        print("清除后的缓存状态:")
        print(f"已缓存的模型: {llm_manager.get_cached_models()}")
        print(f"已缓存的图: {llm_manager.get_cached_graphs()}")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    
    print("="*50 + "\n")


async def demo_rag_with_custom_prompt():
    """演示RAG模式的自定义提示词"""
    print("=== RAG模式自定义提示词演示 ===")
    
    llm_manager = LLMManager()
    
    # 模型配置
    model_config = {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "extra_params": {
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
    
    # 模拟检索到的文档
    retrieved_documents = [
        "LangGraph 是一个用于构建状态化、多角色应用程序的库，基于 LangChain。",
        "PostgresSaver 提供了持久化检查点功能，可以保存对话状态到 PostgreSQL 数据库。",
        "使用提示词管理器可以集中管理不同模式的系统提示词。"
    ]
    
    # 自定义RAG提示词
    custom_rag_prompt = """
# 专业技术文档分析助手

你是一个专门分析技术文档的AI助手。请严格基于提供的文档内容回答问题。

## 分析要求
- 仅基于文档内容回答
- 如果信息不足，明确说明
- 提供具体的技术细节
- 使用专业术语

当前时间: {current_time}
"""
    
    messages = [
        {"role": "user", "content": "根据文档，请解释LangGraph和PostgresSaver的关系。"}
    ]
    
    print("用户: 根据文档，请解释LangGraph和PostgresSaver的关系。")
    print("\n使用自定义RAG提示词:")
    print("助手: ", end="", flush=True)
    
    try:
        async for chunk in llm_manager.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="rag",
            system_prompt=custom_rag_prompt,  # 使用自定义提示词
            retrieved_documents=retrieved_documents,
            conversation_id=uuid4()
        ):
            chunk_data = json.loads(chunk)
            if not chunk_data.get("error", False):
                print(chunk_data.get("content", ""), end="", flush=True)
                
                # 如果有来源信息，打印出来
                if chunk_data.get("done", False) and chunk_data.get("sources"):
                    print(f"\n\n来源文档:")
                    for i, source in enumerate(chunk_data["sources"], 1):
                        print(f"{i}. {source['content']}")
        
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"错误: {e}")


async def demo_agent_with_tools():
    """演示Agent模式的工具使用"""
    print("=== Agent模式工具使用演示 ===")
    
    llm_manager = LLMManager()
    
    # 模型配置
    model_config = {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "extra_params": {
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
    
    # 可用工具
    available_tools = ["web_search", "document_analysis", "code_execution", "data_visualization"]
    
    messages = [
        {"role": "user", "content": "我需要分析一个数据集并生成可视化图表，请制定一个详细的执行计划。"}
    ]
    
    print("用户: 我需要分析一个数据集并生成可视化图表，请制定一个详细的执行计划。")
    print(f"可用工具: {', '.join(available_tools)}")
    print("助手: ", end="", flush=True)
    
    try:
        async for chunk in llm_manager.process_conversation(
            messages=messages,
            model_config=model_config,
            mode="agent",
            available_tools=available_tools,
            conversation_id=uuid4()
        ):
            chunk_data = json.loads(chunk)
            if not chunk_data.get("error", False):
                print(chunk_data.get("content", ""), end="", flush=True)
        
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f"错误: {e}")


async def main():
    """主函数，运行所有演示"""
    print("LangGraph 优化版多轮对话系统演示")
    print("包含提示词管理和 PostgresSaver checkpointer 功能\n")
    
    # 演示提示词管理
    await demo_prompt_management()
    
    # 注意：需要设置有效的API密钥才能运行对话相关的演示
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key":
        print("⚠️  请先设置有效的 OPENAI_API_KEY 环境变量")
        print("将跳过需要API调用的演示\n")
        
        # 只演示状态管理功能（不需要API调用）
        await demo_conversation_state_management()
        return
    
    try:
        # 运行所有演示
        await demo_checkpointer_conversation()
        await demo_conversation_state_management()
        await demo_rag_with_custom_prompt()
        await demo_agent_with_tools()
        
    except KeyboardInterrupt:
        print("\n演示已停止")
    except Exception as e:
        print(f"演示过程中发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 