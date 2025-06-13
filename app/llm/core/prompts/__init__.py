"""
提示词管理模块
提供不同对话模式的系统提示词
"""

import os
from datetime import datetime
from typing import Optional, List

from app.core.config import settings


def load_prompt_template(filename: str) -> str:
    """从文件加载提示词模板"""
    prompt_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"提示词文件未找到: {filename}")


def format_prompt(template: str, **kwargs) -> str:
    """格式化提示词模板"""
    default_vars = {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_name": getattr(settings, 'PROJECT_NAME', 'AI') + " Agent",
    }
    default_vars.update(kwargs)
    return template.format(**default_vars)


class PromptManager:
    """提示词管理器"""
    
    def __init__(self):
        self._cache = {}
    
    def get_chat_prompt(self, **kwargs) -> str:
        """获取聊天模式提示词"""
        if "chat" not in self._cache:
            self._cache["chat"] = load_prompt_template("chat.md")
        return format_prompt(self._cache["chat"], **kwargs)
    
    def get_rag_prompt(self, **kwargs) -> str:
        """获取RAG模式提示词"""
        if "rag" not in self._cache:
            self._cache["rag"] = load_prompt_template("rag.md")
        return format_prompt(self._cache["rag"], **kwargs)
    
    def get_agent_prompt(self, available_tools: Optional[List[str]] = None, **kwargs) -> str:
        """获取Agent模式提示词"""
        if "agent" not in self._cache:
            self._cache["agent"] = load_prompt_template("agent.md")
        
        # 格式化工具列表
        if available_tools:
            tools_text = "\n".join([f"- {tool}" for tool in available_tools])
            kwargs["available_tools"] = tools_text
        else:
            kwargs["available_tools"] = "无特定工具"
        
        return format_prompt(self._cache["agent"], **kwargs)
    
    def get_system_prompt(self, **kwargs) -> str:
        """获取通用系统提示词（向后兼容）"""
        if "system" not in self._cache:
            self._cache["system"] = load_prompt_template("system.md")
        return format_prompt(self._cache["system"], **kwargs)
    
    def get_question_analysis_prompt(self, user_query: str, **kwargs) -> str:
        """获取问题分析提示词"""
        if "question_analysis" not in self._cache:
            self._cache["question_analysis"] = load_prompt_template("question_analysis.md")
        
        kwargs["user_query"] = user_query
        return format_prompt(self._cache["question_analysis"], **kwargs)
    
    def clear_cache(self):
        """清空提示词缓存"""
        self._cache.clear()


# 全局提示词管理器实例
prompt_manager = PromptManager()

# 向后兼容的函数
def load_system_prompt():
    """Load the system prompt from the file."""
    return prompt_manager.get_system_prompt()


# 导出常用的提示词（向后兼容）
SYSTEM_PROMPT = load_system_prompt()
