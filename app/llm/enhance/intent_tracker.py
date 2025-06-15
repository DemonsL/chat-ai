"""
对话意图连续性跟踪器
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from loguru import logger


class IntentType(str, Enum):
    """意图类型枚举"""
    TASK_CONTINUATION = "task_continuation"  # 任务延续
    TASK_SWITCH = "task_switch"             # 任务切换
    TASK_DEEPENING = "task_deepening"       # 任务深化
    TASK_BRANCHING = "task_branching"       # 任务分支
    CLARIFICATION = "clarification"         # 澄清请求
    GREETING = "greeting"                   # 问候
    UNKNOWN = "unknown"                     # 未知


@dataclass
class IntentUpdate:
    """意图更新结果"""
    intent_type: IntentType
    confidence: float
    task_context: Dict[str, Any]
    reasoning: str
    related_previous_turns: List[int]  # 相关的历史轮次


class TaskContext:
    """任务上下文管理"""
    
    def __init__(self):
        self.current_task = None
        self.task_history = []
        self.entities = {}  # 提取的实体
        self.goals = []     # 用户目标
        
    def update_task(self, task_info: Dict[str, Any]):
        """更新当前任务"""
        if self.current_task:
            self.task_history.append(self.current_task)
        self.current_task = task_info
        
    def add_entity(self, entity_type: str, entity_value: str):
        """添加实体"""
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        if entity_value not in self.entities[entity_type]:
            self.entities[entity_type].append(entity_value)


class IntentTracker:
    """对话意图连续性跟踪器"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
        self.intent_history = []
        self.task_context = TaskContext()
        
    async def track_intent(
        self, 
        current_message: str, 
        conversation_history: List[BaseMessage]
    ) -> IntentUpdate:
        """跟踪对话意图变化"""
        try:
            # 1. 分析意图演变
            intent_analysis = await self._analyze_intent_evolution(
                current_message, 
                conversation_history
            )
            
            # 2. 更新任务上下文
            self._update_task_context(intent_analysis)
            
            # 3. 记录意图历史
            self.intent_history.append({
                'timestamp': datetime.now(),
                'message': current_message,
                'intent_type': intent_analysis.intent_type,
                'confidence': intent_analysis.confidence,
                'reasoning': intent_analysis.reasoning
            })
            
            logger.info(f"意图跟踪: {intent_analysis.intent_type} (置信度: {intent_analysis.confidence:.2f})")
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"意图跟踪失败: {str(e)}")
            return IntentUpdate(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                task_context={},
                reasoning=f"意图分析失败: {str(e)}",
                related_previous_turns=[]
            )
    
    async def _analyze_intent_evolution(
        self, 
        message: str, 
        history: List[BaseMessage]
    ) -> IntentUpdate:
        """分析意图演变模式"""
        
        # 准备上下文信息
        context_info = self._prepare_context_info(history)
        
        if self.llm_model:
            # 使用LLM进行意图分析
            return await self._llm_based_intent_analysis(message, context_info)
        else:
            # 使用规则基础的意图分析
            return await self._rule_based_intent_analysis(message, context_info)
    
    def _prepare_context_info(self, history: List[BaseMessage]) -> Dict[str, Any]:
        """准备上下文信息"""
        recent_messages = []
        
        # 获取最近几轮对话
        for i, msg in enumerate(history[-6:]):  # 最近3轮对话（用户+助手）
            if isinstance(msg, (HumanMessage, AIMessage)):
                recent_messages.append({
                    'role': 'user' if isinstance(msg, HumanMessage) else 'assistant',
                    'content': msg.content,
                    'turn_index': len(history) - 6 + i
                })
        
        return {
            'recent_messages': recent_messages,
            'current_task': self.task_context.current_task,
            'entities': self.task_context.entities,
            'intent_history': self.intent_history[-3:]  # 最近3次意图
        }
    
    async def _llm_based_intent_analysis(
        self, 
        message: str, 
        context_info: Dict[str, Any]
    ) -> IntentUpdate:
        """基于LLM的意图分析"""
        
        prompt = self._build_intent_analysis_prompt(message, context_info)
        
        try:
            # 使用结构化输出
            from pydantic import BaseModel
            
            class IntentAnalysisResult(BaseModel):
                intent_type: str
                confidence: float
                reasoning: str
                task_info: Dict[str, Any]
                related_turns: List[int]
            
            structured_model = self.llm_model.with_structured_output(IntentAnalysisResult)
            result = await structured_model.ainvoke([{"role": "user", "content": prompt}])
            
            return IntentUpdate(
                intent_type=IntentType(result.intent_type),
                confidence=result.confidence,
                task_context=result.task_info,
                reasoning=result.reasoning,
                related_previous_turns=result.related_turns
            )
            
        except Exception as e:
            logger.warning(f"LLM意图分析失败，回退到规则分析: {str(e)}")
            return await self._rule_based_intent_analysis(message, context_info)
    
    async def _rule_based_intent_analysis(
        self, 
        message: str, 
        context_info: Dict[str, Any]
    ) -> IntentUpdate:
        """基于规则的意图分析"""
        
        message_lower = message.lower()
        
        # 问候检测
        if any(greeting in message_lower for greeting in ['你好', 'hello', '您好', 'hi']):
            return IntentUpdate(
                intent_type=IntentType.GREETING,
                confidence=0.9,
                task_context={'task_type': 'greeting'},
                reasoning="检测到问候语",
                related_previous_turns=[]
            )
        
        # 澄清请求检测
        if any(clarification in message_lower for clarification in ['什么意思', '能详细', '解释一下', '不明白']):
            return IntentUpdate(
                intent_type=IntentType.CLARIFICATION,
                confidence=0.8,
                task_context={'task_type': 'clarification'},
                reasoning="检测到澄清请求",
                related_previous_turns=list(range(max(0, len(context_info.get('recent_messages', [])) - 2), 
                                                 len(context_info.get('recent_messages', []))))
            )
        
        # 任务延续检测
        if any(continuation in message_lower for continuation in ['继续', '接着', '然后', '下一步']):
            return IntentUpdate(
                intent_type=IntentType.TASK_CONTINUATION,
                confidence=0.8,
                task_context=context_info.get('current_task', {}),
                reasoning="检测到任务延续关键词",
                related_previous_turns=list(range(max(0, len(context_info.get('recent_messages', [])) - 4), 
                                                 len(context_info.get('recent_messages', []))))
            )
        
        # 任务深化检测
        if any(deepening in message_lower for deepening in ['详细', '具体', '深入', '更多']):
            return IntentUpdate(
                intent_type=IntentType.TASK_DEEPENING,
                confidence=0.7,
                task_context=context_info.get('current_task', {}),
                reasoning="检测到任务深化关键词",
                related_previous_turns=list(range(max(0, len(context_info.get('recent_messages', [])) - 2), 
                                                 len(context_info.get('recent_messages', []))))
            )
        
        # 默认为新任务
        return IntentUpdate(
            intent_type=IntentType.TASK_SWITCH,
            confidence=0.6,
            task_context={'task_type': 'new_task', 'query': message},
            reasoning="无明确延续性指示，可能是新任务",
            related_previous_turns=[]
        )
    
    def _build_intent_analysis_prompt(self, message: str, context_info: Dict[str, Any]) -> str:
        """构建意图分析提示词"""
        
        recent_messages_text = ""
        for msg in context_info.get('recent_messages', []):
            recent_messages_text += f"{msg['role']}: {msg['content']}\n"
        
        intent_history_text = ""
        for intent in context_info.get('intent_history', []):
            intent_history_text += f"- {intent['intent_type']}: {intent['reasoning']}\n"
        
        return f"""
你是一个对话意图分析专家。请分析用户当前消息的意图类型。

# 意图类型定义
- task_continuation: 用户在继续之前的任务
- task_switch: 用户开始了新的任务
- task_deepening: 用户要求对当前话题更详细的信息
- task_branching: 从主任务衍生出子任务
- clarification: 用户请求澄清或解释
- greeting: 问候语
- unknown: 无法确定意图

# 对话历史
{recent_messages_text}

# 当前任务上下文
{json.dumps(context_info.get('current_task', {}), ensure_ascii=False, indent=2)}

# 最近意图历史
{intent_history_text}

# 当前用户消息
{message}

请分析用户意图，并以JSON格式返回：
{{
  "intent_type": "意图类型",
  "confidence": 0.0-1.0之间的置信度,
  "reasoning": "分析推理过程",
  "task_info": {{"task_type": "任务类型", "entities": {{}}, "goals": []}},
  "related_turns": [相关的历史轮次索引]
}}
"""
    
    def _update_task_context(self, intent_analysis: IntentUpdate):
        """更新任务上下文"""
        
        if intent_analysis.intent_type == IntentType.TASK_SWITCH:
            # 新任务，重置上下文
            self.task_context.update_task(intent_analysis.task_context)
            
        elif intent_analysis.intent_type in [IntentType.TASK_CONTINUATION, IntentType.TASK_DEEPENING]:
            # 任务延续或深化，更新当前任务
            if self.task_context.current_task:
                self.task_context.current_task.update(intent_analysis.task_context)
            else:
                self.task_context.update_task(intent_analysis.task_context)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话总结"""
        return {
            'intent_distribution': self._calculate_intent_distribution(),
            'task_context': {
                'current_task': self.task_context.current_task,
                'entities': self.task_context.entities,
                'task_count': len(self.task_context.task_history) + (1 if self.task_context.current_task else 0)
            },
            'conversation_flow': [
                {
                    'intent_type': intent['intent_type'],
                    'timestamp': intent['timestamp'].isoformat(),
                    'reasoning': intent['reasoning']
                }
                for intent in self.intent_history
            ]
        }
    
    def _calculate_intent_distribution(self) -> Dict[str, int]:
        """计算意图分布"""
        distribution = {}
        for intent in self.intent_history:
            intent_type = intent['intent_type']
            distribution[intent_type] = distribution.get(intent_type, 0) + 1
        return distribution 