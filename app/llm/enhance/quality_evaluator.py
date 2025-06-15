"""
对话质量评估器
用于评估LLM响应质量并提供改进建议
"""
import asyncio
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import BaseMessage
from loguru import logger


class QualityDimension(str, Enum):
    """质量评估维度"""
    RELEVANCE = "relevance"         # 相关性
    COMPLETENESS = "completeness"   # 完整性
    ACCURACY = "accuracy"           # 准确性
    COHERENCE = "coherence"         # 连贯性
    HELPFULNESS = "helpfulness"     # 有用性
    SAFETY = "safety"               # 安全性


@dataclass
class QualityScore:
    """质量评分"""
    relevance: float = 0.0
    completeness: float = 0.0
    accuracy: float = 0.0
    coherence: float = 0.0
    helpfulness: float = 0.0
    safety: float = 1.0  # 安全性默认为1.0（安全）
    overall: float = 0.0
    
    def __post_init__(self):
        """计算总体评分"""
        scores = [
            self.relevance,
            self.completeness,
            self.accuracy,
            self.coherence,
            self.helpfulness,
            self.safety
        ]
        self.overall = sum(scores) / len(scores)


@dataclass
class QualityReport:
    """质量评估报告"""
    score: QualityScore
    suggestions: List[str]
    issues: List[str]
    strengths: List[str]
    confidence: float


class ConversationQualityEvaluator:
    """对话质量评估器"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
        self.evaluation_cache = {}
        
    async def evaluate_response_quality(
        self, 
        user_query: str, 
        response: str, 
        context: Dict = None
    ) -> QualityReport:
        """评估回答质量"""
        try:
            context = context or {}
            
            # 并行评估各个维度
            tasks = [
                self._evaluate_relevance(user_query, response),
                self._evaluate_completeness(user_query, response, context),
                self._evaluate_accuracy(response, context),
                self._evaluate_coherence(response),
                self._evaluate_helpfulness(user_query, response),
                self._evaluate_safety(response)
            ]
            
            scores = await asyncio.gather(*tasks)
            
            quality_score = QualityScore(
                relevance=scores[0],
                completeness=scores[1],
                accuracy=scores[2],
                coherence=scores[3],
                helpfulness=scores[4],
                safety=scores[5]
            )
            
            # 生成改进建议
            suggestions = self._generate_suggestions(quality_score, user_query, response)
            
            # 识别问题和优点
            issues = self._identify_issues(quality_score)
            strengths = self._identify_strengths(quality_score)
            
            # 计算评估置信度
            confidence = self._calculate_confidence(quality_score, context)
            
            return QualityReport(
                score=quality_score,
                suggestions=suggestions,
                issues=issues,
                strengths=strengths,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            return QualityReport(
                score=QualityScore(),
                suggestions=["评估系统暂时不可用"],
                issues=["无法完成质量评估"],
                strengths=[],
                confidence=0.0
            )
    
    async def _evaluate_relevance(self, user_query: str, response: str) -> float:
        """评估相关性"""
        if self.llm_model:
            return await self._llm_evaluate_relevance(user_query, response)
        else:
            return self._rule_based_relevance(user_query, response)
    
    async def _llm_evaluate_relevance(self, user_query: str, response: str) -> float:
        """基于LLM的相关性评估"""
        try:
            prompt = f"""
请评估以下回答与用户问题的相关性，给出0-1之间的分数：

用户问题：{user_query}

AI回答：{response}

评估标准：
- 1.0: 完全相关，直接回答了用户问题
- 0.8: 高度相关，基本回答了用户问题
- 0.6: 中等相关，部分回答了用户问题
- 0.4: 低度相关，只是略微涉及用户问题
- 0.2: 几乎不相关，基本没有回答用户问题
- 0.0: 完全不相关

请只返回数字分数，不要其他解释。
"""
            
            result = await self.llm_model.ainvoke([{"role": "user", "content": prompt}])
            score_text = result.content.strip()
            
            # 提取数字
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)
            
            return 0.5  # 默认中等分数
            
        except Exception as e:
            logger.warning(f"LLM相关性评估失败: {str(e)}")
            return self._rule_based_relevance(user_query, response)
    
    def _rule_based_relevance(self, user_query: str, response: str) -> float:
        """基于规则的相关性评估"""
        # 简单的关键词匹配
        query_words = set(re.findall(r'\w+', user_query.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        
        if not query_words:
            return 0.5
        
        # 计算词汇重叠度
        overlap = len(query_words.intersection(response_words))
        relevance = overlap / len(query_words)
        
        return min(relevance, 1.0)
    
    async def _evaluate_completeness(self, user_query: str, response: str, context: Dict) -> float:
        """评估完整性"""
        # 检查回答长度
        length_score = min(len(response) / 200, 1.0)  # 200字符为基准
        
        # 检查是否包含关键信息
        key_info_score = self._check_key_information(user_query, response)
        
        # 检查是否有未完成的句子
        completion_score = 1.0 if not response.endswith(('...', '。。。', '...')) else 0.7
        
        return (length_score + key_info_score + completion_score) / 3
    
    def _check_key_information(self, user_query: str, response: str) -> float:
        """检查关键信息覆盖"""
        # 识别问题类型
        question_indicators = {
            'what': ['什么', 'what', '是什么'],
            'how': ['如何', 'how', '怎么', '怎样'],
            'why': ['为什么', 'why', '为何'],
            'when': ['什么时候', 'when', '何时'],
            'where': ['哪里', 'where', '在哪'],
            'who': ['谁', 'who', '什么人']
        }
        
        query_lower = user_query.lower()
        response_lower = response.lower()
        
        score = 0.5  # 基础分数
        
        for q_type, indicators in question_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                # 检查回答是否包含相应的信息类型
                if q_type == 'what' and any(word in response_lower for word in ['是', '为', '指']):
                    score += 0.1
                elif q_type == 'how' and any(word in response_lower for word in ['步骤', '方法', '通过']):
                    score += 0.1
                elif q_type == 'why' and any(word in response_lower for word in ['因为', '由于', '原因']):
                    score += 0.1
        
        return min(score, 1.0)
    
    async def _evaluate_accuracy(self, response: str, context: Dict) -> float:
        """评估准确性"""
        # 检查是否有明显的错误信息
        accuracy_score = 1.0
        
        # 检查数字和日期的合理性
        if not self._check_numerical_accuracy(response):
            accuracy_score -= 0.2
        
        # 检查是否有自相矛盾的内容
        if self._check_contradictions(response):
            accuracy_score -= 0.3
        
        # 检查是否有不确定的表述
        uncertainty_penalty = self._check_uncertainty(response)
        accuracy_score -= uncertainty_penalty
        
        return max(accuracy_score, 0.0)
    
    def _check_numerical_accuracy(self, response: str) -> bool:
        """检查数字准确性"""
        # 检查明显不合理的数字
        numbers = re.findall(r'\d+', response)
        for num in numbers:
            if len(num) > 10:  # 过长的数字可能有问题
                return False
        return True
    
    def _check_contradictions(self, response: str) -> bool:
        """检查自相矛盾"""
        # 简单的矛盾检测
        contradiction_patterns = [
            (r'不是.*是', r'是.*不是'),
            (r'没有.*有', r'有.*没有'),
            (r'不能.*能', r'能.*不能')
        ]
        
        for pattern1, pattern2 in contradiction_patterns:
            if re.search(pattern1, response) and re.search(pattern2, response):
                return True
        
        return False
    
    def _check_uncertainty(self, response: str) -> float:
        """检查不确定性表述"""
        uncertainty_words = ['可能', '也许', '大概', '估计', '应该', '或许', 'maybe', 'probably']
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        
        # 适度的不确定性是好的，过多则影响准确性
        if uncertainty_count <= 2:
            return 0.0
        elif uncertainty_count <= 4:
            return 0.1
        else:
            return 0.2
    
    async def _evaluate_coherence(self, response: str) -> float:
        """评估连贯性"""
        # 检查句子结构
        sentences = re.split(r'[。！？.!?]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.8  # 单句回答
        
        coherence_score = 1.0
        
        # 检查句子长度变化
        lengths = [len(s) for s in sentences]
        if max(lengths) - min(lengths) > 100:  # 句子长度差异过大
            coherence_score -= 0.1
        
        # 检查重复内容
        if self._check_repetition(sentences):
            coherence_score -= 0.2
        
        # 检查逻辑连接词
        if self._has_logical_connectors(response):
            coherence_score += 0.1
        
        return min(max(coherence_score, 0.0), 1.0)
    
    def _check_repetition(self, sentences: List[str]) -> bool:
        """检查重复内容"""
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                # 计算句子相似度
                words1 = set(sent1.split())
                words2 = set(sent2.split())
                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.7:  # 70%相似度认为是重复
                        return True
        return False
    
    def _has_logical_connectors(self, response: str) -> bool:
        """检查逻辑连接词"""
        connectors = ['因此', '所以', '然而', '但是', '而且', '另外', '首先', '其次', '最后']
        return any(connector in response for connector in connectors)
    
    async def _evaluate_helpfulness(self, user_query: str, response: str) -> float:
        """评估有用性"""
        helpfulness_score = 0.5  # 基础分数
        
        # 检查是否提供了具体信息
        if self._provides_specific_info(response):
            helpfulness_score += 0.2
        
        # 检查是否提供了可操作的建议
        if self._provides_actionable_advice(response):
            helpfulness_score += 0.2
        
        # 检查是否回答了用户的隐含需求
        if self._addresses_implicit_needs(user_query, response):
            helpfulness_score += 0.1
        
        return min(helpfulness_score, 1.0)
    
    def _provides_specific_info(self, response: str) -> bool:
        """检查是否提供具体信息"""
        # 检查数字、日期、名称等具体信息
        specific_patterns = [
            r'\d+',  # 数字
            r'\d{4}年',  # 年份
            r'[A-Z][a-z]+',  # 专有名词
            r'http[s]?://\S+',  # 链接
        ]
        
        return any(re.search(pattern, response) for pattern in specific_patterns)
    
    def _provides_actionable_advice(self, response: str) -> bool:
        """检查是否提供可操作建议"""
        action_words = ['可以', '应该', '建议', '推荐', '尝试', '考虑', '步骤', '方法']
        return any(word in response for word in action_words)
    
    def _addresses_implicit_needs(self, user_query: str, response: str) -> bool:
        """检查是否回答了隐含需求"""
        # 简单的隐含需求检测
        if '怎么' in user_query and ('步骤' in response or '方法' in response):
            return True
        if '为什么' in user_query and ('原因' in response or '因为' in response):
            return True
        return False
    
    async def _evaluate_safety(self, response: str) -> float:
        """评估安全性"""
        safety_score = 1.0
        
        # 检查有害内容
        harmful_patterns = [
            r'暴力',
            r'仇恨',
            r'歧视',
            r'违法',
            r'危险'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                safety_score -= 0.3
        
        # 检查隐私信息泄露
        privacy_patterns = [
            r'\d{11}',  # 手机号
            r'\d{15,18}',  # 身份证号
            r'\w+@\w+\.\w+',  # 邮箱
        ]
        
        for pattern in privacy_patterns:
            if re.search(pattern, response):
                safety_score -= 0.5
        
        return max(safety_score, 0.0)
    
    def _generate_suggestions(self, quality_score: QualityScore, user_query: str, response: str) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if quality_score.relevance < 0.7:
            suggestions.append("回答与用户问题的相关性不够，建议更直接地回答用户问题")
        
        if quality_score.completeness < 0.7:
            suggestions.append("回答不够完整，建议补充更多相关信息")
        
        if quality_score.accuracy < 0.8:
            suggestions.append("回答的准确性有待提高，建议核实信息的正确性")
        
        if quality_score.coherence < 0.7:
            suggestions.append("回答的逻辑性和连贯性需要改善")
        
        if quality_score.helpfulness < 0.7:
            suggestions.append("回答的实用性不够，建议提供更多可操作的建议")
        
        if quality_score.safety < 0.9:
            suggestions.append("回答存在安全风险，需要检查和修正")
        
        return suggestions
    
    def _identify_issues(self, quality_score: QualityScore) -> List[str]:
        """识别问题"""
        issues = []
        
        if quality_score.relevance < 0.5:
            issues.append("回答与问题严重不符")
        
        if quality_score.completeness < 0.5:
            issues.append("回答过于简短或不完整")
        
        if quality_score.accuracy < 0.6:
            issues.append("回答可能包含错误信息")
        
        if quality_score.coherence < 0.5:
            issues.append("回答逻辑混乱或自相矛盾")
        
        if quality_score.safety < 0.8:
            issues.append("回答存在安全隐患")
        
        return issues
    
    def _identify_strengths(self, quality_score: QualityScore) -> List[str]:
        """识别优点"""
        strengths = []
        
        if quality_score.relevance > 0.8:
            strengths.append("回答高度相关")
        
        if quality_score.completeness > 0.8:
            strengths.append("回答内容完整")
        
        if quality_score.accuracy > 0.9:
            strengths.append("回答准确可靠")
        
        if quality_score.coherence > 0.8:
            strengths.append("回答逻辑清晰")
        
        if quality_score.helpfulness > 0.8:
            strengths.append("回答实用性强")
        
        return strengths
    
    def _calculate_confidence(self, quality_score: QualityScore, context: Dict) -> float:
        """计算评估置信度"""
        base_confidence = 0.7
        
        # 如果有更多上下文信息，置信度更高
        if context.get('retrieved_documents'):
            base_confidence += 0.1
        
        if context.get('conversation_history'):
            base_confidence += 0.1
        
        # 如果使用了LLM评估，置信度更高
        if self.llm_model:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    async def batch_evaluate(self, evaluations: List[Tuple[str, str, Dict]]) -> List[QualityReport]:
        """批量评估"""
        tasks = [
            self.evaluate_response_quality(query, response, context)
            for query, response, context in evaluations
        ]
        
        return await asyncio.gather(*tasks)
    
    def get_evaluation_stats(self) -> Dict[str, float]:
        """获取评估统计信息"""
        if not hasattr(self, '_evaluation_history'):
            self._evaluation_history = []
        
        if not self._evaluation_history:
            return {}
        
        # 计算平均分数
        avg_scores = {}
        for dimension in QualityDimension:
            scores = [getattr(eval_result.score, dimension.value) for eval_result in self._evaluation_history]
            avg_scores[f"avg_{dimension.value}"] = sum(scores) / len(scores)
        
        return avg_scores 