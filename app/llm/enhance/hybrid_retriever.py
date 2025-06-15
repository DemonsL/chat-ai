"""
混合检索策略实现
结合向量检索和关键词检索，提供更准确的文档检索
"""
import asyncio
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from loguru import logger

# BM25 实现
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 未安装，BM25检索功能不可用")


class SearchStrategy(str, Enum):
    """检索策略枚举"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    vector_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0
    source: str = ""  # "vector", "keyword", "hybrid"


@dataclass
class SearchConfig:
    """搜索配置"""
    strategy: SearchStrategy = SearchStrategy.HYBRID
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 10
    similarity_threshold: float = 0.7
    enable_rerank: bool = True
    rerank_top_k: int = 20  # 重排序前的候选数量


class KeywordSearchEngine:
    """关键词搜索引擎（基于BM25）"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_tokens = []
        
    def index_documents(self, documents: List[Document]):
        """索引文档"""
        if not BM25_AVAILABLE:
            logger.warning("BM25不可用，跳过关键词索引")
            return
            
        self.documents = documents
        
        # 分词处理
        self.doc_tokens = []
        for doc in documents:
            # 简单的中英文分词
            tokens = self._tokenize(doc.page_content)
            self.doc_tokens.append(tokens)
        
        if self.doc_tokens:
            self.bm25 = BM25Okapi(self.doc_tokens)
            logger.info(f"BM25索引创建完成，文档数量: {len(documents)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        import jieba  # 中文分词
        
        # 中文分词
        chinese_tokens = list(jieba.cut(text))
        
        # 英文分词
        english_tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # 合并并过滤
        all_tokens = chinese_tokens + english_tokens
        filtered_tokens = [token.strip() for token in all_tokens if len(token.strip()) > 1]
        
        return filtered_tokens
    
    async def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """搜索文档"""
        if not self.bm25 or not BM25_AVAILABLE:
            return []
        
        # 查询分词
        query_tokens = self._tokenize(query)
        
        # 获取BM25分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取top-k结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的结果
                results.append((self.documents[idx], float(scores[idx])))
        
        return results


class CrossEncoderReranker:
    """交叉编码器重排序"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "BAAI/bge-reranker-base"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"重排序模型加载成功: {self.model_name}")
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {str(e)}")
            self.model = None
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """重排序文档"""
        if not self.model or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        try:
            # 准备输入对
            pairs = [[query, doc.page_content] for doc in documents]
            
            # 在线程池中执行计算密集型操作
            scores = await asyncio.to_thread(self.model.predict, pairs)
            
            # 排序并返回
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return [(doc, 0.0) for doc in documents[:top_k]]


class HybridRetriever:
    """混合检索器"""
    
    def __init__(
        self, 
        vector_store: Chroma, 
        config: SearchConfig = None
    ):
        self.vector_store = vector_store
        self.config = config or SearchConfig()
        self.keyword_search = KeywordSearchEngine()
        self.reranker = CrossEncoderReranker() if self.config.enable_rerank else None
        
        # 初始化关键词搜索索引
        self._initialize_keyword_index()
    
    def _initialize_keyword_index(self):
        """初始化关键词索引"""
        try:
            # 从向量数据库获取所有文档
            # 注意：这里需要根据实际的Chroma API调整
            all_docs = self._get_all_documents_from_vector_store()
            if all_docs:
                self.keyword_search.index_documents(all_docs)
        except Exception as e:
            logger.error(f"初始化关键词索引失败: {str(e)}")
    
    def _get_all_documents_from_vector_store(self) -> List[Document]:
        """从向量数据库获取所有文档"""
        try:
            # 这是一个示例实现，需要根据实际的Chroma API调整
            # 获取集合中的所有文档
            collection = self.vector_store._collection
            results = collection.get()
            
            documents = []
            for i, (doc_id, metadata, content) in enumerate(zip(
                results['ids'], 
                results['metadatas'], 
                results['documents']
            )):
                doc = Document(
                    page_content=content,
                    metadata=metadata or {}
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.warning(f"无法从向量数据库获取所有文档: {str(e)}")
            return []
    
    async def search(
        self, 
        query: str, 
        file_ids: List[str] = None,
        config: SearchConfig = None
    ) -> List[RetrievalResult]:
        """混合搜索"""
        search_config = config or self.config
        
        if search_config.strategy == SearchStrategy.VECTOR_ONLY:
            return await self._vector_search_only(query, file_ids, search_config)
        elif search_config.strategy == SearchStrategy.KEYWORD_ONLY:
            return await self._keyword_search_only(query, search_config)
        elif search_config.strategy == SearchStrategy.ADAPTIVE:
            return await self._adaptive_search(query, file_ids, search_config)
        else:  # HYBRID
            return await self._hybrid_search(query, file_ids, search_config)
    
    async def _vector_search_only(
        self, 
        query: str, 
        file_ids: List[str], 
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """纯向量搜索"""
        try:
            filter_condition = {"file_id": {"$in": file_ids}} if file_ids else None
            
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query=query,
                k=config.top_k,
                filter=filter_condition
            )
            
            retrieval_results = []
            for i, (doc, score) in enumerate(results):
                # 向量数据库通常返回距离，需要转换为相似度
                similarity = 1.0 / (1.0 + score) if score > 0 else 1.0
                
                retrieval_results.append(RetrievalResult(
                    document=doc,
                    vector_score=similarity,
                    combined_score=similarity,
                    rank=i + 1,
                    source="vector"
                ))
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    async def _keyword_search_only(
        self, 
        query: str, 
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """纯关键词搜索"""
        try:
            results = await self.keyword_search.search(query, config.top_k)
            
            retrieval_results = []
            for i, (doc, score) in enumerate(results):
                # 标准化BM25分数
                normalized_score = min(score / 10.0, 1.0)  # 简单标准化
                
                retrieval_results.append(RetrievalResult(
                    document=doc,
                    keyword_score=normalized_score,
                    combined_score=normalized_score,
                    rank=i + 1,
                    source="keyword"
                ))
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {str(e)}")
            return []
    
    async def _hybrid_search(
        self, 
        query: str, 
        file_ids: List[str], 
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """混合搜索"""
        try:
            # 并行执行向量搜索和关键词搜索
            vector_task = self._vector_search_only(query, file_ids, 
                SearchConfig(top_k=config.rerank_top_k if config.enable_rerank else config.top_k))
            keyword_task = self._keyword_search_only(query, 
                SearchConfig(top_k=config.rerank_top_k if config.enable_rerank else config.top_k))
            
            vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)
            
            # 融合结果
            combined_results = self._combine_results(
                vector_results, 
                keyword_results, 
                config
            )
            
            # 重排序（如果启用）
            if config.enable_rerank and self.reranker and combined_results:
                combined_results = await self._rerank_results(query, combined_results, config)
            
            return combined_results[:config.top_k]
            
        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}")
            return []
    
    def _combine_results(
        self, 
        vector_results: List[RetrievalResult], 
        keyword_results: List[RetrievalResult], 
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """融合检索结果"""
        # 使用文档内容作为去重键
        doc_map = {}
        
        # 添加向量搜索结果
        for result in vector_results:
            doc_key = result.document.page_content[:100]  # 使用内容前100字符作为key
            if doc_key not in doc_map:
                doc_map[doc_key] = result
                doc_map[doc_key].source = "vector"
            else:
                # 合并分数
                existing = doc_map[doc_key]
                existing.vector_score = max(existing.vector_score, result.vector_score)
        
        # 添加关键词搜索结果
        for result in keyword_results:
            doc_key = result.document.page_content[:100]
            if doc_key not in doc_map:
                doc_map[doc_key] = result
                doc_map[doc_key].source = "keyword"
            else:
                # 合并分数
                existing = doc_map[doc_key]
                existing.keyword_score = max(existing.keyword_score, result.keyword_score)
                existing.source = "hybrid"
        
        # 计算组合分数并排序
        combined_results = []
        for result in doc_map.values():
            # 使用RRF（倒数排名融合）或加权平均
            if config.vector_weight + config.keyword_weight > 0:
                combined_score = (
                    result.vector_score * config.vector_weight + 
                    result.keyword_score * config.keyword_weight
                ) / (config.vector_weight + config.keyword_weight)
            else:
                combined_score = (result.vector_score + result.keyword_score) / 2
            
            result.combined_score = combined_score
            combined_results.append(result)
        
        # 按组合分数排序
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # 更新排名
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult], 
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """重排序结果"""
        if not self.reranker or not results:
            return results
        
        try:
            documents = [result.document for result in results]
            reranked_docs_scores = await self.reranker.rerank(query, documents, config.top_k)
            
            # 更新结果
            reranked_results = []
            for i, (doc, rerank_score) in enumerate(reranked_docs_scores):
                # 找到原始结果
                original_result = None
                for result in results:
                    if result.document.page_content == doc.page_content:
                        original_result = result
                        break
                
                if original_result:
                    original_result.combined_score = rerank_score
                    original_result.rank = i + 1
                    reranked_results.append(original_result)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return results
    
    async def _adaptive_search(
        self, 
        query: str, 
        file_ids: List[str], 
        config: SearchConfig
    ) -> List[RetrievalResult]:
        """自适应搜索策略"""
        # 分析查询特征决定搜索策略
        query_features = self._analyze_query(query)
        
        if query_features['has_keywords'] > 0.7:
            # 关键词丰富，优先关键词搜索
            adaptive_config = SearchConfig(
                strategy=SearchStrategy.HYBRID,
                vector_weight=0.3,
                keyword_weight=0.7,
                top_k=config.top_k
            )
        elif query_features['is_semantic'] > 0.7:
            # 语义查询，优先向量搜索
            adaptive_config = SearchConfig(
                strategy=SearchStrategy.HYBRID,
                vector_weight=0.8,
                keyword_weight=0.2,
                top_k=config.top_k
            )
        else:
            # 平衡策略
            adaptive_config = config
        
        return await self._hybrid_search(query, file_ids, adaptive_config)
    
    def _analyze_query(self, query: str) -> Dict[str, float]:
        """分析查询特征"""
        import re
        
        features = {}
        
        # 关键词密度分析
        specific_terms = len(re.findall(r'\b[A-Za-z]{3,}\b', query))  # 英文关键词
        chinese_terms = len(re.findall(r'[\u4e00-\u9fff]{2,}', query))  # 中文词汇
        total_chars = len(query)
        
        features['has_keywords'] = min((specific_terms + chinese_terms) / max(total_chars / 10, 1), 1.0)
        
        # 语义查询特征
        semantic_indicators = ['如何', '为什么', '什么是', '怎么样', 'how', 'why', 'what', 'explain']
        semantic_count = sum(1 for indicator in semantic_indicators if indicator in query.lower())
        features['is_semantic'] = min(semantic_count / 3, 1.0)
        
        # 查询长度
        features['query_length'] = min(len(query) / 100, 1.0)
        
        return features
    
    def update_config(self, config: SearchConfig):
        """更新搜索配置"""
        self.config = config
    
    def get_search_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        return {
            "keyword_index_size": len(self.keyword_search.documents),
            "reranker_available": self.reranker is not None,
            "current_config": {
                "strategy": self.config.strategy,
                "vector_weight": self.config.vector_weight,
                "keyword_weight": self.config.keyword_weight,
                "top_k": self.config.top_k
            }
        } 