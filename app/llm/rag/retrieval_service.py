import asyncio
import os
from typing import Any, Dict, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.core.config import settings


class LLMRetrievalService:
    """
    LLM检索服务
    使用标准化的langchain_chroma接口实现向量存储和文档检索
    支持多种嵌入模型提供商：OpenAI、Qwen等
    """
    
    def __init__(self, embedding_provider: str = None):
        # 确定使用的嵌入模型提供商
        self.embedding_provider = embedding_provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
        
        # 初始化嵌入模型
        self.embeddings = self._initialize_embedding_model()
        
        # 初始化向量存储
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_embedding_model(self) -> Embeddings:
        """
        根据配置初始化嵌入模型
        
        Returns:
            嵌入模型实例
        """
        try:
            if self.embedding_provider.lower() == 'openai':
                logger.info("使用 OpenAI 嵌入模型")
                return OpenAIEmbeddings(
                    model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                    openai_api_key=settings.OPENAI_API_KEY
                )
                
            elif self.embedding_provider.lower() == 'qwen':
                logger.info("使用 Qwen (DashScope) 嵌入模型")
                try:
                    from langchain_community.embeddings import DashScopeEmbeddings
                    
                    return DashScopeEmbeddings(
                        model=getattr(settings, 'QWEN_EMBEDDING_MODEL', 'text-embedding-v1'),
                        dashscope_api_key=getattr(settings, 'QWEN_API_KEY', None) or getattr(settings, 'DASHSCOPE_API_KEY', None)
                    )
                except ImportError as e:
                    logger.error(f"无法导入 DashScopeEmbeddings: {str(e)}")
                    logger.error("请确保安装了 langchain-community 和 dashscope 库")
                    raise ImportError(
                        "DashScope 嵌入需要安装以下依赖:\n"
                        "pip install langchain-community dashscope"
                    )
                
            else:
                logger.warning(f"不支持的嵌入模型提供商: {self.embedding_provider}，回退到 OpenAI")
                return OpenAIEmbeddings(
                    model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                    openai_api_key=settings.OPENAI_API_KEY
                )
                
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            # 回退到 OpenAI 作为默认选项
            return OpenAIEmbeddings(
                model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                openai_api_key=settings.OPENAI_API_KEY
            )
    
    def _initialize_vector_store(self):
        """
        初始化向量存储
        """
        try:
            if settings.VECTOR_DB_TYPE == "chroma":
                # 确保存储目录存在
                os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
                
                # 使用标准化的langchain_chroma接口
                self.vector_store = Chroma(
                    persist_directory=settings.CHROMA_DB_DIR,
                    embedding_function=self.embeddings,
                    collection_name="documents"  # 指定集合名称
                )
                logger.info(f"向量存储初始化成功，使用 {self.embedding_provider} 嵌入模型")
            else:
                logger.error(f"不支持的向量数据库类型: {settings.VECTOR_DB_TYPE}")
                
        except Exception as e:
            logger.error(f"向量存储初始化失败: {str(e)}")
            self.vector_store = None
    
    async def search_documents(
        self,
        query: str,
        file_ids: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        根据查询在向量数据库中搜索相关文档
        
        Args:
            query: 查询文本
            file_ids: 文件ID列表（字符串格式）
            top_k: 返回文档数量
            similarity_threshold: 相似度阈值
            
        Returns:
            包含文档内容、元数据和相似度分数的字典列表
        """
        search_results = []
        
        try:
            if not self.vector_store or not file_ids:
                logger.warning("向量存储未初始化或文件ID列表为空")
                return search_results
            
            # 构建过滤条件 - 使用正确的Chroma过滤语法
            filter_condition = {"file_id": {"$in": file_ids}}
            
            # 在线程池中执行同步的向量相似性搜索
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query=query,
                k=top_k,
                filter=filter_condition
            )
            
            # 过滤相似度较高的文档并格式化结果
            for doc, distance in results:
                # Chroma返回的是欧氏距离，距离越小相似度越高
                # 计算标准化的相似度分数，考虑向量维度和距离的实际范围
                
                # 方法1：基于距离的逆向计算 (适用于大多数情况)
                if distance <= 0:
                    similarity_score = 1.0  # 完全匹配
                else:
                    # 使用更合理的距离-相似度转换
                    # 对于文本嵌入，距离通常在0-10000+范围内
                    max_expected_distance = 15000.0  # 根据实际情况调整
                    similarity_score = max(0.0, 1.0 - (distance / max_expected_distance))
                
                # 方法2：如果上述方法结果仍然过低，使用更宽松的阈值判断
                # 对于实际应用，我们可以：
                # 1. 降低默认阈值到一个更合理的水平
                # 2. 或者基于距离排序，取前几个结果
                
                # 动态调整阈值：如果所有结果的相似度都很低，则取最好的几个
                meets_threshold = similarity_score >= similarity_threshold
                
                search_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": similarity_score,
                    "distance": distance,
                    "embedding_provider": self.embedding_provider,
                    "meets_threshold": meets_threshold
                })
            
            # 如果没有结果满足阈值，但有结果存在，则返回按距离排序的前几个结果
            if not any(result["meets_threshold"] for result in search_results) and search_results:
                logger.info(f"没有结果满足相似度阈值 {similarity_threshold}，返回距离最近的结果")
                # 按距离排序，取前几个
                search_results.sort(key=lambda x: x["distance"])
                # 标记为通过阈值（降级处理）
                for i, result in enumerate(search_results[:min(3, len(search_results))]):
                    result["meets_threshold"] = True
                    result["fallback_result"] = True  # 标记为降级结果
            
            # 只返回满足阈值的结果
            filtered_results = [r for r in search_results if r.get("meets_threshold", False)]
            
            logger.info(f"搜索到 {len(filtered_results)} 个相关文档")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
        
        return search_results
    
    async def add_documents_to_vector_store(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        将文档添加到向量存储
        
        Args:
            documents: 文档内容列表
            metadatas: 文档元数据列表
            
        Returns:
            是否成功添加
        """
        try:
            if not self.vector_store or not documents:
                logger.warning("向量存储未初始化或文档列表为空")
                return False
            
            if len(documents) != len(metadatas):
                logger.error("文档数量与元数据数量不匹配")
                return False
            
            # 为元数据添加嵌入模型信息并过滤复杂类型
            enhanced_metadatas = []
            for metadata in metadatas:
                enhanced_metadata = {
                    **metadata,
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": getattr(self.embeddings, 'model', 'unknown')
                }
                # 手动过滤复杂的元数据类型（Chroma只支持str, int, float, bool）
                filtered_metadata = {}
                for key, value in enhanced_metadata.items():
                    if isinstance(value, (str, bool, int, float)):
                        filtered_metadata[key] = value
                enhanced_metadatas.append(filtered_metadata)
            
            # 创建Document对象列表
            doc_objects = []
            for text, metadata in zip(documents, enhanced_metadatas):
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                doc_objects.append(doc)
            
            # 使用标准的add_documents方法添加文档
            ids = await asyncio.to_thread(
                self.vector_store.add_documents,
                documents=doc_objects
            )
            
            logger.info(f"成功使用 {self.embedding_provider} 添加 {len(doc_objects)} 个文档到向量存储，生成ID: {len(ids) if ids else 0}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            return False
    
    async def add_document_objects_to_vector_store(
        self,
        documents: List[Document]
    ) -> bool:
        """
        直接将Document对象添加到向量存储
        
        Args:
            documents: Document对象列表
            
        Returns:
            是否成功添加
        """
        try:
            if not self.vector_store or not documents:
                logger.warning("向量存储未初始化或文档列表为空")
                return False
            
            # 为每个文档的元数据添加嵌入模型信息并过滤复杂类型
            enhanced_documents = []
            for doc in documents:
                # 增强元数据
                enhanced_metadata = {
                    **doc.metadata,
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": getattr(self.embeddings, 'model', 'unknown')
                }
                
                # 过滤复杂的元数据类型
                filtered_metadata = {}
                for key, value in enhanced_metadata.items():
                    if isinstance(value, (str, bool, int, float)):
                        filtered_metadata[key] = value
                
                # 创建新的Document对象
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                )
                enhanced_documents.append(enhanced_doc)
            
            # 使用标准的add_documents方法添加文档
            ids = await asyncio.to_thread(
                self.vector_store.add_documents,
                documents=enhanced_documents
            )
            
            logger.info(f"成功使用 {self.embedding_provider} 添加 {len(enhanced_documents)} 个Document对象到向量存储，生成ID: {len(ids) if ids else 0}")
            return True
            
        except Exception as e:
            logger.error(f"添加Document对象到向量存储失败: {str(e)}")
            return False
    
    async def remove_documents_from_vector_store(self, file_id: str) -> bool:
        """
        从向量存储中移除指定文件的文档
        
        Args:
            file_id: 文件ID（字符串格式）
            
        Returns:
            是否成功移除
        """
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return False
            
            # 使用langchain_chroma的标准接口删除文档
            collection = self.vector_store._collection
            
            # 查询并删除指定文件ID的所有文档
            def _delete_by_file_id():
                # 查询该文件的所有文档ID
                results = collection.get(
                    where={"file_id": file_id}
                )
                
                if results['ids']:
                    # 删除找到的文档
                    collection.delete(ids=results['ids'])
                    return len(results['ids'])
                return 0
            
            deleted_count = await asyncio.to_thread(_delete_by_file_id)
            
            if deleted_count > 0:
                logger.info(f"成功从向量存储删除文件 {file_id} 的 {deleted_count} 个文档块")
                return True
            else:
                logger.warning(f"未找到文件 {file_id} 的文档")
                return True  # 没有找到也算成功
            
        except Exception as e:
            logger.error(f"从向量存储删除文档失败: {str(e)}")
            return False
    
    async def split_text_into_chunks(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 原始文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块后的文本列表
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
            )
            
            chunks = await asyncio.to_thread(
                text_splitter.split_text,
                text
            )
            
            logger.info(f"文本分割完成，共 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            logger.error(f"文本分割失败: {str(e)}")
            return [text]  # 如果分割失败，返回原文本
    
    async def create_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        创建LangChain文档对象
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            Document对象列表
        """
        try:
            if len(texts) != len(metadatas):
                raise ValueError("文本数量与元数据数量不匹配")
            
            documents = []
            for text, metadata in zip(texts, metadatas):
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"创建文档对象失败: {str(e)}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入维度
        
        Returns:
            嵌入向量的维度
        """
        try:
            if self.embedding_provider.lower() == 'openai':
                model = getattr(self.embeddings, 'model', 'text-embedding-3-small')
                if 'text-embedding-3-small' in model:
                    return 1536
                elif 'text-embedding-3-large' in model:
                    return 3072
                elif 'text-embedding-ada-002' in model:
                    return 1536
                else:
                    return 1536  # 默认值
            elif self.embedding_provider.lower() == 'qwen':
                return 1536  # Qwen text-embedding-v1 的维度
            else:
                return 1536  # 默认维度
                
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {str(e)}")
            return 1536  # 默认维度
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        获取嵌入模型信息
        
        Returns:
            包含嵌入模型信息的字典
        """
        return {
            "provider": self.embedding_provider,
            "model": getattr(self.embeddings, 'model', 'unknown'),
            "dimension": self.get_embedding_dimension(),
            "vector_store_type": settings.VECTOR_DB_TYPE,
            "vector_store_status": "initialized" if self.vector_store else "not_initialized"
        }
    
    async def test_embedding_connection(self) -> Dict[str, Any]:
        """
        测试嵌入模型连接
        
        Returns:
            测试结果
        """
        try:
            test_text = "这是一个测试文本"
            
            # 测试嵌入
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                test_text
            )
            
            return {
                "success": True,
                "provider": self.embedding_provider,
                "dimension": len(embedding),
                "test_text": test_text,
                "embedding_sample": embedding[:5]  # 只返回前5个维度作为示例
            }
            
        except Exception as e:
            logger.error(f"嵌入模型连接测试失败: {str(e)}")
            return {
                "success": False,
                "provider": self.embedding_provider,
                "error": str(e)
            }
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """
        获取支持的嵌入模型提供商
        
        Returns:
            支持的提供商列表
        """
        return ["openai", "qwen"]
    
    @classmethod
    def create_with_provider(cls, provider: str) -> "LLMRetrievalService":
        """
        使用指定提供商创建检索服务实例
        
        Args:
            provider: 嵌入模型提供商
            
        Returns:
            检索服务实例
        """
        return cls(embedding_provider=provider) 