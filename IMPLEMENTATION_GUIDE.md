# LLM模块优化实施指南

## 📋 概述

本指南将帮助您逐步实施LLM模块的优化改进，确保平滑过渡和最小化风险。

## 🚀 快速开始

### 前置条件

1. **系统要求**
   - Docker & Docker Compose
   - Python 3.11+
   - 至少16GB RAM
   - 100GB可用磁盘空间

2. **依赖安装**
   ```bash
   # 安装新的依赖包
   pip install rank-bm25 sentence-transformers jieba
   pip install prometheus-client opentelemetry-api
   pip install elasticsearch kibana
   ```

## 📅 分阶段实施计划

### 第一阶段：基础优化 (第1-2周)

#### 1.1 部署意图跟踪器

```bash
# 1. 复制意图跟踪器文件
cp app/llm/enhance/intent_tracker.py app/llm/enhance/

# 2. 在LLM管理器中集成
# 编辑 app/llm/manage.py，添加意图跟踪功能
```

**集成代码示例**：
```python
# 在 LLMManager.__init__ 中添加
from app.llm.enhance.intent_tracker import IntentTracker

def __init__(self, retrieval_service=None):
    # ... 现有代码 ...
    self.intent_tracker = IntentTracker()
    
# 在对话处理中使用
async def process_conversation(self, messages, ...):
    # 跟踪意图
    if len(messages) > 1:
        intent_update = await self.intent_tracker.track_intent(
            current_message=messages[-1]["content"],
            conversation_history=convert_messages_to_langchain(messages[:-1])
        )
        metadata["intent_analysis"] = intent_update
```

#### 1.2 部署质量评估器

```bash
# 1. 复制质量评估器文件
cp app/llm/enhance/quality_evaluator.py app/llm/enhance/

# 2. 创建质量评估API端点
```

**API集成示例**：
```python
# 在 app/api/v1/endpoints/ 中创建新端点
from app.llm.enhance.quality_evaluator import ConversationQualityEvaluator

@router.post("/evaluate-quality")
async def evaluate_quality(
    query: str,
    response: str,
    context: Dict = None
):
    evaluator = ConversationQualityEvaluator()
    report = await evaluator.evaluate_response_quality(query, response, context)
    return report
```

#### 1.3 配置监控系统

```bash
# 1. 创建监控配置目录
mkdir -p monitoring/grafana/{dashboards,datasources}
mkdir -p monitoring/prometheus

# 2. 配置Prometheus
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['llm-api:8000']
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

# 3. 启动基础监控
docker-compose -f docker-compose.enhanced.yml up -d prometheus grafana
```

### 第二阶段：检索增强 (第3-4周)

#### 2.1 部署混合检索器

```bash
# 1. 复制混合检索器文件
cp app/llm/enhance/hybrid_retriever.py app/llm/enhance/

# 2. 安装BM25依赖
pip install rank-bm25

# 3. 配置Elasticsearch
docker-compose -f docker-compose.enhanced.yml up -d elasticsearch kibana
```

**集成混合检索**：
```python
# 在检索服务中集成
from app.llm.enhance.hybrid_retriever import HybridRetriever, SearchConfig

class LLMRetrievalService:
    def __init__(self):
        # ... 现有代码 ...
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            config=SearchConfig(
                strategy="hybrid",
                vector_weight=0.7,
                keyword_weight=0.3
            )
        )
    
    async def retrieve_documents(self, query: str, **kwargs):
        # 使用混合检索
        results = await self.hybrid_retriever.search(query, **kwargs)
        return [result.document for result in results]
```

#### 2.2 优化文档分块

```python
# 创建自适应分块器
class AdaptiveDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", " "]
        )
        
    def process_document(self, document: Document) -> List[Document]:
        # 检测文档类型
        doc_type = self._detect_document_type(document.page_content)
        
        if doc_type == "code":
            return self._split_code_document(document)
        elif doc_type == "academic":
            return self._split_academic_document(document)
        else:
            return self.text_splitter.split_documents([document])
```

### 第三阶段：性能优化 (第5-6周)

#### 3.1 实施多级缓存

```python
# 创建缓存管理器
import redis
from functools import lru_cache

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis.from_url("redis://redis:6379")
        self.memory_cache = {}
        
    @lru_cache(maxsize=1000)
    def get_embedding_cache(self, text_hash: str):
        """内存缓存嵌入向量"""
        return self.redis_client.get(f"embedding:{text_hash}")
    
    async def cache_response(self, query_hash: str, response: str):
        """缓存响应结果"""
        # L1: 内存缓存
        self.memory_cache[query_hash] = response
        
        # L2: Redis缓存
        await self.redis_client.setex(
            f"response:{query_hash}", 
            3600,  # 1小时TTL
            response
        )
```

#### 3.2 批量处理优化

```python
# 批量嵌入处理
class BatchEmbeddingProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.embedding_queue = asyncio.Queue()
        
    async def batch_embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._process_batch(batch)
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    async def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """处理单个批次"""
        # 使用embedding服务的批量API
        return await self.embedding_service.embed_documents(batch)
```

### 第四阶段：企业级部署 (第7-8周)

#### 4.1 微服务架构改造

```bash
# 1. 创建独立的微服务
mkdir -p services/{quality-evaluator,intent-tracker,hybrid-retriever}

# 2. 为每个服务创建Dockerfile
cat > services/quality-evaluator/Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/llm/enhance/quality_evaluator.py .
COPY service_main.py .

CMD ["python", "service_main.py"]
EOF

# 3. 创建服务注册中心
# 使用Consul或etcd进行服务发现
```

#### 4.2 容器化部署

```bash
# 1. 构建所有服务镜像
docker-compose -f docker-compose.enhanced.yml build

# 2. 启动完整系统
docker-compose -f docker-compose.enhanced.yml up -d

# 3. 验证服务状态
docker-compose -f docker-compose.enhanced.yml ps
```

## 🔧 配置管理

### 环境变量配置

```bash
# .env.production
POSTGRES_URL=postgresql://chatai:chatai123@postgres:5432/chatai
REDIS_URL=redis://redis:6379
CHROMA_URL=http://chroma:8000
ELASTICSEARCH_URL=http://elasticsearch:9200

# 功能开关
ENABLE_QUALITY_EVALUATION=true
ENABLE_INTENT_TRACKING=true
ENABLE_HYBRID_RETRIEVAL=true
ENABLE_CACHING=true

# 性能配置
MAX_CONCURRENT_REQUESTS=100
CACHE_TTL=3600
BATCH_SIZE=32
```

### Nginx负载均衡配置

```nginx
# nginx/nginx.conf
upstream llm_api {
    least_conn;
    server llm-api:8000 max_fails=3 fail_timeout=30s;
    server llm-api:8000 max_fails=3 fail_timeout=30s;
    server llm-api:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://llm_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 超时配置
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://llm_api/health;
    }
}
```

## 📊 监控与告警

### Grafana仪表板配置

```json
{
  "dashboard": {
    "title": "LLM System Monitoring",
    "panels": [
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Quality Scores",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(quality_score_overall)",
            "legendFormat": "Average Quality"
          }
        ]
      }
    ]
  }
}
```

### 告警规则配置

```yaml
# monitoring/alert-rules.yml
groups:
  - name: llm_system_alerts
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: LowQualityScore
        expr: avg(quality_score_overall) < 0.7
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Quality score below threshold"
```

## 🧪 测试与验证

### 功能测试

```python
# tests/test_optimizations.py
import pytest
from app.llm.enhance.intent_tracker import IntentTracker
from app.llm.enhance.quality_evaluator import ConversationQualityEvaluator

@pytest.mark.asyncio
async def test_intent_tracking():
    tracker = IntentTracker()
    
    # 测试意图跟踪
    result = await tracker.track_intent(
        current_message="继续上面的话题",
        conversation_history=[]
    )
    
    assert result.intent_type == "task_continuation"
    assert result.confidence > 0.5

@pytest.mark.asyncio
async def test_quality_evaluation():
    evaluator = ConversationQualityEvaluator()
    
    # 测试质量评估
    report = await evaluator.evaluate_response_quality(
        user_query="什么是人工智能？",
        response="人工智能是计算机科学的一个分支..."
    )
    
    assert report.score.relevance > 0.7
    assert len(report.suggestions) >= 0
```

### 性能测试

```python
# tests/test_performance.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def test_concurrent_requests():
    """测试并发请求性能"""
    
    async def make_request():
        # 模拟API请求
        start_time = time.time()
        # ... 执行请求 ...
        return time.time() - start_time
    
    # 并发100个请求
    tasks = [make_request() for _ in range(100)]
    response_times = await asyncio.gather(*tasks)
    
    avg_response_time = sum(response_times) / len(response_times)
    assert avg_response_time < 2.0  # 平均响应时间小于2秒
```

## 🚨 故障排除

### 常见问题及解决方案

1. **内存不足**
   ```bash
   # 增加Docker内存限制
   docker-compose -f docker-compose.enhanced.yml up -d --scale llm-api=2
   ```

2. **数据库连接超时**
   ```python
   # 增加连接池大小
   SQLALCHEMY_POOL_SIZE=20
   SQLALCHEMY_MAX_OVERFLOW=30
   ```

3. **Redis缓存满了**
   ```bash
   # 清理Redis缓存
   docker exec -it redis redis-cli FLUSHDB
   ```

### 日志分析

```bash
# 查看系统日志
docker-compose -f docker-compose.enhanced.yml logs -f llm-api

# 查看特定服务日志
docker-compose -f docker-compose.enhanced.yml logs quality-evaluator

# 使用ELK Stack分析日志
curl -X GET "elasticsearch:9200/_search?q=level:ERROR"
```

## 📈 性能优化建议

### 1. 数据库优化

```sql
-- 创建索引优化查询
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_files_user_id ON files(user_id);

-- 分区表优化
CREATE TABLE conversations_2024 PARTITION OF conversations
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### 2. 缓存策略优化

```python
# 智能缓存失效
class SmartCache:
    def __init__(self):
        self.cache_stats = {}
        
    def should_cache(self, key: str, content: str) -> bool:
        """智能决定是否缓存"""
        # 基于内容长度、访问频率等决定
        if len(content) > 10000:  # 大内容优先缓存
            return True
        if self.cache_stats.get(key, 0) > 5:  # 高频访问缓存
            return True
        return False
```

### 3. 模型路由优化

```python
# 智能模型选择
class ModelRouter:
    def select_model(self, query: str, context: Dict) -> str:
        """根据查询复杂度选择模型"""
        complexity = self._analyze_complexity(query)
        
        if complexity < 0.3:
            return "gpt-3.5-turbo"  # 简单查询用轻量模型
        elif complexity > 0.8:
            return "gpt-4"          # 复杂查询用高端模型
        else:
            return "claude-3-sonnet" # 中等查询用平衡模型
```

## 🎯 成功指标

### 关键性能指标 (KPIs)

1. **响应时间**: < 2秒 (95th percentile)
2. **质量评分**: > 0.8 (平均分)
3. **用户满意度**: > 85%
4. **系统可用性**: > 99.9%
5. **错误率**: < 1%

### 监控仪表板

访问以下URL查看系统状态：
- Grafana: http://localhost:3000 (admin/admin123)
- Prometheus: http://localhost:9090
- Kibana: http://localhost:5601
- Jaeger: http://localhost:16686
- Flower: http://localhost:5555

## 📚 参考资料

1. [LangChain官方文档](https://python.langchain.com/)
2. [LangGraph指南](https://langchain-ai.github.io/langgraph/)
3. [Docker Compose最佳实践](https://docs.docker.com/compose/production/)
4. [Prometheus监控指南](https://prometheus.io/docs/guides/)
5. [Elasticsearch搜索优化](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-optimization.html)

---

通过遵循这个实施指南，您可以逐步将LLM系统升级为企业级的智能对话平台，实现更好的性能、可靠性和用户体验。 