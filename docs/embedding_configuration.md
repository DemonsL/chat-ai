# Embedding 模型配置指南

本项目支持多种 embedding 模型提供商，包括 OpenAI 和 Qwen（阿里云灵积）。

## 支持的 Embedding 提供商

### 1. OpenAI Embeddings
- **提供商代码**: `openai`
- **支持的模型**:
  - `text-embedding-3-small` (默认，1536维)
  - `text-embedding-3-large` (3072维)
  - `text-embedding-ada-002` (1536维)

### 2. Qwen Embeddings (DashScope)
- **提供商代码**: `qwen`
- **支持的模型**:
  - `text-embedding-v1` (默认，1536维)
  - `text-embedding-v2` (1536维)
- **需要依赖**: `langchain-community`, `dashscope` 库
- **实现**: 使用 LangChain 官方的 `DashScopeEmbeddings`

## 环境配置

### 基础配置
在 `.env` 文件中设置以下配置项：

```bash
# 选择嵌入模型提供商
EMBEDDING_PROVIDER="openai"  # 或 "qwen"
```

### OpenAI 配置
```bash
# OpenAI 配置
EMBEDDING_PROVIDER="openai"
EMBEDDING_MODEL="text-embedding-3-small"
OPENAI_API_KEY="your-openai-api-key"
```

### Qwen 配置
```bash
# Qwen/DashScope 配置
EMBEDDING_PROVIDER="qwen"
QWEN_EMBEDDING_MODEL="text-embedding-v1"
QWEN_API_KEY="your-qwen-api-key"
QWEN_BASE_URL="https://dashscope.aliyuncs.com/api/v1"
```

## 安装依赖

### 使用 OpenAI
```bash
pip install openai
```

### 使用 Qwen (DashScope)
```bash
pip install langchain-community dashscope
```

## 代码使用示例

### 创建检索服务实例

```python
from app.llm.rag.retrieval_service import LLMRetrievalService

# 使用默认配置创建（基于环境变量 EMBEDDING_PROVIDER）
retrieval_service = LLMRetrievalService()

# 或者显式指定提供商
openai_service = LLMRetrievalService.create_with_provider("openai")
qwen_service = LLMRetrievalService.create_with_provider("qwen")
```

### 获取模型信息

```python
# 获取当前嵌入模型信息
info = retrieval_service.get_embedding_info()
print(f"提供商: {info['provider']}")
print(f"模型: {info['model_name']}")
print(f"维度: {info['dimension']}")

# 测试连接
result = await retrieval_service.test_embedding_connection()
if result['success']:
    print("嵌入模型连接正常")
else:
    print(f"连接失败: {result['message']}")
```

### 支持的提供商列表

```python
# 获取所有支持的提供商
providers = LLMRetrievalService.get_supported_providers()
print(f"支持的提供商: {providers}")  # ['openai', 'qwen']
```

## 模型对比

| 提供商 | 模型 | 维度 | 特点 |
|--------|------|------|------|
| OpenAI | text-embedding-3-small | 1536 | 性能优异，成本较低 |
| OpenAI | text-embedding-3-large | 3072 | 最高精度，成本较高 |
| Qwen | text-embedding-v1 | 1536 | 中文优化，阿里云服务 |

## 注意事项

1. **API 密钥安全**: 请妥善保管 API 密钥，不要提交到代码仓库
2. **模型兼容性**: 不同提供商的嵌入向量不兼容，切换提供商需要重新生成向量
3. **成本考虑**: 不同模型的调用费用不同，请根据需求选择合适的模型
4. **网络连接**: Qwen 需要访问阿里云服务，确保网络连接正常

## 故障排除

### 常见错误

1. **ImportError: 依赖库未安装**
   ```bash
   pip install langchain-community dashscope
   ```

2. **API 密钥错误**
   - 检查 `.env` 文件中的密钥配置
   - 确保密钥有效且有足够的配额

3. **网络连接超时**
   - 检查网络连接
   - 确认 API 服务器地址正确

### 调试模式

启用详细日志以便调试：

```python
import logging
logging.getLogger("app.llm.rag.retrieval_service").setLevel(logging.DEBUG)
```

## 迁移指南

### 从 OpenAI 迁移到 Qwen

1. 安装 dashscope 依赖
2. 更新环境配置
3. 清空现有向量数据库
4. 重新处理文档以生成新的嵌入向量

```bash
# 1. 安装依赖
pip install langchain-community dashscope

# 2. 更新配置
echo "EMBEDDING_PROVIDER=qwen" >> .env
echo "QWEN_API_KEY=your-key" >> .env

# 3. 清空向量数据库
rm -rf ./chroma_db

# 4. 重新启动应用，重新处理文档
``` 