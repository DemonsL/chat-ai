# 多模型支持的智能聊天应用

这是一个基于 FastAPI 和 LangChain 的多模型智能聊天应用后端，支持多种大语言模型的切换，并提供 RAG（检索增强生成）和 Agent（深度研究）能力。

## 功能特点

- 支持多种 LLM 模型（如 GPT-3.5/4、Claude 3 系列、DeepSeek 等）
- 用户管理（注册、登录、个人信息管理）
- 会话管理（创建、查看、设置）
- 文件上传与管理（支持 PDF、DOCX、TXT、图片）
- 文档内容的提取、处理和向量化
- 检索增强生成（基于用户上传文档的问答）
- 初步的 Agent 深度研究功能
- 流式响应输出
- 异步处理架构

## 技术栈

- **后端框架**：FastAPI
- **大语言模型**：通过 LangChain 集成 OpenAI、Anthropic、DeepSeek 等模型
- **数据库**：
  - PostgreSQL：关系型存储
  - Redis：缓存和队列
  - ChromaDB/Pinecone：向量存储
- **认证**：JWT
- **文件处理**：pypdf、python-docx、pytesseract
- **部署**：Docker、Docker Compose

## 项目结构

```
.
├── app/                     # 主应用目录
│   ├── api/                 # API 层
│   ├── core/                # 核心配置
│   ├── db/                  # 数据库模型和仓库
│   ├── llm/                 # LLM 相关功能
│   ├── schemas/             # Pydantic 数据验证模型
│   ├── services/            # 服务层
│   ├── tasks/               # 后台任务
│   └── utils/               # 通用工具
├── alembic/                 # 数据库迁移
├── tests/                   # 测试目录
├── .env.example             # 环境变量示例
├── Dockerfile               # Docker 配置
├── docker-compose.yml       # Docker Compose 配置
└── pyproject.toml           # 项目依赖
```

## 安装与运行

### 前提条件

- Python 3.11+
- PostgreSQL
- Redis
- Docker 和 Docker Compose (可选)

### 本地开发

1. 克隆仓库
```bash
git clone <仓库地址>
cd multi-llm-chat-app
```

2. 创建并激活虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install poetry
poetry install
```

4. 创建 `.env` 文件
```bash
cp .env.example .env
# 编辑 .env 文件，填入必要的环境变量
```

5. 运行数据库迁移
```bash
alembic upgrade head
```

6. 启动应用
```bash
uvicorn app.main:app --reload
```

### Docker 部署

```bash
docker-compose up -d
```

## API 文档

应用启动后，访问以下地址查看 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 环境变量配置

关键环境变量说明（详见 `.env.example`）：

- `POSTGRES_*`: PostgreSQL 数据库配置
- `REDIS_*`: Redis 配置
- `OPENAI_API_KEY`: OpenAI API 密钥
- `ANTHROPIC_API_KEY`: Anthropic API 密钥
- `VECTOR_DB_TYPE`: 向量数据库类型 (chroma/pinecone)

## 开发指南

- API 端点定义位于 `app/api/v1/endpoints/` 目录
- 数据库模型定义位于 `app/db/models/` 目录
- LLM 相关实现位于 `app/llm/` 目录

## 协议

[MIT](LICENSE) 