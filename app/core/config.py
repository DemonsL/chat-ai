import secrets
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    PROJECT_NAME: str = "多模型支持的智能聊天应用"
    DESCRIPTION: str = "基于FastAPI的AI聊天应用API"
    VERSION: str = "0.1.0"

    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "dev"

    # CORS 配置
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", 
        "http://localhost:8000", 
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000", 
        "http://127.0.0.1:8080",
        "http://localhost:5173",  # Vite 默认端口
        "http://127.0.0.1:5173",
        "http://localhost:5000",  # 其他常用端口
        "http://127.0.0.1:5000"
    ]

    # JWT设置
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    # 默认过期时间为 7 天
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # 数据库配置
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "chatapp"
    DATABASE_URL: Optional[str] = None
    POOL_SIZE: Optional[int] = 5
    MAX_OVERFLOW: Optional[int] = 10

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    # Vector 数据库配置
    VECTOR_DB_TYPE: str = "chroma"
    CHROMA_DB_DIR: str = "data/chroma_db"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None

    # LLM 配置
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None

    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_EMBEDDING_MODEL: str = "models/text-embedding-004"
    OPENAI_GOOGLE_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # Qwen/DashScope 配置
    QWEN_API_KEY: Optional[str] = None
    DASHSCOPE_API_KEY: Optional[str] = None  # 阿里云灵积API密钥（与QWEN_API_KEY相同）
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/api/v1"

    # 文件上传配置
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 20 * 1024 * 1024  # 20MB

    # 嵌入模型配置
    EMBEDDING_PROVIDER: str = "google_genai"  # 支持 'openai', 'qwen'
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI 嵌入模型
    QWEN_EMBEDDING_MODEL: str = "text-embedding-v1"  # Qwen 嵌入模型

    # Sentry设置
    SENTRY_DSN: Optional[str] = None

    # Celery设置
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6380/0"

    USE_POSTGRES_CHECKPOINTER: bool = True
    POSTGRES_DATABASE_URL: Optional[str] = None
    # Postgres Configuration
    POSTGRES_URL: str = ""
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 10
    CHECKPOINT_TABLES: List[str] = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]

    # 文档检索安全配置
    ENABLE_DOCUMENT_ISOLATION: bool = True  # 是否启用文档隔离
    ISOLATION_STRATEGY: str = "metadata_filter"  # 隔离策略: metadata_filter, multi_tenant, namespace
    STRICT_USER_ISOLATION: bool = True  # 是否严格验证用户隔离
    ENABLE_CROSS_USER_ACCESS_LOG: bool = True  # 是否记录跨用户访问日志
    MAX_DOCUMENTS_PER_USER: int = 10000  # 每个用户最大文档数
    MAX_DOCUMENTS_PER_CONVERSATION: int = 1000  # 每个对话最大文档数
    
    # 检索性能配置
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.3  # 默认相似度阈值
    MAX_RETRIEVAL_RESULTS: int = 50  # 最大检索结果数
    ENABLE_RETRIEVAL_CACHE: bool = True  # 是否启用检索缓存
    CACHE_TTL_SECONDS: int = 300  # 缓存过期时间（秒）

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
