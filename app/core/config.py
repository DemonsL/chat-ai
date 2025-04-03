import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, Field, PostgresDsn, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

    PROJECT_NAME: str = "多模型支持的智能聊天应用"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "dev"

    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    # 默认过期时间为 7 天
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # CORS 配置
    CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # 数据库配置
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "chatapp"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None

    # @validator("DATABASE_URI", pre=True)
    # def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
    #     if isinstance(v, str):
    #         return v
    #     return PostgresDsn.build(
    #         scheme="postgresql+asyncpg",
    #         username=values.get("POSTGRES_USER"),
    #         password=values.get("POSTGRES_PASSWORD"),
    #         host=values.get("POSTGRES_SERVER"),
    #         port=values.get("POSTGRES_PORT"),
    #         path=f"/{values.get('POSTGRES_DB') or ''}",
    #     )

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    # Vector 数据库配置
    VECTOR_DB_TYPE: str = "chroma"  # 或 "pinecone"
    CHROMA_DB_DIR: str = "./chroma_db"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None

    # LLM 配置
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None

    # 文件上传配置
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 20 * 1024 * 1024  # 20MB

    # 嵌入模型配置
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI 嵌入模型


settings = Settings()
