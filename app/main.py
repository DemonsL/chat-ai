import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.exceptions import setup_exception_handlers
from app.api.middlewares import RequestLoggingMiddleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.events import shutdown_event_handler, startup_event_handler
from app.core.logging import setup_logging
from app.llm.core.config import init_model_configs
from app.llm.core.factory import register_default_providers
from app.monitoring.metrics import setup_metrics

# 设置日志系统
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时执行
    # 创建必要的目录
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)

    # 注册默认LLM提供商
    register_default_providers()

    # 初始化模型配置
    await init_model_configs()

    logger.info("应用程序启动")
    yield
    # 关闭时执行
    logger.info("应用程序关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)


# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 添加中间件
app.add_middleware(RequestLoggingMiddleware)


# 配置异常处理
setup_exception_handlers(app)


# 设置监控
setup_metrics(app)


# 注册事件处理器
app.add_event_handler("startup", startup_event_handler)
app.add_event_handler("shutdown", shutdown_event_handler)


# 包含API路由
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """根路径响应"""
    return {
        "message": f"欢迎使用 {settings.PROJECT_NAME} API",
        "docs": f"{settings.API_V1_STR}/docs",
        "version": settings.VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
