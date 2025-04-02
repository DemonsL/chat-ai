from fastapi import APIRouter

from app.api.v1.endpoints import auth, conversations, files, health, messages, models, users

api_router = APIRouter()

# 注册各个路由
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
api_router.include_router(messages.router, prefix="/messages", tags=["messages"])
api_router.include_router(files.router, prefix="/files", tags=["files"]) 