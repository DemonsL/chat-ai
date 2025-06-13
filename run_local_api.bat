@echo off
chcp 65001 >nul

echo 🚀 启动本地API开发服务...
echo.
echo 📋 确保Docker服务正在运行：
echo   - Redis (localhost:6379)
echo   - PostgreSQL (localhost:5433) 
echo   - RabbitMQ (localhost:5672)
echo   - MinIO (localhost:9000)
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 设置环境变量
set REDIS_HOST=localhost
set REDIS_PORT=6380
set POSTGRES_SERVER=localhost
set POSTGRES_PORT=5433
set POSTGRES_USER=postgres
set POSTGRES_PASSWORD=postgres
set POSTGRES_DB=chatai
set CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
set CELERY_RESULT_BACKEND=redis://localhost:6380/0
set MINIO_ENDPOINT=localhost:9000
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin
set DEBUG=True
set SECRET_KEY=dev-secret-key-change-in-production

echo 🔧 环境变量设置完成
echo 📡 Redis: %REDIS_HOST%:%REDIS_PORT%
echo 🗄️ PostgreSQL: %POSTGRES_SERVER%:%POSTGRES_PORT%
echo 🐰 RabbitMQ: localhost:5672
echo 📦 MinIO: %MINIO_ENDPOINT%
echo.

echo 🚀 启动API服务...
echo 📄 API文档: http://localhost:8000/docs
echo.
python run_local_api.py 