#!/bin/bash

# 启动所有服务的脚本

echo "🚀 启动聊天AI应用服务..."

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker"
    exit 1
fi

# 检查docker-compose是否存在
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 停止现有服务（如果有）
echo "🛑 停止现有服务..."
docker-compose down

# 构建并启动所有服务
echo "🔨 构建并启动服务..."
docker-compose up --build -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "📊 检查服务状态..."
docker-compose ps

echo ""
echo "✅ 服务启动完成！"
echo ""
echo "🌐 可用的服务："
echo "  - API服务: http://localhost:8000"
echo "  - API文档: http://localhost:8000/docs"
echo "  - Flower监控: http://localhost:5555"
echo "  - RabbitMQ管理: http://localhost:15672 (guest/guest)"
echo "  - PostgreSQL: localhost:5433"
echo "  - Redis: localhost:6380"
echo "  - MinIO: http://localhost:9001"
echo ""
echo "📝 查看日志："
echo "  docker-compose logs -f [service_name]"
echo ""
echo "🛑 停止服务："
echo "  docker-compose down" 