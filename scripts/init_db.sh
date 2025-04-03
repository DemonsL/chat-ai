#!/bin/bash

# 脚本说明：初始化数据库和运行迁移

# 设置默认值
CREATE_DB=0
RUN_MIGRATIONS=1
SEED_DATA=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --create-db)
      CREATE_DB=1
      shift
      ;;
    --no-migrations)
      RUN_MIGRATIONS=0
      shift
      ;;
    --seed)
      SEED_DATA=1
      shift
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 检查环境变量
if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DB" ]; then
  echo "请设置必要的环境变量: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB"
  echo "您可以在.env文件中设置这些变量，然后运行: source .env"
  exit 1
fi

# 如果需要创建数据库
if [ $CREATE_DB -eq 1 ]; then
  echo "创建数据库..."
  # 检查是否安装了psql
  if ! command -v psql &> /dev/null; then
    echo "错误: 未找到 psql 命令，请安装 PostgreSQL 客户端工具"
    exit 1
  fi
  
  # 创建数据库
  PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_SERVER -U $POSTGRES_USER -p $POSTGRES_PORT -c "CREATE DATABASE $POSTGRES_DB WITH ENCODING 'UTF8' LC_COLLATE='en_US.UTF-8' LC_CTYPE='en_US.UTF-8' TEMPLATE=template0;"
  
  if [ $? -ne 0 ]; then
    echo "创建数据库失败"
    exit 1
  fi
  
  echo "数据库 $POSTGRES_DB 创建成功"
fi

# 如果需要运行迁移
if [ $RUN_MIGRATIONS -eq 1 ]; then
  echo "运行数据库迁移..."
  alembic upgrade head
  
  if [ $? -ne 0 ]; then
    echo "数据库迁移失败"
    exit 1
  fi
  
  echo "数据库迁移完成"
fi

# 如果需要填充种子数据
if [ $SEED_DATA -eq 1 ]; then
  echo "填充种子数据..."
  python -m app.seed_data
  
  if [ $? -ne 0 ]; then
    echo "填充种子数据失败"
    exit 1
  fi
  
  echo "种子数据填充完成"
fi

echo "数据库初始化完成！" 