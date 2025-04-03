#!/bin/bash

# 脚本说明：启动开发环境

# 设置默认值
PORT=8000
HOST="0.0.0.0"
RELOAD=1
WORKERS=1
LOG_LEVEL="info"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --host=*)
      HOST="${1#*=}"
      shift
      ;;
    --no-reload)
      RELOAD=0
      shift
      ;;
    --workers=*)
      WORKERS="${1#*=}"
      shift
      ;;
    --log-level=*)
      LOG_LEVEL="${1#*=}"
      shift
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 准备Uvicorn命令
UVICORN_CMD="uvicorn app.main:app --host $HOST --port $PORT --workers $WORKERS --log-level $LOG_LEVEL"

# 如果开启热重载
if [ $RELOAD -eq 1 ]; then
  UVICORN_CMD="$UVICORN_CMD --reload"
fi

# 输出启动信息
echo "正在启动开发服务器..."
echo "主机: $HOST"
echo "端口: $PORT"
echo "热重载: $([ $RELOAD -eq 1 ] && echo '开启' || echo '关闭')"
echo "工作进程: $WORKERS"
echo "日志级别: $LOG_LEVEL"
echo "-----------------------------------"

# 执行命令
$UVICORN_CMD 