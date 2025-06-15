FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    libtesseract-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目配置文件和README
COPY pyproject.toml README.md /app/

# 复制应用代码（为了构建需要）
COPY app/ /app/app/

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 创建目录
RUN mkdir -p /app/uploads /app/chroma_db

# 复制剩余的应用代码
COPY . /app/

# 设置环境变量
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TESSERACT_CMD=/usr/bin/tesseract

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 