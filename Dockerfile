# 使用Python 3.9作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml .
COPY src/ ./src/
COPY run_server.py .
COPY test_chat.py .
COPY test_api.py .

# 安装Python依赖
RUN pip install --no-cache-dir -e .

# 暴露端口
EXPOSE 8123

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8123/health || exit 1

# 启动命令
CMD ["python", "run_server.py"]
