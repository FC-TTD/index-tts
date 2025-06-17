FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN --mount src=requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir -r /tmp/requirements.txt

# 先安装依赖
RUN pip install --no-cache-dir pyloudnorm scipy soundfile soxr fastapi[standard]

# 复制应用代码
COPY . /app
WORKDIR /app


# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 设置环境变量
ENV MODEL_DIR=/checkpoints

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "api.py", "--model_dir", "/app/checkpoints", "--host", "0.0.0.0", "--port", "8000"]
