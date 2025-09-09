FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（用于依赖管理，使用 pip 以避免路径问题）
RUN pip install --no-cache-dir uv

# 仅复制依赖清单，加速构建缓存
COPY pyproject.toml uv.lock /app/

# 使用 uv 安装依赖（遵循 README v2）
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

RUN uv sync --frozen --no-dev

# 安装 API 运行时依赖（不修改 pyproject，直接安装到虚拟环境）
RUN uv add \
    fastapi[standard]>=0.115.0 \
    uvicorn>=0.30.0 \
    soundfile>=0.12.1 \
    soxr>=0.3.7 \
    scipy>=1.11.0 \
    pyloudnorm>=0.1.1

# 复制应用代码
COPY . /app
WORKDIR /app


# 健康检查(3分钟后开始检查)
HEALTHCHECK --interval=30s --timeout=30s --start-period=3m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 设置环境变量
ENV MODEL_DIR=/app/checkpoints \
    PATH=/app/.venv/bin:$PATH

# 暴露端口
EXPOSE 8000

# 启动命令（使用 uv 运行），默认启用 fp16 与 BigVGAN 自定义 CUDA 内核（适配 RTX 3090）
CMD ["uv", "run", "python", "api.py", "--model_dir", "/app/checkpoints", "--host", "0.0.0.0", "--port", "8000", "--fp16", "--use_cuda_kernel"]
