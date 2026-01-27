# syntax=docker/dockerfile:1.7-labs
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG UV_VERSION=0.4.23
ARG PYPI_INDEX="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf_cache \
    MODELSCOPE_CACHE=/opt/modelscope_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      pkg-config \
      libsndfile1 \
      ffmpeg \
      libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv==${UV_VERSION}"
RUN mkdir -p /opt/hf_cache /opt/modelscope_cache
RUN mkdir -p /app/indextts 
RUN touch /app/indextts/__init__.py

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache \
    uv sync --extra webui

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache \
    uv sync --all-extras
RUN uv pip install flash-attn --no-build-isolation

RUN uv pip install -n "ttd_fastapi_utils>=0.3.0" --index-url http://pypi-server/simple/ --trusted-host pypi-server

COPY . /app
RUN find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + || true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=10m --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV TTD_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4e5f3845-cd5f-43a0-b004-231e710c68e4"

CMD ["python", "api.py", "--model_dir", "/app/checkpoints", "--host", "0.0.0.0", "--port", "8000", "--fp16"]