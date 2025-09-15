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

COPY . /app
RUN find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + || true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=3m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "api.py", "--model_dir", "/app/checkpoints", "--host", "0.0.0.0", "--port", "8000", "--fp16"]