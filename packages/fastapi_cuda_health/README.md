# fastapi-cuda-health

CUDA-aware healthcheck plugin for FastAPI/Starlette. It tracks consecutive CUDA-related failures globally and exposes a unified `/health` route for readiness and liveness.

## Features

- Track CUDA errors over a sliding window and mark service unhealthy when the last N results are all CUDA failures
- One-call setup: initialize middleware + mount `/health`
- Global detection (Plan B):
  - Any `HTTPException` with status >= 500 whose `detail` contains CUDA keywords is a CUDA failure
  - Uncaught exceptions whose message contains CUDA keywords are treated as CUDA failures
  - Non-CUDA 5xx responses are recorded as healthy to avoid sticky unhealthy windows
- Optional suppression of `uvicorn.access` logs for `/health` and `/docs`
- Notifier hook for one-shot alerting when the service becomes unhealthy

## Installation

```bash
pip install --index-url <your-private-index> --extra-index-url https://pypi.org/simple fastapi-cuda-health
```

## Quick Start

```python
from fastapi import FastAPI
from fastapi_cuda_health import setup_cuda_health

app = FastAPI()

# Optional: readiness predicate (e.g., model initialized)
is_model_ready = False

def ready_predicate():
    return is_model_ready

# Install the plugin and mount /health
monitor = setup_cuda_health(
    app,
    path="/health",
    limit=None,  # or set an int; default reads env CONSECUTIVE_FAIL_LIMIT (default 3)
    notifier=None,  # optional object with .send_error_message(str)
    suppress_access_paths=("/health", "/docs"),
    ready_predicate=ready_predicate,
)

@app.post("/generate")
async def generate():
    # if you raise HTTPException(status_code=500, detail="CUDA error: ..."),
    # it will be counted as a CUDA failure automatically.
    return {"ok": True}

# after model loads successfully
# is_model_ready = True
```

## 快速开始（中文）

```python
from fastapi import FastAPI
from fastapi_cuda_health import setup_cuda_health

app = FastAPI()

# 可选：就绪判定函数（例如模型是否加载完成）
is_model_ready = False

def ready_predicate():
    return is_model_ready

# 安装插件并挂载 /health
monitor = setup_cuda_health(
    app,
    path="/health",
    limit=None,  # None 表示从环境变量 CONSECUTIVE_FAIL_LIMIT 读取，默认 3
    notifier=None,  # 可选：具有 send_error_message(str) 的通知对象
    suppress_access_paths=("/health", "/docs"),
    ready_predicate=ready_predicate,
)

@app.post("/generate")
async def generate():
    # 如果你抛出 HTTPException(status_code=500, detail="CUDA error: ...")，
    # 插件会自动计为一次 CUDA 失败。
    return {"ok": True}

# 模型加载完成后置为 True
# is_model_ready = True
```

- 若端点抛出的 HTTPException(>=500) 的 detail 含 CUDA 关键词（如“cuda”“cublas”“cudnn”等），会记为失败
- 若返回状态码 >= 500 但你在响应头设置 `x-cuda-error: 1`，也会记为 CUDA 失败
- 未捕获异常若消息含 CUDA 关键词，也会记为失败
- 非 CUDA 的 5xx 会被视为“健康事件”填充窗口，避免粘性不健康
- 最近 N 次（N=`limit` 或环境变量 `CONSECUTIVE_FAIL_LIMIT`）均为 CUDA 失败时判定服务不健康
- 当 `ready_predicate()` 返回 False 时，`/health` 返回 503 并提示“模型未初始化”

## Health Semantics

- If `ready_predicate()` returns False (or `is_ready=False` passed to `check_health`), `/health` returns `503` with detail "模型未初始化".
- When tracked endpoints raise exceptions that include CUDA-related keywords (e.g. "cuda", "cublas", "cudnn", etc.), they are recorded as failures.
- When a tracked endpoint returns an HTTP status code >= 500, you may explicitly mark it as a CUDA failure by setting header `x-cuda-error: 1`.
- The service is considered unhealthy if the last N tracked results (window size = `limit`) are all CUDA failures.

## Environment Variables

- `CONSECUTIVE_FAIL_LIMIT` (default: `3`): sliding window size N used to determine unhealthy status.

## Debug & Observability

- `x-cuda-error: 1` response header will be added automatically to 5xx HTTPException responses whose `detail` contains CUDA keywords. Middlewares and gateways can rely on this header for routing/alerting.
- A `TRACE {path}` endpoint is also mounted alongside `GET {path}` (default `path="/health"`). Sending a TRACE request returns additional diagnostic fields when healthy, which can help verify integration. Example:

  ```bash
  curl -i -X TRACE http://127.0.0.1:8000/health
  ```

  Note: when unhealthy, `/health` still returns `503` via GET; TRACE may not include extra info if an exception is raised.

## Notifications

- You may pass a custom `notifier` object with method `send_error_message(str)` to `setup_cuda_health()`; a one-shot alert is sent when health flips from healthy to unhealthy.
- If you don't pass a notifier explicitly, a default notifier may be used by the package (see `fastapi_cuda_health/notify.py`). If you don't want notifications, pass `notifier=None` explicitly in `setup_cuda_health()`.

## Public API

- `CudaHealthMonitor`
- `init_cuda_health_plugin(app, ...)`
- `check_health(app, is_ready: Optional[bool] = None, default_unhealthy_detail: str = "CUDA 错误导致不健康")`
- `mount_health_route(app, path: str = "/health")`
- `setup_cuda_health(app, ...)`

## Access Log Suppression

By default `suppress_access_paths=("/health", "/docs")` adds a filter to `uvicorn.access` logger, suppressing access lines containing these path strings.

## License

MIT
