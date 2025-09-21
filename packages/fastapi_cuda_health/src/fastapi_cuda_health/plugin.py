"""
fastapi-cuda-health
====================

A lightweight FastAPI plugin that monitors consecutive CUDA failures and exposes a health endpoint.

Design (Plan B):
- Global detection without endpoint annotations. Any 5xx HTTPException whose detail contains CUDA keywords
  is considered a CUDA failure. Uncaught exceptions containing CUDA keywords are also counted.
- Non-CUDA 5xx responses are recorded as healthy to avoid sticky unhealthy windows.
- Health becomes unhealthy (HTTP 503) only when the last N results (default 3) are all CUDA failures.
- The plugin also suppresses uvicorn.access logs for given paths (by default: /health and /docs).

Typical usage:
    from fastapi import FastAPI
    from fastapi_cuda_health.plugin import setup_cuda_health

    app = FastAPI()
    setup_cuda_health(app, path="/health", ready_predicate=lambda: model_is_ready())

Configuration:
- Threshold N is read from env CONSECUTIVE_FAIL_LIMIT (default 3) or can be passed explicitly.
- You can provide a notifier object with send_error_message(str) for one-time notifications when health flips unhealthy.

This module exposes:
- CudaHealthMonitor: internal state tracker for results
- init_cuda_health_plugin: initialize middleware and exception handler
- check_health: helper for /health logic
- mount_health_route: register a GET /health route
- setup_cuda_health: one-call setup that does init + mount
"""

from collections import deque
import logging
import os
import time
from typing import Iterable, Optional, Tuple, Callable

from fastapi import HTTPException
from fastapi.exception_handlers import http_exception_handler
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


CUDA_KEYWORDS = ("cuda", "cublas", "cudnn", "device-side", "illegal memory")


class CudaHealthMonitor:
    """Track recent inference results and determine health state.

    Parameters:
    - limit: int
        Consecutive CUDA failure threshold. Recent window size; unhealthy if last `limit` are all failures.
    - notifier: Optional[object]
        Optional object providing `send_error_message(str)`; invoked once when flipping healthy -> unhealthy.
    """

    def __init__(self, limit: int = 3, notifier: Optional[object] = None):
        self.limit = max(1, int(limit or 3))
        self._results: deque[bool] = deque(maxlen=self.limit)
        self._last_error_message: Optional[str] = None
        self._last_error_ts: Optional[float] = None
        self._unhealthy_notified: bool = False
        if not notifier:
            from .notify import notifier as default_notifier
            notifier = default_notifier
        self._notifier = notifier

    @staticmethod
    def is_cuda_error(exc: Exception) -> bool:
        msg = str(exc)
        msg_l = msg.lower()
        return any(k in msg_l for k in CUDA_KEYWORDS)

    def record_success(self) -> None:
        self._results.append(True)

    def record_error(self, exc: Exception) -> None:
        if self.is_cuda_error(exc):
            self._results.append(False)
            self._last_error_message = str(exc)
            self._last_error_ts = time.time()
        else:
            # Non-CUDA errors don't affect health but fill the window to avoid sticky unhealthy
            self._results.append(True)

    def record_cuda_failure(self, message: Optional[str] = None) -> None:
        self._results.append(False)
        if message:
            self._last_error_message = message
        self._last_error_ts = time.time()

    def is_unhealthy(self) -> bool:
        n = self.limit
        return len(self._results) >= n and all(r is False for r in list(self._results)[-n:])

    def evaluate_and_maybe_notify(self) -> Tuple[bool, Optional[str]]:
        unhealthy = self.is_unhealthy()
        detail = None
        if unhealthy:
            if not self._unhealthy_notified:
                detail = f"最近 {self.limit} 次推理均为 CUDA 错误。"
                if self._last_error_message:
                    detail += f" 最后错误: {self._last_error_message}"
                try:
                    if self._notifier is not None:
                        self._notifier.send_error_message(f"/health 不健康：{detail}")
                except Exception:
                    logger.exception("发送不健康通知失败")
                self._unhealthy_notified = True
        else:
            self._unhealthy_notified = False
        return (not unhealthy), detail


def _build_middleware(monitor: CudaHealthMonitor):
    """Create a Starlette middleware that updates the monitor per response.

    Behavior:
    - response < 500 => record_success()
    - response >= 500 and header x-cuda-error=1 => record_cuda_failure()
    - response >= 500 and no CUDA marker => record_success() (window fill to avoid sticky unhealthy)
    - uncaught exceptions are classified by keyword (monitor.record_error)
    """
    async def dispatch(request: Request, call_next: Callable[[Request], Response]):
        try:
            response = await call_next(request)
            # Healthy when <500; 5xx without CUDA marker also treated as healthy to avoid sticky unhealthy
            if response.status_code < 500:
                monitor.record_success()
            else:
                if response.headers.get("x-cuda-error") == "1":
                    monitor.record_cuda_failure()
                else:
                    monitor.record_success()
            return response
        except Exception as exc:  # noqa: BLE001
            try:
                # Uncaught exception path: count as failure only if CUDA keywords present
                monitor.record_error(exc)
            except Exception:
                logger.exception("记录推理结果失败")
            raise
    return dispatch


def init_cuda_health_plugin(
    app,
    limit: Optional[int] = None,
    notifier: Optional[object] = None,
    suppress_access_paths: Optional[Iterable[str]] = ("/health", "/docs"),
    ready_predicate: Optional[Callable[[], bool]] = None,
) -> CudaHealthMonitor:
    """Initialize core pieces (middleware + HTTPException handler) and attach monitor to app.state.

    Parameters:
    - app: FastAPI
        The application to install the plugin on.
    - limit: Optional[int]
        Consecutive CUDA failure threshold; if None, read from env CONSECUTIVE_FAIL_LIMIT (default 3).
    - notifier: Optional[object]
        Optional object providing `send_error_message(str)`; used when flipping healthy -> unhealthy.
    - suppress_access_paths: Optional[Iterable[str]]
        Paths to suppress from uvicorn.access logs; default ("/health", "/docs").
    - ready_predicate: Optional[Callable[[], bool]]
        Optional readiness predicate stored at app.state.cuda_health_ready_predicate used by check_health.

    Returns:
    - CudaHealthMonitor: the monitor instance stored on app.state.cuda_health_monitor
    """
    if limit is None:
        try:
            limit = int(os.getenv("CONSECUTIVE_FAIL_LIMIT", "3"))
        except Exception:
            limit = 3
    monitor = CudaHealthMonitor(limit=limit, notifier=notifier)

    app.add_middleware(BaseHTTPMiddleware, dispatch=_build_middleware(monitor))

    if suppress_access_paths:
        try:
            paths = tuple(suppress_access_paths)

            class _AccessPathSuppressFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    try:
                        msg = record.getMessage()
                    except Exception:
                        return True
                    return all(p not in msg for p in paths)

            logging.getLogger("uvicorn.access").addFilter(_AccessPathSuppressFilter())
            logger.info(f"已抑制 uvicorn.access 日志路径: {paths}")
        except Exception:
            logger.exception("安装 uvicorn.access 过滤器失败")

    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):
        """Tag CUDA 5xx responses and record failures globally.

        For any HTTPException with status >= 500, if its detail contains CUDA keywords,
        set header x-cuda-error=1 and record a CUDA failure.
        """
        response = await http_exception_handler(request, exc)
        try:
            if response.status_code >= 500:
                detail = exc.detail
                if isinstance(detail, str) and any(k in detail.lower() for k in CUDA_KEYWORDS):
                    response.headers["x-cuda-error"] = "1"
                    try:
                        monitor.record_cuda_failure(detail)
                    except Exception:
                        logger.exception("在异常处理器中记录 CUDA 失败出错")
        except Exception:
            logger.exception("标注 CUDA 错误响应头失败")
        return response

    app.state.cuda_health_monitor = monitor
    app.state.cuda_health_ready_predicate = ready_predicate
    return monitor


def check_health(app, is_ready: Optional[bool] = None, default_unhealthy_detail: str = "CUDA 错误导致不健康", show_trace: bool = False):
    """Compute health using the monitor attached to app.state.

    Parameters:
    - app: FastAPI
        The application instance.
    - is_ready: Optional[bool]
        If None, the function consults app.state.cuda_health_ready_predicate (if provided).
        If False, returns 503 (model not initialized). If True, proceed with CUDA window check.
    - default_unhealthy_detail: str
        Fallback detail message when monitor reports unhealthy.
    - show_trace: bool
        Whether to show the full traceback when raising HTTPException.

    Returns:
    - dict: {"status": "healthy"} if healthy; otherwise raises HTTPException(503).
    """
    if is_ready is None:
        try:
            rp = getattr(app.state, "cuda_health_ready_predicate", None)
            if callable(rp):
                is_ready = bool(rp())
            else:
                is_ready = True
        except Exception:
            is_ready = True

    if not is_ready:
        raise HTTPException(status_code=503, detail="模型未初始化")

    monitor = getattr(app.state, "cuda_health_monitor", None)
    if monitor is None:
        return {"status": "healthy"}

    healthy, detail = monitor.evaluate_and_maybe_notify()
    if not healthy:
        raise HTTPException(status_code=503, detail=detail or default_unhealthy_detail)
    if show_trace:
        return {"status": "healthy", "traceback": detail}
    return {"status": "healthy"}


def mount_health_route(app, path: str = "/health"):
    """Register a GET health route using check_health(app).

    Parameters:
    - app: FastAPI
    - path: str
        Path for health endpoint (default "/health").

    Returns: the APIRouter instance.
    """
    from fastapi import APIRouter

    router = APIRouter()

    @router.get(path)
    async def _health():
        return check_health(app)
    
    @router.trace(path)
    async def _health():
        return check_health(app, show_trace=True)

    app.include_router(router)
    logger.info(f"已挂载健康检查端点: {path}")
    return router


def setup_cuda_health(
    app,
    *,
    path: str = "/health",
    limit: Optional[int] = None,
    notifier: Optional[object] = None,
    suppress_access_paths: Optional[Iterable[str]] = ("/health", "/docs"),
    ready_predicate: Optional[Callable[[], bool]] = None,
):
    """One-step installation that initializes the plugin and mounts a health route.

    This is the recommended integration entrypoint for most apps.

    Parameters:
    - app: FastAPI
    - path: str
        Health endpoint path (default "/health").
    - limit: Optional[int]
        Consecutive CUDA failure threshold. If None, read from env (default 3).
    - notifier: Optional[object]
        Optional notifier with send_error_message(str) used on unhealthy flip.
    - suppress_access_paths: Optional[Iterable[str]]
        Paths to suppress in uvicorn.access logs (default /health, /docs).
    - ready_predicate: Optional[Callable[[], bool]]
        Optional readiness predicate.

    Returns:
    - CudaHealthMonitor: the monitor instance stored on app.state.
    """
    monitor = init_cuda_health_plugin(
        app,
        limit=limit,
        notifier=notifier,
        suppress_access_paths=suppress_access_paths,
        ready_predicate=ready_predicate,
    )
    mount_health_route(app, path=path)
    return monitor
