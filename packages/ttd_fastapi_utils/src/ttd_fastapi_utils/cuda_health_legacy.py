"""
TTD FastAPI Utils - CUDA Health Monitor
---------------------------------------

Migrated from fastapi-cuda-health (Plan B: global CUDA detection).
Exposes a one-call setup function `setup_cuda_health(app, ...)`.

Also embeds a simple Notify utility and a default `notifier` for notifications.
"""
from collections import deque
import json
import logging
import os
import time
import signal
import threading
import re
from typing import Iterable, Optional, Tuple, Callable

from fastapi import HTTPException
from fastapi.exception_handlers import http_exception_handler
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

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

    def __init__(self, limit: int = 3, notifier: Optional[object] = None, terminate_on_unhealthy: bool = True):
        self.limit = max(1, int(limit or 3))
        self._results: deque[bool] = deque(maxlen=self.limit)
        self._last_error_message: Optional[str] = None
        self._last_error_ts: Optional[float] = None
        self._unhealthy_notified: bool = False
        self.terminate_on_unhealthy: bool = bool(terminate_on_unhealthy)
        self._kill_scheduled: bool = False
        if not notifier:
            # Lazy import to avoid requests dependency if unused
            try:
                from .ttd_notify import notifier as default_notifier
            except Exception:
                default_notifier = None
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
            # Schedule a one-shot self-termination if enabled
            if self.terminate_on_unhealthy and not self._kill_scheduled:
                self._kill_scheduled = True
                def _kill_worker():
                    try:
                        # short delay to allow health response/logs to flush
                        time.sleep(0.5)
                        logger.error("服务处于不健康状态（连续 CUDA 失败），将终止进程以触发容器重启…")
                        try:
                            os.kill(os.getpid(), signal.SIGTERM)
                        except Exception:
                            logger.exception("发送 SIGTERM 失败，使用 os._exit(1) 强制退出")
                            os._exit(1)
                        # If process still alive after grace period, force exit
                        time.sleep(2.0)
                        os._exit(1)
                    except Exception:
                        logger.exception("终止任务执行异常，强制退出")
                        os._exit(1)
                t = threading.Thread(target=_kill_worker, daemon=True)
                t.start()
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
    terminate_on_unhealthy: bool = True,
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
    monitor = CudaHealthMonitor(limit=limit, notifier=notifier, terminate_on_unhealthy=terminate_on_unhealthy)

    app.add_middleware(BaseHTTPMiddleware, dispatch=_build_middleware(monitor))

    if suppress_access_paths:
        try:
            paths = tuple(suppress_access_paths)

            def _scope_path_matches(path: str, p: str) -> bool:
                if p == "/":
                    return path == "/"
                return path == p or path.startswith(p + "/")

            def _msg_matches_path(msg: str, p: str) -> bool:
                for method in ("GET", "HEAD", "TRACE"):
                    token = f'"{method} {p}'
                    idx = msg.find(token)
                    if idx < 0:
                        continue
                    j = idx + len(token)
                    if j >= len(msg):
                        return True
                    c = msg[j]
                    if p == "/":
                        if c in (" ", "?"):
                            return True
                    else:
                        if c in (" ", "?", "/"):
                            return True
                return False

            class _AccessPathSuppressFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    try:
                        scope = getattr(record, "scope", None)
                        if scope:
                            headers = scope.get("headers") or []
                            for name, value in headers:
                                try:
                                    if name.lower() == b"user-agent" and "uptime-kuma" in value.decode("latin1").lower():
                                        return False
                                except Exception:
                                    break
                            path = scope.get("path") or ""
                            if any(_scope_path_matches(path, p) for p in paths):
                                return False
                    except Exception:
                        pass
                    try:
                        msg = record.getMessage()
                    except Exception:
                        return True
                    msg_l = msg.lower()
                    if "uptime-kuma" in msg_l:
                        return False
                    if any(_msg_matches_path(msg, p) for p in paths):
                        return False
                    return True

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

    monitor: CudaHealthMonitor | None = getattr(app.state, "cuda_health_monitor", None)
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


def _read_buildinfo() -> dict:
    buildinfo_path = os.getenv("BUILDINFO_PATH", "").strip()
    candidates = []
    if buildinfo_path:
        candidates.append(buildinfo_path)
    candidates.extend([
        os.path.join(os.getcwd(), "buildinfo.json"),
        "/app/buildinfo.json",
    ])
    for p in candidates:
        try:
            if not p:
                continue
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                v = json.load(f)
            if isinstance(v, dict):
                v["path"] = p
                return v
        except Exception:
            logger.exception(f"读取 buildinfo 失败: {p}")
    env_build = {
        "build_commit": os.getenv("BUILD_COMMIT", ""),
        "build_time": os.getenv("BUILD_TIME", ""),
    }
    return {k: v for k, v in env_build.items() if v}


def _read_container_info() -> dict:
    info: dict = {
        "hostname": os.getenv("HOSTNAME", ""),
        "pid": os.getpid(),
        "in_container": bool(os.path.exists("/.dockerenv")),
    }
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            cgroup = f.read()
        m = re.search(r"([0-9a-f]{12,64})", cgroup)
        if m:
            info["container_id"] = m.group(1)
    except Exception:
        pass
    v = os.getenv("NVIDIA_VISIBLE_DEVICES", "")
    if v:
        info["nvidia_visible_devices"] = v
    return info


def mount_default_home_route(app, path: str = "/"):
    from fastapi import APIRouter

    try:
        for r in getattr(app, "router", None).routes or []:
            if getattr(r, "path", None) == path:
                methods = getattr(r, "methods", None)
                if methods and ("GET" in methods or "HEAD" in methods):
                    logger.info(f"默认首页端点已存在，跳过挂载: {path}")
                    return None
    except Exception:
        logger.exception("检查默认首页端点是否已存在失败，将继续尝试挂载")

    router = APIRouter()

    @router.get(path, include_in_schema=False)
    async def _default_home():
        healthy = True
        unhealthy_detail = None
        try:
            check_health(app)
        except HTTPException as e:
            if int(getattr(e, "status_code", 0)) == 503:
                healthy = False
                unhealthy_detail = getattr(e, "detail", None)
            else:
                raise

        app_name = getattr(app, "title", None) or os.getenv("APP_NAME", "FastAPI")
        buildinfo = _read_buildinfo()
        container = _read_container_info()

        message = f"欢迎使用 {app_name}"
        if buildinfo.get("build_commit") or buildinfo.get("build_time"):
            bc = buildinfo.get("build_commit", "")
            bt = buildinfo.get("build_time", "")
            if bc and bt:
                message = f"欢迎使用 {app_name} (build {bc} at {bt})"
            elif bc:
                message = f"欢迎使用 {app_name} (build {bc})"
            elif bt:
                message = f"欢迎使用 {app_name} (build at {bt})"

        payload = {
            "message": message,
            "app": {"name": app_name},
            "status": "healthy" if healthy else "unhealthy",
            "build": buildinfo,
            "container": container,
        }
        if not healthy:
            payload["detail"] = unhealthy_detail
            resp = JSONResponse(payload, status_code=503)
            try:
                d = unhealthy_detail
                if isinstance(d, str) and any(k in d.lower() for k in CUDA_KEYWORDS):
                    resp.headers["x-cuda-error"] = "1"
            except Exception:
                pass
            return resp
        return JSONResponse(payload, status_code=200)

    app.include_router(router)
    logger.info(f"已挂载默认首页探活端点: {path}")
    return router


def setup_cuda_health(
    app,
    *,
    path: str = "/health",
    limit: Optional[int] = None,
    notifier: Optional[object] = None,
    suppress_access_paths: Optional[Iterable[str]] = None,
    ready_predicate: Optional[Callable[[], bool]] = None,
    terminate_on_unhealthy: bool = True,
    enable_default_home: bool = True,
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
    if suppress_access_paths is None:
        _suppress_paths = ("/health", "/docs", "/") if enable_default_home else ("/health", "/docs")
    else:
        _suppress_paths = tuple(suppress_access_paths)

    monitor = init_cuda_health_plugin(
        app,
        limit=limit,
        notifier=notifier,
        suppress_access_paths=_suppress_paths,
        ready_predicate=ready_predicate,
        terminate_on_unhealthy=terminate_on_unhealthy,
    )
    mount_health_route(app, path=path)
    if enable_default_home and not getattr(app.state, "cuda_health_default_home_installed", False):
        app.state.cuda_health_default_home_installed = True

        def _has_user_home_route() -> bool:
            try:
                for r in getattr(app, "router", None).routes or []:
                    if getattr(r, "path", None) != "/":
                        continue
                    methods = getattr(r, "methods", None)
                    if methods and ("GET" in methods or "HEAD" in methods):
                        return True
            except Exception:
                return False
            return False

        async def _default_home_dispatch(request: Request, call_next: Callable[[Request], Response]):
            try:
                if request.method not in ("GET", "HEAD"):
                    return await call_next(request)
                if request.url.path != "/":
                    return await call_next(request)
                if _has_user_home_route():
                    return await call_next(request)

                healthy = True
                unhealthy_detail = None
                try:
                    check_health(app)
                except HTTPException as e:
                    if int(getattr(e, "status_code", 0)) == 503:
                        healthy = False
                        unhealthy_detail = getattr(e, "detail", None)
                    else:
                        raise

                app_name = getattr(app, "title", None) or os.getenv("APP_NAME", "FastAPI")
                buildinfo = _read_buildinfo()
                container = _read_container_info()

                message = f"欢迎使用 {app_name}"
                if buildinfo.get("build_commit") or buildinfo.get("build_time"):
                    bc = buildinfo.get("build_commit", "")
                    bt = buildinfo.get("build_time", "")
                    if bc and bt:
                        message = f"欢迎使用 {app_name} (build {bc} at {bt})"
                    elif bc:
                        message = f"欢迎使用 {app_name} (build {bc})"
                    elif bt:
                        message = f"欢迎使用 {app_name} (build at {bt})"

                payload = {
                    "message": message,
                    "app": {"name": app_name},
                    "status": "healthy" if healthy else "unhealthy",
                    "build": buildinfo,
                    "container": container,
                }
                if not healthy:
                    payload["detail"] = unhealthy_detail
                    resp = JSONResponse(payload, status_code=503)
                    try:
                        d = unhealthy_detail
                        if isinstance(d, str) and any(k in d.lower() for k in CUDA_KEYWORDS):
                            resp.headers["x-cuda-error"] = "1"
                    except Exception:
                        pass
                    return resp
                return JSONResponse(payload, status_code=200)
            except Exception:
                logger.exception("默认首页探活处理中间件异常")
                return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=_default_home_dispatch)
    return monitor


# Compatibility re-export layer
from .cuda_health import CUDA_KEYWORDS, CudaHealthMonitor
from .fastapi_entry import check_health, init_cuda_health_plugin, mount_health_route, setup_cuda_health
from .ttd_notify import notifier

