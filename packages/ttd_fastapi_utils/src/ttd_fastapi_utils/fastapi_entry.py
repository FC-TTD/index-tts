import logging
import os
from typing import Callable, Iterable, Optional, Tuple

from fastapi import APIRouter, HTTPException, FastAPI
from fastapi.exception_handlers import http_exception_handler
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .cuda_health import CUDA_KEYWORDS, CudaHealthMonitor
from .home_probe import build_home_payload

logger = logging.getLogger(__name__)


def _build_middleware(monitor: CudaHealthMonitor):
    async def dispatch(request: Request, call_next: Callable[[Request], Response]):
        try:
            response = await call_next(request)
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
                monitor.record_error(exc)
            except Exception:
                logger.exception("记录推理结果失败")
            raise

    return dispatch


def init_cuda_health_plugin(
    app: FastAPI,
    limit: Optional[int] = None,
    notifier: Optional[object] = None,
    suppress_access_paths: Optional[Iterable[str]] = ("/health", "/docs"),
    ready_predicate: Optional[Callable[[], bool]] = None,
    terminate_on_unhealthy: bool = True,
) -> CudaHealthMonitor:
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


def check_health(
    app,
    is_ready: Optional[bool] = None,
    default_unhealthy_detail: str = "CUDA 错误导致不健康",
    show_trace: bool = False,
):
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

    monitor: Optional[CudaHealthMonitor] = getattr(app.state, "cuda_health_monitor", None)
    if monitor is None:
        return {"status": "healthy"}

    healthy, detail = monitor.evaluate_and_maybe_notify()
    if not healthy:
        raise HTTPException(status_code=503, detail=detail or default_unhealthy_detail)

    if show_trace:
        return {"status": "healthy", "traceback": detail}
    return {"status": "healthy"}


def mount_health_route(app, path: str = "/health"):
    router = APIRouter()

    @router.get(path)
    async def _health():
        return check_health(app)

    @router.trace(path)
    async def _health_trace():
        return check_health(app, show_trace=True)

    app.include_router(router)
    logger.info(f"已挂载健康检查端点: {path}")
    return router


def _has_user_home_route(app) -> bool:
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


def _make_home_response(app, healthy: bool, unhealthy_detail: Optional[object]):
    app_name = getattr(app, "title", None) or os.getenv("APP_NAME", "FastAPI")
    payload = build_home_payload(app_name=app_name, healthy=healthy, unhealthy_detail=unhealthy_detail)

    if not healthy:
        resp = JSONResponse(payload, status_code=503)
        try:
            d = unhealthy_detail
            if isinstance(d, str) and any(k in d.lower() for k in CUDA_KEYWORDS):
                resp.headers["x-cuda-error"] = "1"
        except Exception:
            pass
        return resp

    return JSONResponse(payload, status_code=200)


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

        async def _default_home_dispatch(request: Request, call_next: Callable[[Request], Response]):
            try:
                if request.method not in ("GET", "HEAD"):
                    return await call_next(request)
                if request.url.path != "/":
                    return await call_next(request)
                if _has_user_home_route(app):
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

                return _make_home_response(app, healthy=healthy, unhealthy_detail=unhealthy_detail)
            except Exception:
                logger.exception("默认首页探活处理中间件异常")
                return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=_default_home_dispatch)

    return monitor
