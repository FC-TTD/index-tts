import os
import time
import logging
import weakref
from collections import deque
from typing import Optional, Callable, Tuple, Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException

logger = logging.getLogger(__name__)


CUDA_KEYWORDS = ("cuda", "cublas", "cudnn", "device-side", "illegal memory")
_ATTR_TRACK_FLAG = "__cuda_health_tracked__"


def track_cuda_health(func: Callable):
    """装饰器：标记该端点为需要跟踪 CUDA 错误的端点。"""
    try:
        setattr(func, _ATTR_TRACK_FLAG, True)
    except Exception:
        pass
    return func


class CudaHealthMonitor:
    def __init__(self, limit: int = 3, notifier: Optional[object] = None):
        self.limit = max(1, int(limit or 3))
        self._results: deque[bool] = deque(maxlen=self.limit)
        self._last_error_message: Optional[str] = None
        self._last_error_ts: Optional[float] = None
        self._unhealthy_notified: bool = False
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
            # 非 CUDA 错误不影响健康，但用于“填充”窗口，避免粘性不健康
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
                # 发送一次通知
                try:
                    if self._notifier is not None:
                        self._notifier.send_error_message(f"/health 不健康：{detail}")
                except Exception:
                    logger.exception("发送不健康通知失败")
                self._unhealthy_notified = True
        else:
            # 恢复健康则允许下次再次通知
            self._unhealthy_notified = False
        return (not unhealthy), detail


def _build_middleware(
    monitor: CudaHealthMonitor,
    track_predicate: Optional[Callable[[Request], bool]] = None,
    track_path_prefixes: Optional[Iterable[str]] = None,
):
    prefixes = tuple(track_path_prefixes or ())

    def _path_match(req: Request) -> bool:
        if not prefixes:
            return False
        p = req.url.path
        return any(p.startswith(pre) for pre in prefixes)

    def _endpoint_marked(req: Request) -> bool:
        try:
            endpoint = req.scope.get("endpoint")
            return bool(getattr(endpoint, _ATTR_TRACK_FLAG, False))
        except Exception:
            return False

    def _should_track(req: Request) -> bool:
        try:
            if track_predicate and track_predicate(req):
                return True
        except Exception:
            logger.exception("track_predicate 执行失败")
        return _endpoint_marked(req) or _path_match(req)

    async def dispatch(request: Request, call_next: Callable[[Request], Response]):
        try:
            response = await call_next(request)
            if _should_track(request):
                # 仅当业务未抛异常且状态码 < 500 视为成功
                if response.status_code < 500:
                    monitor.record_success()
                else:
                    # 若端点通过响应头显式标注 CUDA 错误，则记录为失败
                    if response.headers.get("x-cuda-error") == "1":
                        monitor.record_cuda_failure()
            return response
        except Exception as exc:  # noqa: BLE001 - 需捕获以分类 CUDA 错误
            try:
                if _should_track(request):
                    monitor.record_error(exc)
            except Exception:
                logger.exception("记录推理结果失败")
            raise
    return dispatch


def init_cuda_health_plugin(
    app,
    limit: Optional[int] = None,
    notifier: Optional[object] = None,
    track_predicate: Optional[Callable[[Request], bool]] = None,
    suppress_access_paths: Optional[Iterable[str]] = ("/health", "/docs"),
    track_path_prefixes: Optional[Iterable[str]] = None,
    ready_predicate: Optional[Callable[[], bool]] = None,
) -> CudaHealthMonitor:
    """
    初始化并挂载 CUDA 健康检查插件。
    - limit: 连续 CUDA 错误阈值（默认读取环境变量 CONSECUTIVE_FAIL_LIMIT，缺省为 3）
    - notifier: 可选的通知对象（须具备 send_error_message(str) 方法）
    返回：monitor 实例，可用于 /health 端点查询。
    """
    if limit is None:
        try:
            limit = int(os.getenv("CONSECUTIVE_FAIL_LIMIT", "3"))
        except Exception:
            limit = 3
    monitor = CudaHealthMonitor(limit=limit, notifier=notifier)

    # 注册中间件（以函数形式的 BaseHTTPMiddleware）
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=_build_middleware(
            monitor,
            track_predicate=track_predicate,
            track_path_prefixes=track_path_prefixes,
        ),
    )

    # 可选：抑制 uvicorn.access 对特定路径的访问日志
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

    # 暴露到 app.state 方便其他地方获取（可选）
    app.state.cuda_health_monitor = monitor
    app.state.cuda_health_ready_predicate = ready_predicate
    return monitor


def check_health(app, is_ready: Optional[bool] = None, default_unhealthy_detail: str = "CUDA 错误导致不健康"):
    """
    统一的 /health 判定辅助：
    - is_ready=False -> 503(模型未初始化)
    - 若无 monitor -> healthy
    - 若 monitor 判定不健康 -> 503(detail)
    - 其余 -> {"status":"healthy"}
    """
    if is_ready is None:
        # 若未显式传入，尝试使用插件注册的 ready_predicate
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
    return {"status": "healthy"}


def mount_health_route(app, path: str = "/health"):
    """为应用注册一个健康检查路由，使用插件的统一判定逻辑。"""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get(path)
    async def _health():
        return check_health(app)

    app.include_router(router)
    return router


def setup_cuda_health(
    app,
    *,
    path: str = "/health",
    limit: Optional[int] = None,
    notifier: Optional[object] = None,
    track_predicate: Optional[Callable[[Request], bool]] = None,
    suppress_access_paths: Optional[Iterable[str]] = ("/health", "/docs"),
    track_path_prefixes: Optional[Iterable[str]] = None,
    ready_predicate: Optional[Callable[[], bool]] = None,
):
    """
    一步式安装 CUDA 健康检查：初始化插件并挂载健康路由。
    返回 monitor 实例。
    """
    monitor = init_cuda_health_plugin(
        app,
        limit=limit,
        notifier=notifier,
        track_predicate=track_predicate,
        suppress_access_paths=suppress_access_paths,
        track_path_prefixes=track_path_prefixes,
        ready_predicate=ready_predicate,
    )
    mount_health_route(app, path=path)
    return monitor
