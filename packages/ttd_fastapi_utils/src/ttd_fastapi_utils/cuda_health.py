from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections import deque
from typing import Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

CUDA_KEYWORDS = ("cuda", "cublas", "cudnn", "device-side", "illegal memory")


class CudaHealthMonitor:
    def __init__(self, limit: int = 3, notifier: Optional[object] = None, terminate_on_unhealthy: bool = True):
        self.limit = max(1, int(limit or 3))
        self._results: deque[bool] = deque(maxlen=self.limit)
        self._last_error_message: Optional[str] = None
        self._last_error_context: Optional[dict[str, str]] = None
        self._last_error_ts: Optional[float] = None
        self._unhealthy_notified: bool = False
        self.terminate_on_unhealthy: bool = bool(terminate_on_unhealthy)
        self._kill_scheduled: bool = False

        if not notifier:
            try:
                from .ttd_notify import notifier as default_notifier
            except Exception:
                default_notifier = None
            notifier = default_notifier
        self._notifier = notifier

    @staticmethod
    def is_cuda_error(exc: Exception) -> bool:
        msg_l = str(exc).lower()
        return any(k in msg_l for k in CUDA_KEYWORDS)

    def record_success(self) -> None:
        self._results.append(True)

    def _store_failure_context(self, context: Optional[Mapping[str, object]] = None) -> None:
        if not context:
            return
        normalized: dict[str, str] = {}
        for key, value in context.items():
            if value is None:
                continue
            text = str(value).strip()
            if text:
                normalized[str(key)] = text
        if normalized:
            self._last_error_context = normalized

    def record_error(self, exc: Exception, context: Optional[Mapping[str, object]] = None) -> None:
        if self.is_cuda_error(exc):
            self._results.append(False)
            self._last_error_message = str(exc)
            self._store_failure_context(context)
            self._last_error_ts = time.time()
        else:
            self._results.append(True)

    def record_cuda_failure(self, message: Optional[str] = None, context: Optional[Mapping[str, object]] = None) -> None:
        self._results.append(False)
        if message:
            self._last_error_message = message
        self._store_failure_context(context)
        self._last_error_ts = time.time()

    def _format_failure_detail(self) -> str:
        parts = [f"最近 {self.limit} 次推理均为 CUDA 错误。"]
        context = self._last_error_context or {}

        app_name = context.get("app_name")
        business = context.get("business")
        method = context.get("method")
        route = context.get("route")
        request_id = context.get("request_id")
        container = context.get("container")

        if app_name:
            parts.append(f"服务: {app_name}")
        if business:
            parts.append(f"业务: {business}")
        if method or route:
            endpoint = " ".join(part for part in (method, route) if part)
            parts.append(f"接口: {endpoint}")
        if request_id:
            parts.append(f"请求ID: {request_id}")
        if container:
            parts.append(f"实例: {container}")
        if self._last_error_message:
            parts.append(f"最后错误: {self._last_error_message}")
        return " ".join(parts)

    def is_unhealthy(self) -> bool:
        n = self.limit
        return len(self._results) >= n and all(r is False for r in list(self._results)[-n:])

    def evaluate_and_maybe_notify(self) -> Tuple[bool, Optional[str]]:
        unhealthy = self.is_unhealthy()
        detail = None

        if unhealthy:
            if not self._unhealthy_notified:
                detail = self._format_failure_detail()
                try:
                    if self._notifier is not None:
                        self._notifier.send_error_message(f"/health 不健康：{detail}")
                except Exception:
                    logger.exception("发送不健康通知失败")
                self._unhealthy_notified = True
            else:
                detail = self._format_failure_detail()

            if self.terminate_on_unhealthy and not self._kill_scheduled:
                self._kill_scheduled = True

                def _kill_worker():
                    try:
                        time.sleep(0.5)
                        logger.error("服务处于不健康状态（连续 CUDA 失败），将终止进程以触发容器重启…")
                        try:
                            os.kill(os.getpid(), signal.SIGTERM)
                        except Exception:
                            logger.exception("发送 SIGTERM 失败，使用 os._exit(1) 强制退出")
                            os._exit(1)
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
