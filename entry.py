#!/usr/bin/env python3
import os

import gradio as gr
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from api import create_app, parse_args
from webui import build_demo


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ui_mount_path() -> str:
    value = (os.getenv("GRADIO_MOUNT_PATH", "/") or "/").strip()
    if not value.startswith("/"):
        value = f"/{value}"
    return value or "/"


def _ui_allowed_hosts() -> set[str]:
    value = os.getenv("GRADIO_ALLOWED_HOSTS", "")
    return {item.strip().lower() for item in value.split(",") if item.strip()}


class GradioHostGateMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, allowed_hosts: set[str], ui_mount_path: str):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts
        self.ui_mount_path = ui_mount_path.rstrip("/") or "/"
        self.gradio_prefixes = (
            "/assets/",
            "/favicon",
            "/manifest.json",
            "/gradio_api/",
            "/theme.css",
            "/robots.txt",
        )

    async def dispatch(self, request, call_next):
        if not self.allowed_hosts:
            return await call_next(request)

        host = request.headers.get("host", "").split(":", 1)[0].lower()
        path = request.url.path or "/"

        if host in self.allowed_hosts:
            if path == "/":
                rewritten = f"{self.ui_mount_path}/"
                request.scope["path"] = rewritten
                request.scope["raw_path"] = rewritten.encode()
            elif any(path == prefix[:-1] or path.startswith(prefix) for prefix in self.gradio_prefixes):
                rewritten = f"{self.ui_mount_path}{path}"
                request.scope["path"] = rewritten
                request.scope["raw_path"] = rewritten.encode()

        return await call_next(request)


def create_entry_app(args=None):
    args = args or parse_args()
    app = create_app(args)
    enable_gradio_ui = _env_flag("ENABLE_GRADIO_UI", True)
    if enable_gradio_ui:
        ui_mount_path = _ui_mount_path()
        demo = build_demo(args, tts_manager_getter=lambda: app.state.tts_manager)
        app = gr.mount_gradio_app(app, demo, path=ui_mount_path)
        app.add_middleware(
            GradioHostGateMiddleware,
            allowed_hosts=_ui_allowed_hosts(),
            ui_mount_path=ui_mount_path,
        )
    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(
        create_entry_app(args), host=args.host, port=args.port, log_level="info"
    )


if __name__ == "__main__":
    main()
