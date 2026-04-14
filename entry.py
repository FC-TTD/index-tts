#!/usr/bin/env python3
import os

import gradio as gr
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


def create_entry_app(args=None):
    args = args or parse_args()
    app = create_app(args)
    enable_gradio_ui = _env_flag("ENABLE_GRADIO_UI", True)
    if enable_gradio_ui:
        demo = build_demo(args, tts_manager_getter=lambda: app.state.tts_manager)
        app = gr.mount_gradio_app(app, demo, path=_ui_mount_path())
    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(
        create_entry_app(args), host=args.host, port=args.port, log_level="info"
    )


if __name__ == "__main__":
    main()
