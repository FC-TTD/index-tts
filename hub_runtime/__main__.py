"""Model-owned API/UI assembled by the shared Runtime HTTP foundation."""

import os
import sys
import json
from importlib.metadata import version
from .adapter import load_model, completion, cleanup
from .api import build_api
from .startup import parse_args, validate_model_dir
from .http_compat import preserve_business_routes


def create_app(args=None):
    from ttd_model_runtime import Runtime
    from ttd_model_runtime.integrations.fastapi import attach

    args = args or parse_args()
    validate_model_dir(args.model_dir)
    runtime = Runtime(
        loader=lambda: load_model(args), completion=completion, cleanup=cleanup
    )
    app = build_api(args, runtime)

    def ui_factory():
        from .ui import build_demo

        return build_demo(args, runtime)

    enabled = os.getenv("ENABLE_GRADIO_UI", "1").lower() in ("1", "true", "yes")
    ui_path = os.getenv("GRADIO_MOUNT_PATH", "/__gradio__")
    preserve_business_routes(
        app,
        runtime,
        ui_path=ui_path,
        allowed_hosts={
            host.strip().lower()
            for host in os.getenv("GRADIO_ALLOWED_HOSTS", "").split(",")
            if host.strip()
        },
        ui_enabled=enabled,
    )
    return attach(
        app,
        runtime=runtime,
        ui_factory=ui_factory if enabled else None,
        ui_path=ui_path,
    )


def main():
    if sys.argv[1:] == ["describe"]:
        from types import SimpleNamespace

        app = build_api(SimpleNamespace(verbose=False), None)
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": "index",
                    "runtime_package": "ttd-model-runtime",
                    "runtime_version": version("ttd-model-runtime"),
                    "openapi": app.openapi(),
                },
                ensure_ascii=False,
            )
        )
        return
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
