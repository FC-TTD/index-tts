"""Keep the existing Index API home and host-based native UI routes."""

from fastapi.responses import JSONResponse
from ttd_model_runtime.home_probe import build_home_payload


class GradioHostRoutes:
    # Same public aliases as entry.py. The business path is rewritten internally;
    # browsers and API callers do not move to a different URL or hostname.
    prefixes = (
        "/assets/",
        "/favicon",
        "/manifest.json",
        "/gradio_api/",
        "/theme.css",
        "/robots.txt",
    )

    def __init__(self, app, *, allowed_hosts, ui_path):
        self.app = app
        self.allowed_hosts = allowed_hosts
        self.ui_path = ui_path.rstrip("/")

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            host = headers.get(b"host", b"").decode("latin-1").split(":", 1)[0].lower()
            path = scope.get("path", "/")
            if host in self.allowed_hosts and (
                path == "/"
                or any(path == p[:-1] or path.startswith(p) for p in self.prefixes)
            ):
                scope = dict(scope)
                scope["path"] = self.ui_path + path
                scope["raw_path"] = self.ui_path.encode() + scope.get(
                    "raw_path", path.encode()
                )
        await self.app(scope, receive, send)


def preserve_business_routes(app, runtime, *, ui_path, allowed_hosts, ui_enabled):
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def home():
        status = runtime.status()
        healthy = (
            status["accepting"]
            and status["healthy"]
            and status["residency"] != "unknown"
        )
        return JSONResponse(
            status_code=200 if healthy else 503,
            content=build_home_payload(
                app_name=app.title,
                healthy=healthy,
                unhealthy_detail=runtime.health.snapshot(),
            ),
        )

    if ui_enabled:
        app.add_middleware(
            GradioHostRoutes, allowed_hosts=allowed_hosts, ui_path=ui_path
        )
