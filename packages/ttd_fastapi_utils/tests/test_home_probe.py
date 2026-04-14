import json
import os
import tempfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from ttd_fastapi_utils import setup_cuda_health
from ttd_fastapi_utils.cuda_health import CudaHealthMonitor
from ttd_fastapi_utils.fastapi_entry import _failure_context_from_request


def _make_app(*, ready: bool) -> FastAPI:
    app = FastAPI(title="ttd-fastapi-utils test")

    def _ready_predicate() -> bool:
        return bool(ready)

    setup_cuda_health(
        app,
        path="/health",
        ready_predicate=_ready_predicate,
        enable_default_home=True,
        terminate_on_unhealthy=False,
    )
    return app


def _call_home(app: FastAPI):
    client = TestClient(app)
    resp = client.get("/")
    return resp


def test_buildinfo_path_json_takes_effect():
    old_env = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "buildinfo.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "build_commit": "commit_from_file",
                        "build_time": "time_from_file",
                    },
                    f,
                )

            os.environ["BUILDINFO_PATH"] = p
            os.environ["BUILD_COMMIT"] = "commit_from_env"
            os.environ["BUILD_TIME"] = "time_from_env"

            app = _make_app(ready=True)
            resp = _call_home(app)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["build"]["build_commit"] == "commit_from_file"
            assert data["build"]["build_time"] == "time_from_file"
            assert data["build"].get("path") == p
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_buildinfo_fallback_env_when_no_file():
    old_env = dict(os.environ)
    try:
        os.environ.pop("BUILDINFO_PATH", None)
        os.environ["BUILD_COMMIT"] = "commit_from_env"
        os.environ["BUILD_TIME"] = "time_from_env"

        app = _make_app(ready=True)
        resp = _call_home(app)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["build"]["build_commit"] == "commit_from_env"
        assert data["build"]["build_time"] == "time_from_env"
        assert "path" not in data["build"]
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_unhealthy_when_not_ready():
    old_env = dict(os.environ)
    try:
        os.environ.pop("BUILDINFO_PATH", None)
        os.environ["BUILD_COMMIT"] = "commit_from_env"
        os.environ["BUILD_TIME"] = "time_from_env"

        app = _make_app(ready=False)
        resp = _call_home(app)
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data.get("detail") == "模型未初始化"
        assert data["build"]["build_commit"] == "commit_from_env"
        assert data["build"]["build_time"] == "time_from_env"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_health_detail_includes_service_and_route_context_on_cuda_failures():
    old_env = dict(os.environ)
    try:
        os.environ["HOSTNAME"] = "test-container"

        app = FastAPI(title="ctx-test")
        scope = {
            "type": "http",
            "app": app,
            "method": "POST",
            "path": "/generate",
            "headers": [(b"x-request-id", b"req-123")],
        }
        request = Request(scope)

        context = _failure_context_from_request(request)
        assert context["app_name"] == "ctx-test"
        assert context["method"] == "POST"
        assert context["route"] == "/generate"
        assert context["request_id"] == "req-123"
        assert context["container"] == "test-container"

        monitor = CudaHealthMonitor(limit=3, terminate_on_unhealthy=False)
        for _ in range(3):
            monitor.record_cuda_failure(
                "语音生成处理错误: CUDA out of memory",
                context=context,
            )

        healthy, detail = monitor.evaluate_and_maybe_notify()
        assert healthy is False
        assert detail is not None
        assert "服务: ctx-test" in detail
        assert "接口: POST /generate" in detail
        assert "请求ID: req-123" in detail
        assert "实例: test-container" in detail
        assert "最后错误: 语音生成处理错误: CUDA out of memory" in detail
    finally:
        os.environ.clear()
        os.environ.update(old_env)


if __name__ == "__main__":
    test_buildinfo_path_json_takes_effect()
    test_buildinfo_fallback_env_when_no_file()
    test_unhealthy_when_not_ready()
    test_health_detail_includes_service_and_route_context_on_cuda_failures()
    print("ok")
