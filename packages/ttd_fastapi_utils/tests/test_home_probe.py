import json
import os
import tempfile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ttd_fastapi_utils import setup_cuda_health


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
                json.dump({"build_commit": "commit_from_file", "build_time": "time_from_file"}, f)

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


if __name__ == "__main__":
    test_buildinfo_path_json_takes_effect()
    test_buildinfo_fallback_env_when_no_file()
    test_unhealthy_when_not_ready()
    print("ok")
