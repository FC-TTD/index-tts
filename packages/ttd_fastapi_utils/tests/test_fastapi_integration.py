import os
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi import File, Form, UploadFile
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf
from fastapi.responses import Response

from ttd_fastapi_utils import SmartModel, postprocess, preset, setup_cuda_health
from ttd_fastapi_utils import speed_control
from ttd_fastapi_utils.ttd_notify import TTDNotify


def _make_test_wav_bytes(*, sr: int = 24000, seconds: float = 0.35) -> bytes:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    buf = BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def _read_wav_bytes(payload: bytes):
    wav, sr = sf.read(BytesIO(payload), dtype="float32")
    return wav, sr


def test_plugin_integrates_with_real_fastapi_app_and_tracks_cuda_failures():
    old_env = dict(os.environ)
    try:
        os.environ["HOSTNAME"] = "integration-container"

        app = FastAPI(title="IndexTTS API")
        ready = True

        setup_cuda_health(
            app,
            path="/health",
            ready_predicate=lambda: ready,
            enable_default_home=True,
            terminate_on_unhealthy=False,
            limit=3,
        )

        @app.post("/generate")
        async def generate_audio():
            raise HTTPException(
                status_code=500,
                detail="语音生成处理错误: CUDA out of memory",
            )

        @app.get("/generate-ok")
        async def generate_ok():
            return {"ok": True}

        client = TestClient(app)

        home_resp = client.get("/")
        assert home_resp.status_code == 200
        assert home_resp.json()["status"] == "healthy"

        success_resp = client.get("/generate-ok")
        assert success_resp.status_code == 200
        assert success_resp.json() == {"ok": True}

        for _ in range(3):
            resp = client.post("/generate", headers={"x-request-id": "req-int-001"})
            assert resp.status_code == 500
            assert resp.headers.get("x-cuda-error") == "1"

        health_resp = client.get("/health")
        assert health_resp.status_code == 503
        detail = health_resp.json()["detail"]
        assert "服务: IndexTTS API" in detail
        assert "接口: POST /generate" in detail
        assert "请求ID: req-int-001" in detail
        assert "实例: integration-container" in detail
        assert "CUDA out of memory" in detail

        home_after_fail = client.get("/")
        assert home_after_fail.status_code == 503
        assert home_after_fail.json()["status"] == "unhealthy"
        assert "POST /generate" in str(home_after_fail.json().get("detail"))
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_plugin_integrates_with_real_audio_postprocess_route():
    app = FastAPI(title="IndexTTS audio integration")

    @app.post("/postprocess")
    async def postprocess_audio(
        audio: UploadFile = File(...),
        enable: bool = Form(True),
        trim_silence: bool = Form(False),
        enable_eq: bool = Form(True),
        lufs: float = Form(-23.0),
    ):
        wav, sr = sf.read(BytesIO(await audio.read()), dtype="float32")
        processed = postprocess.apply_postprocess(
            wav,
            sr,
            target_loudness=lufs,
            enable=enable,
            trim_silence=trim_silence,
            enable_eq=enable_eq,
        )
        buf = BytesIO()
        sf.write(buf, processed, sr, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    client = TestClient(app)
    source_bytes = _make_test_wav_bytes()
    source_wav, source_sr = _read_wav_bytes(source_bytes)

    resp = client.post(
        "/postprocess",
        files={"audio": ("demo.wav", source_bytes, "audio/wav")},
        data={
            "enable": "true",
            "trim_silence": "false",
            "enable_eq": "true",
            "lufs": "-21.0",
        },
    )
    assert resp.status_code == 200

    processed_wav, processed_sr = _read_wav_bytes(resp.content)
    assert processed_sr == source_sr
    assert processed_wav.ndim == 1
    assert processed_wav.shape == source_wav.shape
    assert np.max(np.abs(processed_wav)) <= 1.0
    assert not np.allclose(processed_wav, source_wav)


def test_plugin_integrates_with_real_audio_preset_route():
    app = FastAPI(title="IndexTTS preset integration")

    @app.post("/preset")
    async def preset_audio(
        audio: UploadFile = File(...),
        preset_name: str = Form(...),
    ):
        wav, sr = sf.read(BytesIO(await audio.read()), dtype="float32")
        processed = preset.apply_preset(preset_name, wav, sr)
        buf = BytesIO()
        sf.write(buf, processed, sr, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    client = TestClient(app)
    source_bytes = _make_test_wav_bytes(seconds=0.5)
    source_wav, source_sr = _read_wav_bytes(source_bytes)

    resp = client.post(
        "/preset",
        files={"audio": ("demo.wav", source_bytes, "audio/wav")},
        data={"preset_name": "telephone"},
    )
    assert resp.status_code == 200

    processed_wav, processed_sr = _read_wav_bytes(resp.content)
    assert processed_sr == source_sr
    assert processed_wav.ndim == 1
    assert processed_wav.shape == source_wav.shape
    assert np.max(np.abs(processed_wav)) <= 1.0
    assert not np.allclose(processed_wav, source_wav)


def test_plugin_integrates_with_speed_control_route_via_bypass_mode():
    old_env = dict(os.environ)
    try:
        os.environ["TTD_SPEED_CONTROL_BYPASS_SOX"] = "1"

        app = FastAPI(title="IndexTTS speed integration")

        @app.post("/speed")
        async def speed_audio(
            audio: UploadFile = File(...),
            speed: float = Form(...),
        ):
            wav, sr = sf.read(BytesIO(await audio.read()), dtype="float32")
            processed = speed_control.time_stretch_wav(wav, sr, speed)
            buf = BytesIO()
            sf.write(buf, processed, sr, format="WAV")
            return Response(content=buf.getvalue(), media_type="audio/wav")

        client = TestClient(app)
        source_bytes = _make_test_wav_bytes(seconds=0.4)
        source_wav, source_sr = _read_wav_bytes(source_bytes)

        resp = client.post(
            "/speed",
            files={"audio": ("demo.wav", source_bytes, "audio/wav")},
            data={"speed": "1.25"},
        )
        assert resp.status_code == 200

        processed_wav, processed_sr = _read_wav_bytes(resp.content)
        assert processed_sr == source_sr
        assert processed_wav.shape == source_wav.shape
        assert np.allclose(processed_wav, source_wav, atol=5e-5)
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_plugin_integrates_with_smart_model_lifecycle_in_fastapi_app():
    class MockModel:
        def __init__(self, token: int):
            self.token = token
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True

    state = {"loads": 0, "stopped": False, "last_model": None}

    def load_model():
        state["loads"] += 1
        model = MockModel(token=state["loads"])
        state["last_model"] = model
        return model

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = SmartModel(load_model, timeout_seconds=-1)
        app.state.manager = manager
        yield
        manager.stop()
        manager.unload()
        state["stopped"] = True

    app = FastAPI(title="IndexTTS SmartModel integration", lifespan=lifespan)

    @app.get("/infer")
    async def infer():
        model = app.state.manager.get()
        return {"token": model.token}

    with TestClient(app) as client:
        resp1 = client.get("/infer")
        resp2 = client.get("/infer")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json() == {"token": 1}
        assert resp2.json() == {"token": 1}
        assert state["loads"] == 1

    assert state["stopped"] is True
    assert state["last_model"] is not None
    assert state["last_model"].cpu_called is True


def test_plugin_integrates_with_notify_route(monkeypatch):
    calls = []

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"errcode": 0}

    def fake_post(url, headers=None, data=None):
        calls.append({"url": url, "headers": headers, "data": data})
        return DummyResponse()

    import ttd_fastapi_utils.ttd_notify as notify_module

    monkeypatch.setattr(
        notify_module,
        "requests",
        type("RequestsStub", (), {"post": staticmethod(fake_post)}),
    )

    app = FastAPI(title="IndexTTS notify integration")
    notifier = TTDNotify("https://notify.example.test/webhook")

    @app.post("/notify")
    async def notify_route():
        notifier.send_info_message("TEST ONLY: integration notification")
        return {"ok": True}

    client = TestClient(app)
    resp = client.post("/notify")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["url"] == "https://notify.example.test/webhook"
    assert calls[0]["headers"] == {"Content-Type": "application/json"}
    assert "TEST ONLY: integration notification" in calls[0]["data"]
