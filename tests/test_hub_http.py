"""Real FastAPI/SDK transport tests using the actual generate callback, without weights."""

import ast
import gc
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
import wave

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import Response
    from fastapi.testclient import TestClient
    from ttd_hub_runtime import HubError, ManagedModel
except ModuleNotFoundError:
    HTTP_DEPS = False
else:
    HTTP_DEPS = True

import hub_adapter
from download_filename import build_content_disposition, build_download_filename

LEASE = {"id": "index-lease", "generation": 1, "revision": 1,
         "service": "index", "fingerprint": "a" * 64}


class Node:
    service = "index"

    def __init__(self):
        self.begins = []
        self.finishes = []
        self.failure = None

    def begin(self, key, grant=None):
        if self.failure:
            raise HubError(self.failure)
        activity = {"id": uuid4().hex, "key": key, "service": "index",
                    "lease_id": LEASE["id"], "generation": 1}
        self.begins.append((activity, grant))
        return activity

    def finish(self, activity, state):
        self.finishes.append((activity["id"], state))


class AudioFixture:
    @staticmethod
    def write(destination, samples, sample_rate, **kwargs):
        with wave.open(destination, "wb") as audio:
            audio.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x01\x00" * len(samples))

    @staticmethod
    def read(source, **kwargs):
        with wave.open(source, "rb") as audio:
            return [1] * audio.getnframes(), audio.getframerate()


@unittest.skipUnless(HTTP_DEPS, "install FastAPI/httpx/python-multipart and expose the real Hub SDK")
class HubHTTPTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.calls = []
        calls = self.calls

        class Model:
            def infer(self, output_path, **kwargs):
                calls.append(kwargs)
                AudioFixture.write(output_path, [1] * 2205, 22050)
                return output_path

        self.node = Node()
        self.manager = ManagedModel(Model, cleanup=gc.collect, synchronize=lambda: None,
                                    node=self.node, version="v1", token="test-runtime-token-00000000000",
                                    state_dir=directory.name)
        self.addCleanup(self.manager.stop)
        self.manager.load(LEASE)
        self.app = FastAPI()
        self.app.state.tts_manager = self.manager
        hub_adapter.add_managed_middleware(self.app)
        source = ast.parse((Path(__file__).resolve().parents[1] / "api.py").read_text())
        callback = next(node for node in ast.walk(source)
                        if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_audio")
        ns = dict(vars(hub_adapter), app=self.app, File=File, Form=Form, HTTPException=HTTPException,
                  UploadFile=UploadFile, Response=Response, os=os, json=json, tempfile=tempfile,
                  logger=logging.getLogger("hub-http-fixture"), sf=AudioFixture, BytesIO=BytesIO,
                  args=SimpleNamespace(verbose=False), normalize_language=lambda value: value.upper(),
                  speed_to_duration_factor=lambda speed: 1.0 / speed,
                  build_download_filename=build_download_filename,
                  build_content_disposition=build_content_disposition)
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "actual_generate_callback", "exec"), ns)

    def post(self, client, **kwargs):
        return client.post("/generate", data={"text": "测试", "speed": "1.25",
                                            "remove_silence": "false", "postprocess": "false"},
                           files={"prompt_speech": ("voice.wav", b"reference-audio", "audio/wav")}, **kwargs)

    def test_real_multipart_route_preserves_schema_audio_and_gateway_grant(self):
        with TestClient(self.app) as client:
            schema = client.get("/openapi.json").json()
            operation = schema["paths"]["/generate"]["post"]
            self.assertIn("multipart/form-data", operation["requestBody"]["content"])
            self.assertNotIn("parameters", operation)
            response = self.post(client, headers={"X-Hub-Lease-Id": LEASE["id"],
                "X-Hub-Generation": "1", "X-Hub-Fingerprint": LEASE["fingerprint"],
                "X-Hub-Service": "index", "X-Request-Id": "gateway-request"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        with wave.open(BytesIO(response.content), "rb") as audio:
            self.assertEqual((audio.getnframes(), audio.getframerate()), (2205, 22050))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["duration_factor"], 0.8)
        activity, grant = self.node.begins[0]
        self.assertEqual(activity["key"], "gateway-request")
        self.assertEqual(grant, {key: LEASE[key] for key in ("id", "generation", "fingerprint", "service")})
        self.assertEqual(self.node.finishes, [(activity["id"], "succeeded")])

    def test_real_api_busy_and_partial_grant_never_execute_model(self):
        self.node.failure = "resource_busy"
        with TestClient(self.app) as client:
            response = self.post(client)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "resource_busy"})
            response = self.post(client, headers={"X-Hub-Lease-Id": LEASE["id"]})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])

    def test_real_api_gpu_completion_failure_returns_unknown_not_audio(self):
        def fail_sync():
            raise RuntimeError("GPU completion unavailable")
        self.manager.synchronize = fail_sync
        with TestClient(self.app) as client:
            response = self.post(client)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "execution_unknown"})
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.node.finishes[0][1], "unknown")


if __name__ == "__main__":
    unittest.main()
