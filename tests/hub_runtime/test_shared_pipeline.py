import ast
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from uuid import uuid4
import wave

import importlib.util
import pytest

if importlib.util.find_spec("ttd_model_runtime") is None:
    pytest.skip(
        "Install the fixed ttd-model-runtime SDK to run Hub integration tests",
        allow_module_level=True,
    )

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from ttd_model_runtime import Runtime
from ttd_model_runtime.integrations.fastapi import attach
from hub_runtime.api import build_api
from hub_runtime.adapter import generate_ui
from hub_runtime.parameters import DEFAULTS

LEASE = {
    "id": "lease-one",
    "service": "index",
    "generation": 1,
    "revision": 1,
    "fingerprint": "a" * 64,
}


class Node:
    service = "index"

    def __init__(self):
        self.ends = []

    def begin(self, key, grant=None):
        return {
            "id": uuid4().hex,
            "key": key,
            "service": "index",
            "lease_id": "lease-one",
            "generation": 1,
        }

    def finish(self, a, state):
        self.ends.append(state)


class Model:
    def __init__(self):
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        sr = 22050
        data = np.sin(np.arange(sr * 2) * 2 * np.pi * 220 / sr) * 0.15
        sf.write(kwargs["output_path"], data, sr)
        return kwargs["output_path"]


class SharedPipeline(unittest.TestCase):
    def test_api_and_ui_share_effective_parameters_and_processed_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            node = Node()
            runtime = Runtime(
                lambda: model,
                node=node,
                version="test",
                token="t" * 32,
                state_dir=directory,
            )
            runtime.start()
            runtime.load(LEASE)
            args = SimpleNamespace(verbose=False)
            app = attach(build_api(args, runtime), runtime=runtime)
            reference = Path(directory) / "reference.wav"
            sf.write(
                reference,
                np.sin(np.arange(22050) * 2 * np.pi * 180 / 22050) * 0.1,
                22050,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/generate",
                    data={
                        "text": "你好",
                        "top_k": "0",
                        "emo_vector": "[0.5,0.5,0,0,0,0,0,0]",
                    },
                    files={
                        "prompt_speech": (
                            "reference.wav",
                            reference.read_bytes(),
                            "audio/wav",
                        )
                    },
                )
                self.assertEqual(
                    response.status_code,
                    200,
                    response.text[:200] if response.status_code != 200 else "",
                )
                advanced = [
                    DEFAULTS[x]
                    for x in (
                        "do_sample",
                        "top_p",
                        "top_k",
                        "temperature",
                        "length_penalty",
                        "num_beams",
                        "repetition_penalty",
                        "max_mel_tokens",
                    )
                ]
                advanced[2] = 0
                previous = os.getcwd()
                os.chdir(directory)
                try:
                    with runtime.execution():
                        output = generate_ui(
                            runtime,
                            args,
                            2,
                            str(reference),
                            "reference.wav",
                            "你好",
                            "ZH",
                            None,
                            DEFAULTS["emo_alpha"],
                            [0.5, 0.5, 0, 0, 0, 0, 0, 0],
                            None,
                            False,
                            120,
                            advanced,
                            None,
                        )
                    self.assertEqual(Path(output).read_bytes(), response.content)
                finally:
                    os.chdir(previous)
                for key in (
                    "lang",
                    "emo_alpha",
                    "emo_vector",
                    "duration_factor",
                    "num_beams",
                    "do_sample",
                    "top_p",
                ):
                    self.assertEqual(model.calls[0][key], model.calls[1][key], key)
                self.assertNotIn("top_k", model.calls[0])
                self.assertNotIn("top_k", model.calls[1])
                self.assertEqual(node.ends, ["succeeded", "succeeded"])
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.request("TRACE", "/health").status_code, 200)

    def test_openapi_preserves_existing_form_contract(self):
        runtime = Runtime(lambda: None)
        schema = build_api(SimpleNamespace(verbose=False), runtime).openapi()
        field = schema["paths"]["/generate"]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]["$ref"].split("/")[-1]
        props = schema["components"]["schemas"][field]["properties"]
        for name, value in DEFAULTS.items():
            self.assertIn(name, props)
            if value is not None:
                self.assertEqual(props[name]["default"], value, name)
        self.assertIn("prompt_speech", props)
        self.assertIn("text", props)

    def test_copied_ui_defaults_reference_api_baseline(self):
        source = Path(__file__).resolve().parents[2] / "hub_runtime/ui.py"
        tree = ast.parse(source.read_text())
        connected = set()
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name)
                and n.value.id == "DEFAULTS"
                and isinstance(n.slice, ast.Constant)
            ):
                connected.add(n.slice.value)
        self.assertTrue(
            {
                "emo_alpha",
                "top_k",
                "temperature",
                "num_beams",
                "max_text_tokens_per_sentence",
            }
            <= connected
        )


if __name__ == "__main__":
    unittest.main()


class OriginalUI(unittest.TestCase):
    def test_original_gradio_generation_callback_uses_shared_runtime(self):
        from hub_runtime.ui import build_demo
        from hub_runtime.cli import build_parser
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights"
            weights.mkdir()
            for name in (
                "config.yaml",
                "gpt.pth",
                "bpe.model",
                "s2mel.pth",
                "codec.pth",
                "multilingual_zh_ja_yue_char_del.tiktoken",
                "wav2vec2bert_stats.pt",
                "qwen0.6bemo4-merge",
            ):
                (weights / name).write_text("fixture")
            args = build_parser().parse_args(["--model_dir", str(weights)])
            model = Model()
            node = Node()
            runtime = Runtime(
                lambda: model,
                node=node,
                version="test",
                token="t" * 32,
                state_dir=Path(directory) / "state",
            )
            app = attach(
                build_api(args, runtime),
                runtime=runtime,
                ui_factory=lambda: build_demo(args, runtime),
            )
            old = os.getcwd()
            os.chdir(directory)
            try:
                with (
                    patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}),
                    TestClient(app) as client,
                ):
                    runtime.load(LEASE)
                    config = client.get("/__gradio__/config").json()
                    dep = next(
                        x
                        for x in config["dependencies"]
                        if x.get("api_name") == "gen_single"
                    )
                    components = {x["id"]: x for x in config["components"]}
                    audio = BytesIO()
                    sf.write(
                        audio,
                        np.sin(np.arange(22050) * 2 * np.pi * 180 / 22050) * 0.1,
                        22050,
                        format="WAV",
                    )
                    uploaded = client.post(
                        "/__gradio__/gradio_api/upload",
                        files={
                            "files": ("reference.wav", audio.getvalue(), "audio/wav")
                        },
                    )
                    self.assertEqual(uploaded.status_code, 200, uploaded.text)
                    values = []
                    for ident in dep["inputs"]:
                        props = components[ident]["props"]
                        value = props.get("value")
                        if props.get("label") == "音色参考音频":
                            value = {
                                "path": uploaded.json()[0],
                                "orig_name": "reference.wav",
                                "meta": {"_type": "gradio.FileData"},
                            }
                        if props.get("label") == "文本":
                            value = "你好"
                        values.append(value)
                    queued = client.post(
                        "/__gradio__/gradio_api/call/gen_single", json={"data": values}
                    )
                    self.assertEqual(queued.status_code, 200, queued.text)
                    result = client.get(
                        "/__gradio__/gradio_api/call/gen_single/"
                        + queued.json()["event_id"]
                    )
                    self.assertIn("event: complete", result.text, result.text)
                    self.assertEqual(node.ends, ["succeeded"])
                    self.assertEqual(model.calls[0]["emo_alpha"], DEFAULTS["emo_alpha"])
                    self.assertTrue(list((Path(directory) / "outputs").glob("*.wav")))
            finally:
                os.chdir(old)
