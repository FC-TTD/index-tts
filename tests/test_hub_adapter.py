"""Weight-free service-boundary checks, runnable with Python's standard library."""

import ast
import argparse
import asyncio
import contextlib
import contextvars
import importlib
import inspect
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hub_adapter

ROOT = Path(__file__).resolve().parents[1]


class FakeManaged:
    def __init__(self, loader, cleanup):
        self.loader = loader
        self.cleanup = cleanup
        self.model = None
        self.depth = contextvars.ContextVar("depth", default=0)
        self.begins = 0
        self.finishes = 0

    @classmethod
    def from_env(cls, loader, cleanup, synchronize):
        result = cls(loader, cleanup)
        result.synchronize = synchronize
        return result

    def get(self):
        if not self.depth.get() or self.model is None:
            raise RuntimeError("ungranted access")
        return self.model

    @contextlib.contextmanager
    def execution(self):
        if self.depth.get():
            yield
            return
        self.model = self.loader()  # Fixture simulates a node-confirmed grant/load.
        self.begins += 1
        token = self.depth.set(1)
        try:
            yield
        finally:
            self.depth.reset(token)
            self.finishes += 1


def fake_managed_call(getter):
    def decorate(function):
        def wrapped(*args, **kwargs):
            with getter().execution():
                return function(*args, **kwargs)
        return wrapped
    return decorate


class HubAdapterTests(unittest.TestCase):
    def test_legacy_decorator_does_not_require_sdk_or_change_callback(self):
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "0"}):
            callback = lambda: "native"
            self.assertIs(hub_adapter.managed_call(lambda: None)(callback), callback)

    def test_managed_factory_is_lazy_and_nested_callbacks_share_admission(self):
        sdk = SimpleNamespace(ManagedModel=FakeManaged, managed_call=fake_managed_call)
        calls = []
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"}), patch.dict(sys.modules, {"ttd_hub_runtime": sdk}), patch.object(hub_adapter, "observe_model_devices", lambda model: None):
            manager = hub_adapter.create_managed_model(lambda: calls.append("load") or SimpleNamespace(infer=lambda: "native inference"))
            self.assertEqual(calls, [])
            self.assertIs(manager.cleanup, hub_adapter.cleanup_cuda)
            self.assertIs(manager.synchronize, hub_adapter.synchronize_cuda)
            with self.assertRaisesRegex(RuntimeError, "ungranted"):
                manager.get()

            @hub_adapter.managed_call(lambda: manager)
            def inner():
                return manager.get()

            @hub_adapter.managed_call(lambda: manager)
            def outer():
                self.assertIs(inner(), manager.get())
                self.assertEqual(manager.get().infer(), "native inference")
                raise ValueError("inference failed")

            with self.assertRaisesRegex(ValueError, "inference failed"):
                outer()
            self.assertEqual((manager.begins, manager.finishes), (1, 1))
            self.assertEqual(calls, ["load"])
            with self.assertRaisesRegex(RuntimeError, "ungranted"):
                manager.get()

    def test_api_factory_preserves_precision_and_routes_only_opted_in_manager(self):
        tree = ast.parse((ROOT / "api.py").read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_tts_manager")
        calls = []
        ns = dict(vars(hub_adapter), os=os, argparse=argparse, IndexTTS2=lambda **kw: calls.append(kw) or kw,
                  SmartModel=lambda loader, timeout_seconds: SimpleNamespace(loader=loader, ttl=timeout_seconds))
        exec(compile(ast.Module(body=[factory], type_ignores=[]), "api_factory", "exec"), ns)
        args = SimpleNamespace(model_dir="checkpoints", bf16=True, device=None, use_cuda_kernel=False,
                               use_deepspeed=False, use_accel=False, use_torch_compile=False)
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "0"}):
            native = ns["create_tts_manager"](args)
            self.assertEqual(native.ttl, 7200)
            native.loader()
            self.assertIsNone(calls[-1]["device"])
        ns["create_managed_model"] = lambda loader: FakeManaged(loader, lambda: None)
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"}):
            managed = ns["create_tts_manager"](args)
            self.assertIsInstance(managed, FakeManaged)
            self.assertEqual(len(calls), 1)
            with managed.execution():
                managed.get()
            self.assertIsNone(calls[-1]["device"])
            self.assertIs(calls[-1]["use_bf16"], True)
            self.assertIs(calls[-1]["use_qwen_emo"], True)

    def test_observation_accepts_absent_optional_components(self):
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"}):
            model = SimpleNamespace()
            hub_adapter.observe_model_devices(model)
            self.assertEqual(model.__hub_device_summary__, {"components": {}})

    def test_cleanup_only_reclaims_and_never_moves_weights(self):
        calls = []
        cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: calls.append("sync"),
                               empty_cache=lambda: calls.append("empty"), ipc_collect=lambda: calls.append("ipc"))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}), patch.object(hub_adapter.gc, "collect", lambda: calls.append("gc")):
            hub_adapter.cleanup_cuda()
        self.assertEqual(calls, ["gc", "sync", "empty", "ipc"])

    def test_qwen_loader_preserves_official_auto_placement_and_precision(self):
        tree = ast.parse((ROOT / "indextts/infer_v2_5.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenEmotion")
        cls.body = [next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")]
        calls = []
        ns = dict(AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: object()),
                  AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda path, **kw: calls.append(kw)))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "qwen_loader", "exec"), ns)
        for enabled, expected in (("0", "auto"), ("1", "auto")):
            with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": enabled}):
                ns["QwenEmotion"]("fixture")
                self.assertEqual(calls[-1], {"torch_dtype": "float16", "device_map": expected})

    def test_managed_import_mutes_inherited_notification_url(self):
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1", "TTD_WEBHOOK_URL": "test-only"}):
            importlib.reload(hub_adapter)
            self.assertEqual(os.environ["TTD_WEBHOOK_URL"], "")

    def test_api_and_ui_model_access_callbacks_have_execution_scopes(self):
        expected = {"generate_audio", "format_glossary_markdown", "gen_single", "on_input_text_change",
                    "on_add_glossary_term", "on_glossary_checkbox_change", "on_demo_load"}
        decorated = set()
        for name in ("api.py", "webui.py"):
            for node in ast.walk(ast.parse((ROOT / name).read_text())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in {"managed_call", "managed_api_call"}:
                            decorated.add(node.name)
        self.assertEqual(decorated, expected)

    def test_managed_api_and_ui_inference_share_cache_lock(self):
        active = 0
        peak = 0
        start = threading.Barrier(3)

        @hub_adapter.serialize_managed_inference
        def infer():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(0.02)
            active -= 1

        def invoke():
            start.wait()
            infer()

        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"}):
            threads = [threading.Thread(target=invoke) for _ in range(2)]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join()
        self.assertEqual(peak, 1)

    def test_api_maps_control_errors_and_preserves_callback_signature(self):
        class HubError(Exception):
            def __init__(self, code):
                self.code = code

        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail

        sdk = SimpleNamespace(HubError=HubError, managed_call=lambda getter: lambda fn: fn)
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"}), patch.dict(sys.modules, {
            "ttd_hub_runtime": sdk, "fastapi": SimpleNamespace(HTTPException=HTTPException),
        }):
            for code, status in (("resource_busy", 503), ("model_preparing", 503), ("node_unavailable", 503),
                                 ("execution_unknown", 503), ("conflict", 409), ("duplicate", 409), ("cancelled", 409)):
                async def callback(text: str, speed: float = 1.0):
                    raise HubError(code)
                decorated = hub_adapter.managed_api_call(lambda: None)(callback)
                self.assertEqual(inspect.signature(decorated), inspect.signature(callback))
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(decorated("fixture"))
                self.assertEqual((caught.exception.status_code, caught.exception.detail), (status, code))

    def test_context_middleware_is_managed_only_and_terminal_sync_waits_for_cuda(self):
        calls = []
        middleware = object()
        app = SimpleNamespace(add_middleware=lambda value: calls.append(value))
        with patch.dict(sys.modules, {"ttd_hub_runtime": SimpleNamespace(HubContextMiddleware=middleware)}):
            with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "0"}):
                hub_adapter.add_managed_middleware(app)
                self.assertEqual(calls, [])
            with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"}):
                hub_adapter.add_managed_middleware(app)
                self.assertEqual(calls, [middleware])
        cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: calls.append("synchronized"))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            hub_adapter.synchronize_cuda()
        self.assertEqual(calls[-1], "synchronized")

    def test_device_summary_observes_mixed_native_placement_without_mutation(self):
        import json

        tensor = lambda device="cuda:0", dtype="torch.float32": SimpleNamespace(device=device, dtype=dtype)

        class Module:
            def __init__(self):
                self.weight = tensor(dtype="torch.bfloat16")
                self.buffer = tensor()
                self.hf_device_map = {"": "cuda:0"}

            def named_parameters(self):
                return [("weight", self.weight)]

            def named_buffers(self):
                return [("buffer", self.buffer)]

        def model_fixture():
            model = SimpleNamespace(**{name: Module() for name in (
                "gpt", "semantic_model", "semantic_codec", "s2mel", "campplus_model", "bigvgan")})
            model.qwen_emo = SimpleNamespace(model=Module())
            model.semantic_mean = tensor()
            model.semantic_std = tensor()
            model.emo_matrix = (tensor(), tensor())
            model.spk_matrix = (tensor(),)
            # Native CPU-only helpers have no tensor inspection API and must be ignored.
            model.extract_features = object()
            model.tokenizer = object()
            model.text_process = object()
            return model

        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "1"}):
            model = model_fixture()
            model.gpt.buffer.device = "cpu"
            model.semantic_mean.device = "cpu"
            model.qwen_emo.model.weight.device = "meta"
            model.qwen_emo.model.hf_device_map = {"layer0": "cpu", "layer1": "cuda:0", "layer2": "disk"}
            hub_adapter.observe_model_devices(model)
            summary = json.loads(json.dumps(model.__hub_device_summary__))
            self.assertEqual(len(summary["components"]), 11)
            self.assertEqual(summary["components"]["gpt"]["devices"], ["cpu", "cuda:0"])
            self.assertEqual(summary["components"]["semantic_mean"]["devices"], ["cpu"])
            self.assertEqual(summary["components"]["qwen_emo.model"]["device_map"], model.qwen_emo.model.hf_device_map)
            self.assertEqual(model.gpt.buffer.device, "cpu")
            self.assertEqual(model.qwen_emo.model.weight.device, "meta")

    def test_unmanaged_observer_does_not_inspect_model(self):
        with patch.dict(os.environ, {"HUB_RUNTIME_ENABLED": "0"}):
            hub_adapter.observe_model_devices(object())


if __name__ == "__main__":
    unittest.main()
