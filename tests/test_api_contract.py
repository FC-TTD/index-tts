from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import api


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "ZH"),
        ("zh-cn", "ZH"),
        ("EN", "EN"),
        ("日语", "JA"),
        ("spanish", "ES"),
        ("arabic", "AR"),
    ],
)
def test_normalize_language(value, expected):
    assert api.normalize_language(value) == expected


def test_normalize_language_rejects_unknown_value():
    with pytest.raises(ValueError, match="language must be one of"):
        api.normalize_language("FR")


def test_v25_loader_keeps_smart_model_lazy(monkeypatch, tmp_path):
    calls = []

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(api, "IndexTTS2", FakeIndexTTS2)
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        bf16=True,
        device=None,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
    )

    manager = api.create_tts_manager(args)
    try:
        assert calls == []
        manager.get()
        assert calls == [
            {
                "cfg_path": str(tmp_path / "config.yaml"),
                "model_dir": str(tmp_path),
                "use_bf16": True,
                "device": None,
                "use_cuda_kernel": False,
                "use_deepspeed": False,
                "use_accel": False,
                "use_torch_compile": False,
                "use_qwen_emo": True,
            }
        ]
    finally:
        manager.stop()


class FakeManager:
    def __init__(self, model):
        self.model = model
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        return self.model

    def stop(self):
        pass


def _args(model_dir):
    return SimpleNamespace(
        model_dir=str(model_dir),
        bf16=True,
        device=None,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        preload_model=False,
        verbose=False,
        host="127.0.0.1",
        port=8000,
    )


def _complete_model_dir(path):
    path.mkdir()
    for name in (
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "codec.pth",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "wav2vec2bert_stats.pt",
    ):
        (path / name).touch()
    (path / "qwen0.6bemo4-merge").mkdir()


def test_generate_contract_defaults_language_and_preserves_speed_semantics(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "checkpoints"
    _complete_model_dir(model_dir)
    infer_calls = []

    class FakeModel:
        def infer(self, output_path, **kwargs):
            infer_calls.append(kwargs)
            sf.write(output_path, np.zeros(2205, dtype=np.float32), 22050)
            return output_path

    manager = FakeManager(FakeModel())
    monkeypatch.setattr(api, "create_tts_manager", lambda args: manager)
    app = api.create_app(_args(model_dir))

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={"text": "hello", "speed": "1.25", "postprocess": "false"},
            files={"prompt_speech": ("voice.wav", b"RIFF", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert infer_calls[0]["lang"] == "ZH"
    assert infer_calls[0]["duration_factor"] == 0.8


def test_expected_duration_retries_in_the_converging_direction(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "checkpoints"
    _complete_model_dir(model_dir)
    duration_factors = []

    class FakeModel:
        def infer(self, output_path, **kwargs):
            duration_factor = kwargs["duration_factor"]
            duration_factors.append(duration_factor)
            sample_count = round(22050 * 4.0 * duration_factor)
            sf.write(output_path, np.ones(sample_count, dtype=np.float32), 22050)
            return output_path

    manager = FakeManager(FakeModel())
    monkeypatch.setattr(api, "create_tts_manager", lambda args: manager)
    app = api.create_app(_args(model_dir))

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "text": "hello",
                "expected_duration": "2.0",
                "remove_silence": "false",
                "postprocess": "false",
            },
            files={"prompt_speech": ("voice.wav", b"RIFF", "audio/wav")},
        )

    assert response.status_code == 200
    assert duration_factors == pytest.approx([1.0, 0.5])


def test_generate_contract_accepts_language_without_renaming_fields(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "checkpoints"
    _complete_model_dir(model_dir)
    infer_calls = []

    class FakeModel:
        def infer(self, output_path, **kwargs):
            infer_calls.append(kwargs)
            sf.write(output_path, np.zeros(2205, dtype=np.float32), 22050)
            return output_path

    manager = FakeManager(FakeModel())
    monkeypatch.setattr(api, "create_tts_manager", lambda args: manager)
    app = api.create_app(_args(model_dir))

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={"text": "hello", "language": "english", "postprocess": "false"},
            files={"prompt_speech": ("voice.wav", b"RIFF", "audio/wav")},
        )

    assert response.status_code == 200
    assert infer_calls[0]["lang"] == "EN"


def test_invalid_language_does_not_load_model(monkeypatch, tmp_path):
    model_dir = tmp_path / "checkpoints"
    _complete_model_dir(model_dir)
    manager = FakeManager(object())
    monkeypatch.setattr(api, "create_tts_manager", lambda args: manager)
    app = api.create_app(_args(model_dir))

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={"text": "bonjour", "language": "FR"},
            files={"prompt_speech": ("voice.wav", b"RIFF", "audio/wav")},
        )

    assert response.status_code == 400
    assert manager.get_calls == 0
