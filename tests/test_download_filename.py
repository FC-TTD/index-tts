from types import SimpleNamespace

import gradio as gr

from download_filename import (
    build_content_disposition,
    build_download_filename,
    sanitize_filename_part,
)
from webui import NamedAudio, NamedAudioInput


def test_build_download_filename_preserves_chinese_and_epoch():
    assert (
        build_download_filename("音色/角色A.wav", "你好/世界", epoch_ms=1234567890)
        == "角色A-你好_世界-1234567890.wav"
    )


def test_sanitize_filename_part_replaces_illegal_and_fallbacks():
    assert (
        sanitize_filename_part("../a<b:c*?.wav", "source", 48, strip_extension=True)
        == "a_b_c"
    )
    assert sanitize_filename_part("hello.world", "text", 48) == "hello.world"
    assert sanitize_filename_part("", "source", 48) == "source"
    assert sanitize_filename_part("   ...___", "text", 40) == "text"


def test_build_download_filename_truncates_parts():
    filename = build_download_filename("s" * 60 + ".wav", "文" * 45, epoch_ms=7)
    source, text, epoch_ext = filename.rsplit("-", 2)
    assert len(source) == 48
    assert len(text) == 40
    assert epoch_ext == "7.wav"


def test_content_disposition_has_ascii_and_utf8_filename():
    header = build_content_disposition("角色-你好-1.wav")
    assert 'filename="??-??-1.wav"' in header
    assert "filename*=UTF-8''" in header
    assert "%E8%A7%92%E8%89%B2" in header


def test_named_audio_preprocess_keeps_orig_name(monkeypatch):
    def fake_preprocess(self, payload):
        return payload.path

    monkeypatch.setattr(gr.Audio, "preprocess", fake_preprocess)
    component = NamedAudio(type="filepath")
    result = component.preprocess(
        SimpleNamespace(path="/tmp/upload.wav", orig_name="原始.wav")
    )

    assert isinstance(result, NamedAudioInput)
    assert result.path == "/tmp/upload.wav"
    assert result.orig_name == "原始.wav"
