#!/usr/bin/env python3
import sys
import os
import math
import numpy as np

try:
    import pyloudnorm as _pyln  # noqa: F401
except Exception:
    print("[SKIP] pyloudnorm is not available; skipping postprocess tests.")
    sys.exit(0)

# Add local src to sys.path for monorepo execution without installation
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ttd_fastapi_utils.postprocess import (
    apply_postprocess,
    bandpass,
    delay,
    eq,
    highpass,
    lowpass,
    loudnorm,
    mix,
    reverb,
    saturate,
)
from ttd_fastapi_utils import preset


def gen_sine(sr: int, freq: float, seconds: float, amp: float = 0.5, dtype=np.float32):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    return x.astype(dtype)


def assert_allclose(a, b, atol=1e-6, rtol=1e-6):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError("arrays not close within tolerance")


def test_loudnorm_short_audio_padding_no_exception():
    sr = 22050
    x = gen_sine(sr=sr, freq=1000.0, seconds=0.232, amp=0.5)
    y, lufs = loudnorm(x, sr, target_loudness=-23.0, threshold=0.99, block_sec=0.4)
    if not isinstance(y, np.ndarray):
        raise AssertionError("expected ndarray output")
    if y.ndim != 1:
        raise AssertionError("expected mono output")
    if len(y) != len(x):
        raise AssertionError("output length mismatch")
    if np.max(np.abs(y)) > 0.99 + 1e-6:
        raise AssertionError("peak safety limiter failed")
    if not isinstance(lufs, float):
        raise AssertionError("expected float LUFS")


def test_loudnorm_multichannel_first_channel_only_consistency():
    sr = 24000
    dur = 0.5
    ch0 = gen_sine(sr=sr, freq=800.0, seconds=dur, amp=0.5)
    ch1 = gen_sine(sr=sr, freq=400.0, seconds=dur, amp=0.9)
    x_stereo = np.stack([ch0, ch1], axis=-1)

    y_stereo, lufs_stereo = loudnorm(x_stereo, sr, target_loudness=-23.0, threshold=0.99, block_sec=0.4)
    y_ch0, lufs_ch0 = loudnorm(ch0, sr, target_loudness=-23.0, threshold=0.99, block_sec=0.4)

    if y_stereo.shape != y_ch0.shape or y_ch0.shape != ch0.shape:
        raise AssertionError("shape mismatch")
    assert_allclose(y_stereo, y_ch0)
    if not (math.isnan(lufs_stereo) and math.isnan(lufs_ch0)):
        if abs(lufs_stereo - lufs_ch0) >= 1e-3:
            raise AssertionError("LUFS mismatch beyond tolerance")


def test_apply_postprocess_wraps_and_safe():
    sr = 16000
    x = gen_sine(sr=sr, freq=1000.0, seconds=0.2, amp=0.7)
    y = apply_postprocess(x, sr, enable=True)
    if not isinstance(y, np.ndarray) or len(y) != len(x):
        raise AssertionError("apply_postprocess output invalid")


def test_eq_runs_and_limits():
    sr = 44100
    x = gen_sine(sr=sr, freq=4000.0, seconds=0.5, amp=0.5)
    y = eq(x, sr)
    if not isinstance(y, np.ndarray):
        raise AssertionError("eq output not ndarray")
    if np.max(np.abs(y)) > 1.0 + 1e-6:
        raise AssertionError("eq peak limit exceeded")


def test_basic_filters_run_and_preserve_shape():
    sr = 24000
    x = gen_sine(sr=sr, freq=600.0, seconds=0.5, amp=0.5)
    y_lp = lowpass(x, sr, cutoff_hz=2000.0)
    y_hp = highpass(x, sr, cutoff_hz=200.0)
    y_bp = bandpass(x, sr, low_cut_hz=250.0, high_cut_hz=3000.0)
    for name, y in [("lowpass", y_lp), ("highpass", y_hp), ("bandpass", y_bp)]:
        if not isinstance(y, np.ndarray) or y.shape != x.shape:
            raise AssertionError(f"{name} output invalid")


def test_saturate_runs_and_limits():
    sr = 24000
    x = gen_sine(sr=sr, freq=900.0, seconds=0.5, amp=0.5)
    y = saturate(x, drive=1.8)
    if not isinstance(y, np.ndarray) or y.shape != x.shape:
        raise AssertionError("saturate output invalid")
    if np.max(np.abs(y)) > 1.0 + 1e-6:
        raise AssertionError("saturate peak limit exceeded")


def test_delay_and_mix_run():
    sr = 22050
    x = gen_sine(sr=sr, freq=700.0, seconds=0.35, amp=0.5)
    delayed = delay(x, sr, delay_ms=60.0, decay=0.35, repeats=2)
    y = mix(x, delayed, wet_ratio=0.4)
    if not isinstance(y, np.ndarray) or y.shape != x.shape:
        raise AssertionError("delay/mix output invalid")


def test_reverb_runs_and_preserve_shape():
    sr = 24000
    x = gen_sine(sr=sr, freq=520.0, seconds=0.45, amp=0.5)
    y = reverb(x, sr, room_size=0.45, damping=0.35, pre_delay_ms=18.0)
    if not isinstance(y, np.ndarray) or y.shape != x.shape:
        raise AssertionError("reverb output invalid")
    if not np.isfinite(y).all():
        raise AssertionError("reverb output contains non-finite values")


def test_mix_shape_mismatch_raises():
    x = np.zeros(16, dtype=np.float32)
    y = np.zeros(15, dtype=np.float32)
    try:
        mix(x, y, wet_ratio=0.5)
    except ValueError:
        return
    raise AssertionError("mix should raise on shape mismatch")


def test_preset_list_and_apply():
    names = tuple(preset.list_presets())
    expected = ("telephone", "smart_assistant", "inner_monologue", "radio", "intercom")
    if names != expected:
        raise AssertionError("preset list mismatch")

    sr = 24000
    x = gen_sine(sr=sr, freq=750.0, seconds=0.45, amp=0.5)
    for name in expected:
        y = preset.apply_preset(name, x, sr)
        if not isinstance(y, np.ndarray) or y.shape != x.shape:
            raise AssertionError("preset %s output invalid" % name)
        if np.max(np.abs(y)) > 1.0 + 1e-6:
            raise AssertionError("preset %s peak limit exceeded" % name)


def test_preset_aliases_work():
    sr = 22050
    x = gen_sine(sr=sr, freq=680.0, seconds=0.4, amp=0.5)
    y_phone = preset.apply_preset("电话", x, sr)
    y_inner = preset.apply_preset("心声", x, sr)
    if y_phone.shape != x.shape or y_inner.shape != x.shape:
        raise AssertionError("preset alias output invalid")


def main():
    tests = [
        test_loudnorm_short_audio_padding_no_exception,
        test_loudnorm_multichannel_first_channel_only_consistency,
        test_apply_postprocess_wraps_and_safe,
        test_eq_runs_and_limits,
        test_basic_filters_run_and_preserve_shape,
        test_saturate_runs_and_limits,
        test_delay_and_mix_run,
        test_reverb_runs_and_preserve_shape,
        test_mix_shape_mismatch_raises,
        test_preset_list_and_apply,
        test_preset_aliases_work,
    ]
    for t in tests:
        t()
    print("All postprocess tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FAIL]", str(e))
        sys.exit(1)
