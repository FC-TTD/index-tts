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

from ttd_fastapi_utils.postprocess import loudnorm, eq, apply_postprocess


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


def main():
    tests = [
        test_loudnorm_short_audio_padding_no_exception,
        test_loudnorm_multichannel_first_channel_only_consistency,
        test_apply_postprocess_wraps_and_safe,
        test_eq_runs_and_limits,
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
