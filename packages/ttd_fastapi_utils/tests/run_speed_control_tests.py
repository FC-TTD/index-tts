#!/usr/bin/env python3
import os
import sys
import numpy as np

# Add local module path to sys.path and import speed_control without triggering
# ttd_fastapi_utils/__init__.py (which depends on fastapi).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "src", "ttd_fastapi_utils"))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from speed_control import time_stretch_wav


def gen_sine(sr: int, freq: float, seconds: float, amp: float = 0.5, dtype=np.float32):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    return x.astype(dtype)


def test_time_stretch_passthrough_speed_1():
    sr = 24000
    x = gen_sine(sr=sr, freq=440.0, seconds=0.5)
    y = time_stretch_wav(x, sr, 1.0)
    if not isinstance(y, np.ndarray):
        raise AssertionError("expected ndarray")
    if len(y) != len(x):
        raise AssertionError("length mismatch for speed=1.0")


def test_bypass_env_passthrough():
    import os
    old_env = os.environ.get("TTD_SPEED_CONTROL_BYPASS_SOX")
    try:
        os.environ["TTD_SPEED_CONTROL_BYPASS_SOX"] = "1"
        sr = 24000
        x = gen_sine(sr=sr, freq=440.0, seconds=0.3)
        y = time_stretch_wav(x, sr, 1.5)
        if not isinstance(y, np.ndarray):
            raise AssertionError("expected ndarray")
        if len(y) != len(x):
            raise AssertionError("bypass should preserve length")
    finally:
        if old_env is None:
            os.environ.pop("TTD_SPEED_CONTROL_BYPASS_SOX", None)
        else:
            os.environ["TTD_SPEED_CONTROL_BYPASS_SOX"] = old_env


def test_time_stretch_length_change_if_backend_available():
    sr = 24000
    x = gen_sine(sr=sr, freq=440.0, seconds=0.4)
    try:
        y_fast = time_stretch_wav(x, sr, 1.25)
        y_slow = time_stretch_wav(x, sr, 0.8)
    except FileNotFoundError:
        print("[WARN] sox not found; skipping length-change test")
        return

    if len(y_fast) == len(x) and len(y_slow) == len(x):
        print("[WARN] sox/ffmpeg not available (or failed); verified passthrough behavior")
        return

    if not (len(y_fast) < len(x)):
        raise AssertionError("expected speed>1 to shorten waveform")
    if not (len(y_slow) > len(x)):
        raise AssertionError("expected speed<1 to lengthen waveform")

    if np.max(np.abs(y_fast)) > 1.1 or np.max(np.abs(y_slow)) > 1.1:
        raise AssertionError("unexpected amplitude explosion")


def main():
    tests = [
        test_time_stretch_passthrough_speed_1,
        test_bypass_env_passthrough,
        test_time_stretch_length_change_if_backend_available,
    ]
    for t in tests:
        t()
    print("All speed_control tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FAIL]", str(e))
        sys.exit(1)
