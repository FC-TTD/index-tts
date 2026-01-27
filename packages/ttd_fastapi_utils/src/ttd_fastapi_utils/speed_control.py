import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def _as_float32_mono(wav: np.ndarray) -> np.ndarray:
    if wav is None:
        return wav

    if isinstance(wav, list):
        wav = np.asarray(wav)

    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    if wav.ndim == 2:
        if wav.shape[0] < wav.shape[1]:
            wav = np.mean(wav, axis=0)
        else:
            wav = np.mean(wav, axis=1)

    return wav


def _write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    wav_f32 = _as_float32_mono(wav)
    wav_i16 = np.clip(wav_f32, -1.0, 1.0)
    wav_i16 = (wav_i16 * 32767.0).astype(np.int16)
    wavfile.write(path, int(sr), wav_i16)


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data is None:
        return np.zeros((0,), dtype=np.float32), int(sr)

    if data.dtype == np.int16:
        wav = (data.astype(np.float32) / 32768.0).astype(np.float32)
    elif np.issubdtype(data.dtype, np.floating):
        wav = data.astype(np.float32)
    else:
        wav = data.astype(np.float32)
        m = float(np.max(np.abs(wav))) if wav.size else 1.0
        if m > 0:
            wav = wav / m

    wav = _as_float32_mono(wav)
    return wav, int(sr)


def _has_bin(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def time_stretch_wav(
    wav: np.ndarray,
    sr: int,
    speed: float,
    *,
    allow_passthrough_on_failure: bool = True,
) -> np.ndarray:
    if wav is None:
        return wav

    spd = float(speed)
    if spd <= 0 or abs(spd - 1.0) < 1e-9:
        return wav

    bypass = bool(os.getenv("TTD_SPEED_CONTROL_BYPASS_SOX"))
    if bypass:
        logger.warning("speed_control: bypassing SoX due to TTD_SPEED_CONTROL_BYPASS_SOX")
        return wav

    in_path: Optional[str] = None
    out_path: Optional[str] = None
    try:
        fd_in, in_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_in)
        fd_out, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_out)

        _write_wav(in_path, wav, sr)

        if not _has_bin("sox"):
            raise FileNotFoundError("sox is required but not found on this system")
        _run(["sox", in_path, out_path, "tempo", "-s", str(spd)])

        stretched, sr2 = _read_wav(out_path)
        if int(sr2) != int(sr):
            logger.warning("speed_control: sample rate changed %s -> %s; keeping waveform and original sr", sr2, sr)
        return stretched.astype(np.float32)
    except Exception as e:
        if allow_passthrough_on_failure:
            logger.warning("speed_control: time_stretch failed (%s), passthrough", str(e))
            return wav
        raise
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def apply_speed_to_wav_list(
    wavs: Sequence[np.ndarray],
    sr: int,
    speed: float,
    *,
    allow_passthrough_on_failure: bool = True,
) -> List[np.ndarray]:
    return [
        time_stretch_wav(
            w,
            sr,
            speed,
            allow_passthrough_on_failure=allow_passthrough_on_failure,
        )
        for w in wavs
    ]
