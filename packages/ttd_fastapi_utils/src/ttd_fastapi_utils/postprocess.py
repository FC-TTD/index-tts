import logging
import numpy as np
import pyloudnorm as pyln
import scipy.signal as signal
from typing import Tuple

logger = logging.getLogger(__name__)


def loudnorm(
    wav_data: np.ndarray,
    sr: int,
    target_loudness: float = -23,
    threshold: float = 0.99,
    block_sec: float = 0.4,
) -> Tuple[np.ndarray, float]:
    """
    Loudness normalization (LUFS) with short-audio padding and peak safety limiting.
    - Multi-channel input: only the first channel is used and returned.
    """
    try:
        if wav_data is None:
            return wav_data, float("nan")

        # Ensure float ndarray; for multi-channel, only use the first channel
        if isinstance(wav_data, list):
            wav_data = np.asarray(wav_data, dtype=np.float32)
        if not np.issubdtype(wav_data.dtype, np.floating):
            wav_data = wav_data.astype(np.float32)
        if wav_data.ndim == 2:
            try:
                logger.warning("后处理: 输入为多通道音频，仅使用第一个声道参与处理并作为输出")
            except Exception:
                pass
            wav_mono = wav_data[:, 0]
        else:
            wav_mono = wav_data

        n_samples = wav_mono.shape[0]
        min_samples = int(sr * block_sec)
        pad_needed = max(0, (min_samples + 1) - n_samples)
        if pad_needed > 0:
            logger.warning(
                "后处理: 音频过短(%d samples, %.3fs) 小于 LUFS block_size %.3fs, 将静音填充 %d samples 后计算",
                n_samples,
                n_samples / float(sr) if sr > 0 else -1.0,
                block_sec,
                pad_needed,
            )
            wav_for_meter = np.pad(wav_mono, (0, pad_needed), mode="constant")
        else:
            wav_for_meter = wav_mono

        meter = pyln.Meter(sr, block_size=block_sec)
        original_loudness: float = meter.integrated_loudness(wav_for_meter)

        wav_norm = pyln.normalize.loudness(wav_mono, original_loudness, target_loudness)

    except ValueError as ve:
        logger.warning("后处理: LUFS 计算失败(ValueError)，将回退为仅峰值限幅: %s", str(ve))
        original_loudness = float("nan")
        wav_norm = wav_mono if 'wav_mono' in locals() else wav_data
    except Exception:
        logger.exception("后处理: LUFS 归一化异常，将回退为仅峰值限幅")
        original_loudness = float("nan")
        wav_norm = wav_mono if 'wav_mono' in locals() else wav_data

    abs_wav = np.abs(wav_norm)
    abs_wav = np.where(abs_wav == 0, 1e-10, abs_wav)
    reduction_factor = np.where(abs_wav > threshold, threshold / abs_wav, 1.0)
    limited_audio = wav_norm * reduction_factor
    return limited_audio, original_loudness


def eq(wav_data: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple EQ: enhance a high-frequency band while ensuring stability across sample rates.
    """
    def enhance_frequency_band(audio, sample_rate, low_freq, high_freq, gain_factor):
        nyquist = 0.5 * sample_rate
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)
        if low >= high or low <= 0 or high >= 1:
            return audio
        b, a = signal.butter(4, [low, high], btype="bandpass")
        filtered_audio = signal.filtfilt(b, a, audio)
        enhanced_audio = audio + gain_factor * filtered_audio
        return enhanced_audio

    nyquist = 0.5 * sr
    low_freq = min(5000, nyquist * 0.6)
    high_freq = min(10000, nyquist * 0.95)
    gain_factor = 0.2

    audio = wav_data.astype(np.float64)
    enhanced_audio = enhance_frequency_band(audio, sr, low_freq, high_freq, gain_factor)
    enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
    return enhanced_audio


def apply_postprocess(wav: np.ndarray, sr: int, enable: bool = True) -> np.ndarray:
    """Apply loudnorm + eq with exception safety."""
    if not enable:
        return wav
    try:
        wav_p, _ = loudnorm(wav, sr)
        wav_p = eq(wav_p, sr)
        return wav_p
    except Exception:
        logger.exception("后处理阶段失败，已跳过后处理并返回原始生成音频")
        return wav
