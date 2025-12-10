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


def trim_silence(wav_data: np.ndarray, sr: int, threshold_db: float = -40.0, min_silence_duration_ms: int = 200) -> np.ndarray:
    """对音频首尾静音段进行裁剪。

    Parameters
    ----------
    wav_data : np.ndarray
        输入音频数据。
    sr : int
        采样率。
    threshold_db : float
        静音判定阈值（分贝），低于该阈值的部分视为静音。
    min_silence_duration_ms : int
        在开始和结束处保留的静音时长（毫秒）。
    """
    import librosa
    
    # 使用 librosa 来裁剪静音
    # top_db = -threshold_db（librosa 使用相对于峰值的正 dB）
    
    # Convert numpy array to float32 if needed for librosa
    if wav_data.dtype != np.float32 and wav_data.dtype != np.float64:
        wav_data = wav_data.astype(np.float32)

    # top_db：相对于参考电平的静音阈值（单位 dB）
    top_db = -threshold_db
    
    try:
        non_silent_intervals = librosa.effects.split(wav_data, top_db=top_db)
        if len(non_silent_intervals) == 0:
            return wav_data # 如果整段都是静音，则直接返回原始音频
            
        start_idx = non_silent_intervals[0][0]
        end_idx = non_silent_intervals[-1][1]
        
        # 计算需要保留的首尾静音样本数
        pad_samples = int(min_silence_duration_ms * sr / 1000)
        
        start_idx = max(0, start_idx - pad_samples)
        end_idx = min(len(wav_data), end_idx + pad_samples)
        
        return wav_data[start_idx:end_idx]
    except Exception:
        # 如果 librosa 调用失败，则直接返回原始音频作为降级策略
        return wav_data


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
