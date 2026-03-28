import logging
import numpy as np
import pyloudnorm as pyln
import scipy.signal as signal
from typing import Tuple, Union

logger = logging.getLogger(__name__)
_LIBROSA_IMPORT_WARNED = False


def _ensure_float_audio(wav_data: np.ndarray) -> np.ndarray:
    if isinstance(wav_data, list):
        wav_data = np.asarray(wav_data, dtype=np.float32)
    elif not isinstance(wav_data, np.ndarray):
        wav_data = np.asarray(wav_data)
    if not np.issubdtype(wav_data.dtype, np.floating):
        wav_data = wav_data.astype(np.float32)
    return wav_data


def butter_filter(
    audio: np.ndarray,
    sr: int,
    *,
    btype: str,
    cutoff: Union[float, Tuple[float, float]],
    order: int = 4,
) -> np.ndarray:
    nyquist = 0.5 * sr
    if nyquist <= 0:
        return audio

    if isinstance(cutoff, tuple):
        low_hz, high_hz = cutoff
        low = max(low_hz / nyquist, 1e-6)
        high = min(high_hz / nyquist, 0.99)
        if low >= high:
            return audio
        wn = (low, high)
    else:
        wn = max(min(cutoff / nyquist, 0.99), 1e-6)

    try:
        b, a = signal.butter(order, wn, btype=btype)
        return signal.filtfilt(b, a, audio)
    except Exception:
        logger.exception("后处理: %s 滤波失败，已回退为原始音频", btype)
        return audio


def lowpass(
    wav_data: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """低通滤波。"""
    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)
    filtered = butter_filter(audio, sr, btype="lowpass", cutoff=float(cutoff_hz), order=order)
    return filtered.astype(orig_dtype, copy=False)


def highpass(
    wav_data: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """高通滤波。"""
    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)
    filtered = butter_filter(audio, sr, btype="highpass", cutoff=float(cutoff_hz), order=order)
    return filtered.astype(orig_dtype, copy=False)


def bandpass(
    wav_data: np.ndarray,
    sr: int,
    low_cut_hz: float,
    high_cut_hz: float,
    order: int = 4,
) -> np.ndarray:
    """带通滤波。"""
    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)
    filtered = butter_filter(
        audio,
        sr,
        btype="bandpass",
        cutoff=(float(low_cut_hz), float(high_cut_hz)),
        order=order,
    )
    return filtered.astype(orig_dtype, copy=False)


def delay(
    wav_data: np.ndarray,
    sr: int,
    *,
    delay_ms: float,
    decay: float,
    repeats: int,
) -> np.ndarray:
    """多次衰减延迟，返回包含原始信号的 composite signal。"""
    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)
    delay_samples = max(int(sr * delay_ms / 1000.0), 1)
    repeats = max(int(repeats), 0)
    if repeats == 0 or decay == 0.0:
        return audio.astype(orig_dtype, copy=True)

    out = audio.astype(np.float64, copy=True)
    for i in range(1, repeats + 1):
        gain = decay ** i
        start = delay_samples * i
        if start >= len(audio):
            break
        out[start:] += gain * audio[:-start]
    return out.astype(orig_dtype, copy=False)


def delay_tail(
    wav_data: np.ndarray,
    sr: int,
    *,
    delay_ms: float,
    decay: float,
    repeats: int,
) -> np.ndarray:
    """多次衰减延迟，仅返回 wet-only echo tail，不包含原始信号。"""
    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)
    delay_samples = max(int(sr * delay_ms / 1000.0), 1)
    repeats = max(int(repeats), 0)
    out = np.zeros_like(audio, dtype=np.float64)
    if repeats == 0 or decay == 0.0:
        return out.astype(orig_dtype, copy=False)

    for i in range(1, repeats + 1):
        gain = decay ** i
        start = delay_samples * i
        if start >= len(audio):
            break
        out[start:] += gain * audio[:-start]
    return out.astype(orig_dtype, copy=False)


def saturate(
    wav_data: np.ndarray,
    drive: float = 1.3,
) -> np.ndarray:
    """基于 tanh 的软削波/饱和。"""
    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)
    drive = max(float(drive), 1e-6)
    saturated = np.tanh(audio * drive) / np.tanh(drive)
    return saturated.astype(orig_dtype, copy=False)


def mix(
    dry: np.ndarray,
    wet: np.ndarray,
    wet_ratio: float = 0.5,
) -> np.ndarray:
    """干湿混合。"""
    dry_audio = _ensure_float_audio(dry)
    wet_audio = _ensure_float_audio(wet)
    wet_ratio = float(np.clip(wet_ratio, 0.0, 1.0))
    if dry_audio.shape != wet_audio.shape:
        raise ValueError("dry and wet audio must have the same shape")
    mixed = (1.0 - wet_ratio) * dry_audio + wet_ratio * wet_audio
    return mixed.astype(dry_audio.dtype, copy=False)


def _apply_channelwise(
    wav_data: np.ndarray,
    processor,
) -> np.ndarray:
    if wav_data.ndim == 1:
        return processor(wav_data)
    if wav_data.ndim != 2:
        raise ValueError("audio must be 1D or 2D")

    if wav_data.shape[0] < wav_data.shape[1]:
        channels = [processor(wav_data[idx, :]) for idx in range(wav_data.shape[0])]
        return np.stack(channels, axis=0)

    channels = [processor(wav_data[:, idx]) for idx in range(wav_data.shape[1])]
    return np.stack(channels, axis=1)


def reverb(
    wav_data: np.ndarray,
    sr: int,
    *,
    room_size: float = 0.45,
    damping: float = 0.35,
    pre_delay_ms: float = 18.0,
) -> np.ndarray:
    """轻量 Schroeder 风格混响，返回带残响的 wet signal。"""
    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)

    room_size = float(np.clip(room_size, 0.0, 1.0))
    damping = float(np.clip(damping, 0.0, 1.0))
    pre_delay_samples = max(int(sr * max(pre_delay_ms, 0.0) / 1000.0), 0)
    if sr <= 0:
        return base_audio.astype(orig_dtype, copy=True)

    delay_ms = np.array([29.7, 37.1, 41.1, 43.7], dtype=np.float64)
    gain_base = np.array([0.805, 0.773, 0.753, 0.733], dtype=np.float64)
    scaled_delay_samples = np.maximum(
        (delay_ms * (0.72 + 0.65 * room_size) * sr / 1000.0).astype(int),
        1,
    )
    gains = gain_base * (0.52 + 0.36 * room_size)
    tail_seconds = 0.22 + 1.9 * room_size
    ir_len = pre_delay_samples + int(sr * tail_seconds) + 1
    impulse = np.zeros(ir_len, dtype=np.float64)
    impulse[0] = 1.0

    feedback_damping = 1.0 - 0.22 - 0.55 * damping
    feedback_damping = float(np.clip(feedback_damping, 0.15, 0.95))

    for delay_samples, base_gain in zip(scaled_delay_samples, gains):
        tap = pre_delay_samples + delay_samples
        gain = float(base_gain)
        while tap < ir_len and gain > 1e-4:
            impulse[tap] += gain
            tap += delay_samples
            gain *= base_gain * feedback_damping

    impulse /= max(np.max(np.abs(impulse)), 1.0)

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        wet = signal.fftconvolve(channel, impulse, mode="full")[: len(channel)]
        if damping > 0.0:
            nyquist = 0.5 * sr
            cutoff_hz = max(400.0, min(nyquist * (0.92 - 0.68 * damping), nyquist * 0.99))
            wet = butter_filter(wet, sr, btype="lowpass", cutoff=cutoff_hz, order=2)
        return wet.astype(orig_dtype, copy=False)

    reverbed = _apply_channelwise(audio, _process_channel)
    return reverbed.astype(orig_dtype, copy=False)


def limiter(data: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """
    Apply a simple limiter to the audio data.
    
    Parameters:
    - data: NumPy array of audio data (float)
    - threshold: The threshold level for the limiter (linear scale, e.g., 0.99 for -0.1 dB)
    
    Returns:
    - Limited audio data as a NumPy array (float)
    """
    # Calculate the gain reduction factor
    # Support both single and multi-channel (element-wise)
    abs_data = np.abs(data)
    # Avoid division by zero
    abs_data = np.where(abs_data == 0, 1e-10, abs_data)
    
    reduction_factor = np.where(abs_data > threshold, threshold / abs_data, 1.0)
    
    # Apply the gain reduction to the audio signal
    limited_audio = data * reduction_factor
    
    return limited_audio


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
            # Heuristic: if shape[0] < shape[1], assume (Channels, Samples) -> (1, T) or (2, T)
            # Otherwise assume (Samples, Channels) -> (T, 1) or (T, 2)
            if wav_data.shape[0] < wav_data.shape[1]:
                 # (C, T) -> Take first channel
                wav_mono = wav_data[0, :]
            else:
                # (T, C) -> Take first channel
                try:
                    logger.warning("后处理: 输入为多通道音频(T, C)，仅使用第一个声道参与处理并作为输出")
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

    # Use the standalone limiter function
    limited_audio = limiter(wav_norm, threshold)
    return limited_audio, original_loudness


def eq(wav_data: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple EQ: enhance a high-frequency band while ensuring stability across sample rates.
    """
    nyquist = 0.5 * sr
    low_freq = min(5000, nyquist * 0.6)
    high_freq = min(10000, nyquist * 0.95)
    gain_factor = 0.2

    base_audio = _ensure_float_audio(wav_data)
    orig_dtype = base_audio.dtype
    audio = base_audio.astype(np.float64, copy=False)
    enhanced_band = bandpass(audio, sr, low_freq, high_freq, order=4).astype(np.float64, copy=False)
    enhanced_audio = audio + gain_factor * enhanced_band
    enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
    return enhanced_audio.astype(orig_dtype, copy=False)


def trim_silence(
    wav_data: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_duration_ms: int = 200,
    min_segment_ms: int = 50,
    ignore_trailing_gap_ms: int = 300,
    fade_ms: int = 10,
) -> np.ndarray:
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
    min_segment_ms : int
        最小保留片段时长（毫秒），小于此长度的非静音片段会被忽略。
    ignore_trailing_gap_ms : int
        如果尾部片段与前一片段间隔超过此值且自身很短，则将其视为噪音丢弃。
    fade_ms : int
        裁剪后首尾淡入淡出时长（毫秒）。
    """
    # 使用 librosa 来裁剪静音
    # top_db = -threshold_db（librosa 使用相对于峰值的正 dB）
    
    # Convert numpy array to float32 if needed for librosa
    if wav_data.dtype != np.float32 and wav_data.dtype != np.float64:
        wav_data = wav_data.astype(np.float32)

    # top_db：相对于参考电平的静音阈值（单位 dB）
    top_db = -threshold_db
    
    try:
        global _LIBROSA_IMPORT_WARNED
        try:
            import librosa
        except Exception:
            if not _LIBROSA_IMPORT_WARNED:
                _LIBROSA_IMPORT_WARNED = True
                logger.warning("后处理: 未安装 librosa，已跳过 trim_silence")
            return wav_data

        # 转 float，兼容多声道
        if wav_data.ndim > 1:
            # Heuristic: if shape[0] < shape[1], assume (Channels, Samples)
            if wav_data.shape[0] < wav_data.shape[1]:
                wav_mono = np.mean(wav_data, axis=0)
            else:
                wav_mono = np.mean(wav_data, axis=1)
        else:
            wav_mono = wav_data

        non_silent_intervals = librosa.effects.split(wav_mono, top_db=top_db)
        if len(non_silent_intervals) == 0:
            return wav_data

        min_len = int(min_segment_ms / 1000 * sr)
        intervals = [i for i in non_silent_intervals if (i[1] - i[0]) >= min_len]
        if not intervals:
            # 保留最长片段，避免极短语料被全切除
            lengths = [i[1] - i[0] for i in non_silent_intervals]
            intervals = [non_silent_intervals[int(np.argmax(lengths))]]

        # 丢弃尾部与主体间隔过大且很短的片段
        if len(intervals) > 1:
            last_i, prev_i = intervals[-1], intervals[-2]
            gap_ms = (last_i[0] - prev_i[1]) / sr * 1000
            last_ms = (last_i[1] - last_i[0]) / sr * 1000
            if gap_ms > ignore_trailing_gap_ms and last_ms < 200:
                intervals.pop()

        start_idx, end_idx = intervals[0][0], intervals[-1][1]
        pad_samples = int(min_silence_duration_ms * sr / 1000)
        start_idx = max(0, start_idx - pad_samples)
        end_idx = min(len(wav_data), end_idx + pad_samples)

        trimmed = wav_data[start_idx:end_idx]

        # 10ms 淡入淡出，防止切割爆音
        fade_samples = int(fade_ms / 1000 * sr)
        if fade_samples > 0 and len(trimmed) > 2 * fade_samples:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            if trimmed.ndim == 1:
                trimmed[:fade_samples] *= fade_in
                trimmed[-fade_samples:] *= fade_out
            else:
                trimmed[:fade_samples, ...] *= fade_in[:, None]
                trimmed[-fade_samples:, ...] *= fade_out[:, None]

        return trimmed
    except Exception:
        # 如果 librosa 调用失败，则直接返回原始音频作为降级策略
        return wav_data


def apply_postprocess(
    wav: np.ndarray,
    sr: int,
    target_loudness: float = -23.0,
    enable: bool = True,
    trim_silence: bool = False,
    enable_eq: bool = True,
) -> np.ndarray:
    """Apply optional trim + loudnorm + eq with exception safety."""
    if not enable:
        return wav
    try:
        if trim_silence:
            wav = globals()["trim_silence"](wav, sr)
        wav_p, _ = loudnorm(wav, sr, target_loudness=target_loudness)
        if enable_eq:
            wav_p = eq(wav_p, sr)
        return wav_p
    except Exception:
        logger.exception("后处理阶段失败，已跳过后处理并返回原始生成音频")
        return wav
