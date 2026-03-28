from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from . import postprocess

_CANONICAL_PRESETS = (
    "telephone",
    "smart_assistant",
    "inner_monologue",
    "radio",
    "intercom",
)


def _ensure_float_audio(wav_data: np.ndarray) -> np.ndarray:
    if isinstance(wav_data, list):
        wav_data = np.asarray(wav_data, dtype=np.float32)
    elif not isinstance(wav_data, np.ndarray):
        wav_data = np.asarray(wav_data)
    if not np.issubdtype(wav_data.dtype, np.floating):
        wav_data = wav_data.astype(np.float32)
    return wav_data


def _apply_reverb(
    wav_data: np.ndarray,
    sr: int,
    *,
    reverb_room_size: float,
    reverb_damping: float,
    reverb_pre_delay_ms: float,
    reverb_wet: float,
) -> np.ndarray:
    if reverb_wet <= 0.0:
        return wav_data
    reverbed = postprocess.reverb(
        wav_data,
        sr,
        room_size=reverb_room_size,
        damping=reverb_damping,
        pre_delay_ms=reverb_pre_delay_ms,
    )
    return postprocess.mix(wav_data, reverbed, wet_ratio=reverb_wet)


def telephone(
    wav_data: np.ndarray,
    sr: int,
    enable_saturate: bool = True,
    enable_reverb: bool = False,
    enable_limiter: bool = True,
    low_cut_hz: float = 300.0,
    high_cut_hz: float = 3400.0,
    wet_ratio: float = 0.95,
    drive: float = 1.45,
    reverb_room_size: float = 0.18,
    reverb_damping: float = 0.65,
    reverb_pre_delay_ms: float = 12.0,
    reverb_wet: float = 0.0,
    limiter_threshold: float = 0.98,
) -> np.ndarray:
    """电话音：窄带 + 轻饱和。"""
    audio = _ensure_float_audio(wav_data)
    out = audio
    filtered = postprocess.bandpass(out, sr, low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz)
    colored = filtered
    if enable_saturate:
        colored = postprocess.saturate(filtered * 1.10, drive=drive)
    mixed = postprocess.mix(out, colored, wet_ratio=wet_ratio)
    if enable_reverb:
        mixed = _apply_reverb(
            mixed,
            sr,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_pre_delay_ms=reverb_pre_delay_ms,
            reverb_wet=reverb_wet,
        )
    if enable_limiter:
        return postprocess.limiter(mixed, threshold=limiter_threshold)
    return mixed


def smart_assistant(
    wav_data: np.ndarray,
    sr: int,
    enable_saturate: bool = True,
    enable_delay: bool = True,
    enable_reverb: bool = False,
    enable_limiter: bool = True,
    low_cut_hz: float = 170.0,
    high_cut_hz: float = 4250.0,
    drive: float = 1.45,
    saturate_wet: float = 0.06,
    wet_ratio: float = 0.95,
    delay_ms: float = 41.0,
    decay: float = 0.53,
    repeats: int = 2,
    delay_wet: float = 0.70,
    reverb_room_size: float = 0.32,
    reverb_damping: float = 0.42,
    reverb_pre_delay_ms: float = 16.0,
    reverb_wet: float = 0.0,
    limiter_threshold: float = 0.98,
) -> np.ndarray:
    """模拟智能语音：带通 + 轻饱和 + 空间尾音。"""
    audio = _ensure_float_audio(wav_data)
    base = audio

    out = postprocess.bandpass(base, sr, low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz)

    if enable_saturate:
        saturated = postprocess.saturate(out, drive=drive)
        out = postprocess.mix(out, saturated, wet_ratio=saturate_wet)

    if enable_delay:
        delayed = postprocess.delay(out, sr, delay_ms=delay_ms, decay=decay, repeats=repeats)
        out = postprocess.mix(out, delayed, wet_ratio=delay_wet)
    if enable_reverb:
        out = _apply_reverb(
            out,
            sr,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_pre_delay_ms=reverb_pre_delay_ms,
            reverb_wet=reverb_wet,
        )
    out = postprocess.mix(base, out, wet_ratio=wet_ratio)

    if enable_limiter:
        return postprocess.limiter(out, threshold=limiter_threshold)
    return out


def inner_monologue(
    wav_data: np.ndarray,
    sr: int,
    enable_delay: bool = True,
    enable_reverb: bool = True,
    enable_limiter: bool = True,
    lowpass_hz: float = 2200.0,
    delay_ms: float = 85.0,
    decay: float = 0.45,
    repeats: int = 2,
    wet_ratio: float = 0.28,
    reverb_room_size: float = 1.0,
    reverb_damping: float = 0.0,
    reverb_pre_delay_ms: float = 120.0,
    reverb_wet: float = 0.30,
    limiter_threshold: float = 0.98,
) -> np.ndarray:
    """心声独白：柔和低通 + 短回声，带一点空气感。"""
    audio = _ensure_float_audio(wav_data)
    out = audio
    softened = postprocess.lowpass(out, sr, cutoff_hz=lowpass_hz, order=3)
    airy = softened - postprocess.lowpass(
        softened,
        sr,
        cutoff_hz=min(lowpass_hz * 0.55, 0.45 * sr),
        order=2,
    )
    wet = softened + 0.12 * airy
    if enable_delay:
        delayed = postprocess.delay(softened, sr, delay_ms=delay_ms, decay=decay, repeats=repeats)
        wet = delayed + 0.12 * airy
    if enable_reverb:
        wet = _apply_reverb(
            wet,
            sr,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_pre_delay_ms=reverb_pre_delay_ms,
            reverb_wet=reverb_wet,
        )
    mixed = postprocess.mix(out, wet, wet_ratio=wet_ratio)
    if enable_limiter:
        return postprocess.limiter(mixed, threshold=limiter_threshold)
    return mixed


def radio(
    wav_data: np.ndarray,
    sr: int,
    enable_saturate: bool = True,
    enable_delay: bool = True,
    enable_reverb: bool = True,
    enable_limiter: bool = True,
    low_cut_hz: float = 280.0,
    high_cut_hz: float = 1800.0,
    drive: float = 2.8,
    delay_ms: float = 84.0,
    decay: float = 0.5,
    repeats: int = 0,
    wet_ratio: float = 0.93,
    reverb_room_size: float = 0.28,
    reverb_damping: float = 0.58,
    reverb_pre_delay_ms: float = 14.0,
    reverb_wet: float = 0.08,
    limiter_threshold: float = 0.9,
) -> np.ndarray:
    """收音机/广播：中频突出，带一点箱体和空间感。"""
    audio = _ensure_float_audio(wav_data)
    out = audio
    filtered = postprocess.bandpass(out, sr, low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz)
    colored = filtered
    if enable_saturate:
        colored = postprocess.saturate(filtered, drive=drive)
    roomy = colored
    if enable_delay:
        roomy = postprocess.delay(colored, sr, delay_ms=delay_ms, decay=decay, repeats=repeats)
    wet = 0.85 * colored + 0.15 * roomy
    if enable_reverb:
        wet = _apply_reverb(
            wet,
            sr,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_pre_delay_ms=reverb_pre_delay_ms,
            reverb_wet=reverb_wet,
        )
    mixed = postprocess.mix(out, wet, wet_ratio=wet_ratio)
    if enable_limiter:
        return postprocess.limiter(mixed, threshold=limiter_threshold)
    return mixed


def intercom(
    wav_data: np.ndarray,
    sr: int,
    enable_saturate: bool = True,
    enable_reverb: bool = False,
    enable_limiter: bool = True,
    low_cut_hz: float = 450.0,
    high_cut_hz: float = 2800.0,
    drive: float = 1.75,
    wet_ratio: float = 1.0,
    reverb_room_size: float = 0.16,
    reverb_damping: float = 0.72,
    reverb_pre_delay_ms: float = 10.0,
    reverb_wet: float = 0.0,
    limiter_threshold: float = 0.97,
) -> np.ndarray:
    """对讲机：更窄、更硬的中频质感。"""
    audio = _ensure_float_audio(wav_data)
    out = audio
    filtered = postprocess.bandpass(out, sr, low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz)
    colored = filtered
    if enable_saturate:
        colored = postprocess.saturate(filtered, drive=drive)
    mixed = postprocess.mix(out, colored, wet_ratio=wet_ratio)
    if enable_reverb:
        mixed = _apply_reverb(
            mixed,
            sr,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_pre_delay_ms=reverb_pre_delay_ms,
            reverb_wet=reverb_wet,
        )
    if enable_limiter:
        return postprocess.limiter(mixed, threshold=limiter_threshold)
    return mixed


_PRESET_METADATA: Dict[str, Dict[str, Any]] = {
    "telephone": {
        "display_name": "telephone 电话音",
        "summary": "bandpass + saturation",
        "primary_controls": ["low_cut_hz", "high_cut_hz", "wet_ratio", "drive", "limiter_threshold"],
        "secondary_controls": ["reverb_room_size", "reverb_damping", "reverb_pre_delay_ms", "reverb_wet"],
        "recommended_standard_postprocess": {
            "enable": False,
            "target_loudness": -23.0,
            "trim_silence": False,
            "enable_eq": True,
        },
    },
    "smart_assistant": {
        "display_name": "smart_assistant 模拟智能语音",
        "summary": "bandpass + light saturation + delay tail",
        "primary_controls": [
            "low_cut_hz",
            "high_cut_hz",
            "wet_ratio",
            "drive",
            "saturate_wet",
            "delay_ms",
            "decay",
            "repeats",
            "delay_wet",
            "reverb_wet",
            "limiter_threshold",
        ],
        "secondary_controls": ["reverb_room_size", "reverb_damping", "reverb_pre_delay_ms"],
        "recommended_standard_postprocess": {
            "enable": False,
            "target_loudness": -23.0,
            "trim_silence": False,
            "enable_eq": True,
        },
        "implementation_note": "Business-side standard postprocess should be applied explicitly before this preset when needed.",
    },
    "inner_monologue": {
        "display_name": "inner_monologue 心声独白",
        "summary": "lowpass + short echo + soft reverb",
        "primary_controls": ["lowpass_hz", "delay_ms", "decay", "repeats", "wet_ratio", "reverb_wet", "limiter_threshold"],
        "secondary_controls": ["reverb_room_size", "reverb_damping", "reverb_pre_delay_ms"],
        "recommended_standard_postprocess": {
            "enable": False,
            "target_loudness": -23.0,
            "trim_silence": False,
            "enable_eq": False,
        },
    },
    "radio": {
        "display_name": "radio 广播/收音机",
        "summary": "mid-focused bandpass + saturation + tiny room tail",
        "primary_controls": ["low_cut_hz", "high_cut_hz", "drive", "delay_ms", "decay", "repeats", "wet_ratio", "reverb_wet", "limiter_threshold"],
        "secondary_controls": ["reverb_room_size", "reverb_damping", "reverb_pre_delay_ms"],
        "recommended_standard_postprocess": {
            "enable": False,
            "target_loudness": -23.0,
            "trim_silence": False,
            "enable_eq": False,
        },
    },
    "intercom": {
        "display_name": "intercom 对讲机",
        "summary": "narrow bandpass + harder saturation",
        "primary_controls": ["low_cut_hz", "high_cut_hz", "wet_ratio", "drive", "limiter_threshold"],
        "secondary_controls": ["reverb_room_size", "reverb_damping", "reverb_pre_delay_ms", "reverb_wet"],
        "recommended_standard_postprocess": {
            "enable": False,
            "target_loudness": -23.0,
            "trim_silence": False,
            "enable_eq": False,
        },
    },
}


_PRESETS = {
    "telephone": telephone,
    "phone": telephone,
    "电话": telephone,
    "smart_assistant": smart_assistant,
    "assistant": smart_assistant,
    "智能语音": smart_assistant,
    "inner_monologue": inner_monologue,
    "inner_voice": inner_monologue,
    "心声独白": inner_monologue,
    "心声": inner_monologue,
    "radio": radio,
    "广播": radio,
    "intercom": intercom,
    "对讲机": intercom,
}


def list_presets() -> Iterable[str]:
    """返回可用 preset 名称（去重后的 canonical names）。"""
    return _CANONICAL_PRESETS


def apply_preset(name: str, wav_data: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """按名称应用 preset。"""
    key = str(name).lower()
    if key not in _PRESETS:
        raise ValueError("unsupported preset: %s" % name)
    preset_fn = _PRESETS[key]
    return preset_fn(wav_data, sr, **kwargs)


def preset_map() -> Dict[str, Callable[..., np.ndarray]]:
    """返回 preset 注册表副本，便于业务层查看或扩展。"""
    return dict(_PRESETS)


def preset_metadata(name: Optional[str] = None) -> Dict[str, Any]:
    """返回 preset 元数据；传 name 时返回单个 preset 的元数据副本。"""
    if name is None:
        return deepcopy(_PRESET_METADATA)
    key = str(name).lower()
    if key not in _PRESETS:
        raise ValueError("unsupported preset: %s" % name)
    canonical = next(
        canonical_name for canonical_name in _CANONICAL_PRESETS if _PRESETS[canonical_name] is _PRESETS[key]
    )
    return deepcopy(_PRESET_METADATA[canonical])
