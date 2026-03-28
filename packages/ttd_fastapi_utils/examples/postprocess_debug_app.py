#!/usr/bin/env python3
import argparse
import json
import importlib
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent / "src"
_LOG_DIR = _THIS_DIR.parent / "logs"
_LOG_PATH = _LOG_DIR / "postprocess_debug_history.jsonl"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

postprocess = importlib.import_module("ttd_fastapi_utils.postprocess")
preset = importlib.import_module("ttd_fastapi_utils.preset")


PRESET_CHOICES = list(preset.list_presets())
PRESET_METADATA = preset.preset_metadata()
FILTER_CHOICES = ["none", "lowpass", "highpass", "bandpass"]
MODE_CHOICES = [("Preset 预设模式", "preset"), ("Custom 自定义模式", "custom")]
PRESET_LABELS = {
    name: PRESET_METADATA[name]["display_name"]
    for name in PRESET_CHOICES
}
FILTER_LABELS = {
    "none": "none 不滤波",
    "lowpass": "lowpass 低通",
    "highpass": "highpass 高通",
    "bandpass": "bandpass 带通",
}

PRESET_PARAM_ORDER = [
    "enable_saturate",
    "enable_delay",
    "enable_reverb",
    "enable_limiter",
    "low_cut_hz",
    "high_cut_hz",
    "lowpass_hz",
    "highpass_hz",
    "drive",
    "saturate_wet",
    "wet_ratio",
    "delay_ms",
    "decay",
    "repeats",
    "delay_wet",
    "reverb_room_size",
    "reverb_damping",
    "reverb_pre_delay_ms",
    "reverb_wet",
    "limiter_threshold",
]

PRESET_FALLBACK_DEFAULTS = {
    "enable_saturate": True,
    "enable_delay": True,
    "enable_reverb": True,
    "enable_limiter": True,
    "low_cut_hz": 300.0,
    "high_cut_hz": 3400.0,
    "lowpass_hz": 2200.0,
    "highpass_hz": 110.0,
    "drive": 1.45,
    "saturate_wet": 0.8,
    "wet_ratio": 0.95,
    "delay_ms": 85.0,
    "decay": 0.45,
    "repeats": 2,
    "delay_wet": 0.28,
    "reverb_room_size": 0.35,
    "reverb_damping": 0.45,
    "reverb_pre_delay_ms": 18.0,
    "reverb_wet": 0.10,
    "limiter_threshold": 0.98,
}

def _preset_help_text(preset_name: str) -> str:
    metadata = PRESET_METADATA.get(preset_name, {})
    lines = [
        metadata.get("display_name", preset_name),
        "- Core character 核心音色: %s" % metadata.get("summary", "n/a"),
    ]
    primary = metadata.get("primary_controls") or []
    secondary = metadata.get("secondary_controls") or []
    if primary:
        lines.append("- Main active controls 主要生效参数: %s" % " / ".join(primary))
    if secondary:
        lines.append("- Secondary controls 次要参数: %s" % " / ".join(secondary))
    note = metadata.get("implementation_note")
    if note:
        lines.append("- Recommended pipeline 推荐链路: %s" % note)
    standard = metadata.get("recommended_standard_postprocess") or {}
    lines.append(
        "- Standard postprocess default 标准后处理默认建议: enable=%s / trim_silence=%s / enable_eq=%s / target_loudness=%s"
        % (
            standard.get("enable", False),
            standard.get("trim_silence", False),
            standard.get("enable_eq", False),
            standard.get("target_loudness", -23.0),
        )
    )
    return "\n".join(lines)


def _append_apply_log(payload: Dict[str, object]) -> str:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {"timestamp": timestamp, **payload}
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return timestamp


def _get_preset_defaults(name: str) -> Dict[str, object]:
    preset_fn = preset.preset_map()[name]
    sig = inspect.signature(preset_fn)
    values = dict(PRESET_FALLBACK_DEFAULTS)
    for key, param in sig.parameters.items():
        if key in {"wav_data", "sr"}:
            continue
        if param.default is not inspect._empty:
            values[key] = param.default
    return values


def sync_preset_controls(preset_name: str):
    values = _get_preset_defaults(preset_name)
    metadata = PRESET_METADATA.get(preset_name, {})
    standard = metadata.get("recommended_standard_postprocess") or {
        "enable": False,
        "target_loudness": -23.0,
        "trim_silence": False,
        "enable_eq": True,
    }
    return (
        gr.update(value=standard["enable"]),
        gr.update(value=standard["target_loudness"]),
        gr.update(value=standard["trim_silence"]),
        gr.update(value=standard["enable_eq"]),
        *(gr.update(value=values[key]) for key in PRESET_PARAM_ORDER),
    )


def _apply_standard_postprocess(
    wav: np.ndarray,
    sr: int,
    *,
    enable: bool,
    target_loudness: float,
    trim_silence: bool,
    enable_eq: bool,
) -> np.ndarray:
    if not enable:
        return wav
    return postprocess.apply_postprocess(
        wav,
        sr,
        target_loudness=target_loudness,
        enable=True,
        trim_silence=trim_silence,
        enable_eq=enable_eq,
    )


def _normalize_audio_dtype(audio: np.ndarray) -> np.ndarray:
    if audio is None:
        return audio
    if np.issubdtype(audio.dtype, np.integer):
        max_value = float(np.iinfo(audio.dtype).max)
        if max_value > 0:
            audio = audio.astype(np.float32) / max_value
        else:
            audio = audio.astype(np.float32)
    elif not np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
    return audio


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio is None or audio.ndim <= 1:
        return audio
    if audio.shape[0] < audio.shape[1]:
        return np.mean(audio, axis=0)
    return np.mean(audio, axis=1)


def _call_preset(name: str, wav: np.ndarray, sr: int, params: Dict[str, float]) -> np.ndarray:
    preset_fn = preset.preset_map()[name]
    sig = inspect.signature(preset_fn)
    accepted = {}
    for key, value in params.items():
        if key in sig.parameters:
            accepted[key] = value
    return preset_fn(wav, sr, **accepted)


def _apply_custom_chain(
    wav: np.ndarray,
    sr: int,
    *,
    use_standard_chain: bool,
    target_loudness: float,
    trim_silence: bool,
    enable_eq: bool,
    filter_type: str,
    low_cut_hz: float,
    high_cut_hz: float,
    lowpass_hz: float,
    highpass_hz: float,
    enable_saturate: bool,
    drive: float,
    saturate_wet: float,
    wet_ratio: float,
    enable_delay: bool,
    delay_ms: float,
    decay: float,
    repeats: int,
    delay_wet: float,
    enable_reverb: bool,
    reverb_room_size: float,
    reverb_damping: float,
    reverb_pre_delay_ms: float,
    reverb_wet: float,
    enable_limiter: bool,
    limiter_threshold: float,
) -> np.ndarray:
    dry = wav
    out = wav
    if use_standard_chain:
        out = postprocess.apply_postprocess(
            out,
            sr,
            target_loudness=target_loudness,
            enable=True,
            trim_silence=trim_silence,
            enable_eq=enable_eq,
        )

    if filter_type == "lowpass":
        out = postprocess.lowpass(out, sr, cutoff_hz=lowpass_hz)
    elif filter_type == "highpass":
        out = postprocess.highpass(out, sr, cutoff_hz=highpass_hz)
    elif filter_type == "bandpass":
        out = postprocess.bandpass(out, sr, low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz)

    if enable_saturate:
        saturated = postprocess.saturate(out, drive=drive)
        out = postprocess.mix(out, saturated, wet_ratio=saturate_wet)

    if enable_delay:
        delayed = postprocess.delay(out, sr, delay_ms=delay_ms, decay=decay, repeats=repeats)
        out = postprocess.mix(out, delayed, wet_ratio=delay_wet)

    if enable_reverb:
        reverbed = postprocess.reverb(
            out,
            sr,
            room_size=reverb_room_size,
            damping=reverb_damping,
            pre_delay_ms=reverb_pre_delay_ms,
        )
        out = postprocess.mix(out, reverbed, wet_ratio=reverb_wet)

    out = postprocess.mix(dry, out, wet_ratio=wet_ratio)

    if enable_limiter:
        out = postprocess.limiter(out, threshold=limiter_threshold)

    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def process_audio(
    input_audio: Tuple[int, np.ndarray],
    use_mono: bool,
    mode: str,
    preset_name: str,
    use_standard_chain: bool,
    target_loudness: float,
    trim_silence: bool,
    enable_eq: bool,
    filter_type: str,
    low_cut_hz: float,
    high_cut_hz: float,
    lowpass_hz: float,
    highpass_hz: float,
    enable_saturate: bool,
    drive: float,
    saturate_wet: float,
    wet_ratio: float,
    enable_delay: bool,
    delay_ms: float,
    decay: float,
    repeats: int,
    delay_wet: float,
    enable_reverb: bool,
    reverb_room_size: float,
    reverb_damping: float,
    reverb_pre_delay_ms: float,
    reverb_wet: float,
    enable_limiter: bool,
    limiter_threshold: float,
) -> Tuple[Tuple[int, np.ndarray], str]:
    if input_audio is None:
        raise gr.Error("请先上传音频文件。")

    sr, wav = input_audio
    wav = _normalize_audio_dtype(np.asarray(wav))
    if use_mono:
        wav = _to_mono(wav)
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32, copy=False)

    if mode == "preset":
        standard_chain_params = {
            "enable": use_standard_chain,
            "target_loudness": target_loudness,
            "trim_silence": trim_silence,
            "enable_eq": enable_eq,
        }
        ui_preset_params = {
            "enable_saturate": enable_saturate,
            "enable_delay": enable_delay,
            "enable_reverb": enable_reverb,
            "enable_limiter": enable_limiter,
            "low_cut_hz": low_cut_hz,
            "high_cut_hz": high_cut_hz,
            "lowpass_hz": lowpass_hz,
            "highpass_hz": highpass_hz,
            "wet_ratio": wet_ratio,
            "drive": drive,
            "saturate_wet": saturate_wet,
            "delay_ms": delay_ms,
            "decay": decay,
            "repeats": repeats,
            "delay_wet": delay_wet,
            "reverb_room_size": reverb_room_size,
            "reverb_damping": reverb_damping,
            "reverb_pre_delay_ms": reverb_pre_delay_ms,
            "reverb_wet": reverb_wet,
            "limiter_threshold": limiter_threshold,
        }
        preset_params = _get_preset_defaults(preset_name)
        preset_params.update(ui_preset_params)
        wav_for_preset = _apply_standard_postprocess(
            wav,
            sr,
            enable=use_standard_chain,
            target_loudness=target_loudness,
            trim_silence=trim_silence,
            enable_eq=enable_eq,
        )
        out = _call_preset(preset_name, wav_for_preset, sr, preset_params)
        log_timestamp = _append_apply_log(
            {
                "mode": mode,
                "preset": preset_name,
                "sample_rate": int(sr),
                "mono": bool(use_mono),
                "input_samples": int(wav.shape[0]) if wav.ndim == 1 else list(wav.shape),
                "output_peak": float(np.max(np.abs(out))) if out.size else 0.0,
                "standard_chain": standard_chain_params,
                "preset_style_params": preset_params,
            }
        )
        summary = [
            "Mode 模式: preset 预设模式",
            "Preset 预设: %s" % PRESET_LABELS.get(preset_name, preset_name),
            "Sample Rate 采样率: %s" % sr,
            "Mono 单声道: %s" % use_mono,
            "Standard Chain 标准链: %s" % use_standard_chain,
            "Peak 峰值: %.4f" % float(np.max(np.abs(out))) if out.size else "Peak 峰值: 0.0000",
            "Log 记录时间: %s" % log_timestamp,
            "Log File 日志文件: %s" % _LOG_PATH,
            "",
            _preset_help_text(preset_name),
        ]
    else:
        out = _apply_custom_chain(
            wav,
            sr,
            use_standard_chain=use_standard_chain,
            target_loudness=target_loudness,
            trim_silence=trim_silence,
            enable_eq=enable_eq,
            filter_type=filter_type,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
            lowpass_hz=lowpass_hz,
            highpass_hz=highpass_hz,
            enable_saturate=enable_saturate,
            drive=drive,
            saturate_wet=saturate_wet,
            wet_ratio=wet_ratio,
            enable_delay=enable_delay,
            delay_ms=delay_ms,
            decay=decay,
            repeats=repeats,
            delay_wet=delay_wet,
            enable_reverb=enable_reverb,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_pre_delay_ms=reverb_pre_delay_ms,
            reverb_wet=reverb_wet,
            enable_limiter=enable_limiter,
            limiter_threshold=limiter_threshold,
        )
        log_timestamp = _append_apply_log(
            {
                "mode": mode,
                "preset": None,
                "sample_rate": int(sr),
                "mono": bool(use_mono),
                "input_samples": int(wav.shape[0]) if wav.ndim == 1 else list(wav.shape),
                "output_peak": float(np.max(np.abs(out))) if out.size else 0.0,
                "custom_params": {
                    "use_standard_chain": use_standard_chain,
                    "target_loudness": target_loudness,
                    "trim_silence": trim_silence,
                    "enable_eq": enable_eq,
                    "filter_type": filter_type,
                    "low_cut_hz": low_cut_hz,
                    "high_cut_hz": high_cut_hz,
                    "lowpass_hz": lowpass_hz,
                    "highpass_hz": highpass_hz,
                    "enable_saturate": enable_saturate,
                    "drive": drive,
                    "saturate_wet": saturate_wet,
                    "wet_ratio": wet_ratio,
                    "enable_delay": enable_delay,
                    "delay_ms": delay_ms,
                    "decay": decay,
                    "repeats": repeats,
                    "delay_wet": delay_wet,
                    "enable_reverb": enable_reverb,
                    "reverb_room_size": reverb_room_size,
                    "reverb_damping": reverb_damping,
                    "reverb_pre_delay_ms": reverb_pre_delay_ms,
                    "reverb_wet": reverb_wet,
                    "enable_limiter": enable_limiter,
                    "limiter_threshold": limiter_threshold,
                },
            }
        )
        summary = [
            "Mode 模式: custom 自定义模式",
            "Sample Rate 采样率: %s" % sr,
            "Mono 单声道: %s" % use_mono,
            "Standard Chain 标准链: %s" % use_standard_chain,
            "Filter 滤波: %s" % FILTER_LABELS.get(filter_type, filter_type),
            "Saturation 饱和: %s" % enable_saturate,
            "Delay 延迟: %s" % enable_delay,
            "Reverb 混响: %s" % enable_reverb,
            "Limiter 限幅: %s" % enable_limiter,
            "Peak 峰值: %.4f" % float(np.max(np.abs(out))) if out.size else "Peak 峰值: 0.0000",
            "Log 记录时间: %s" % log_timestamp,
            "Log File 日志文件: %s" % _LOG_PATH,
        ]

    return (sr, out), "\n".join(summary)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="TTD Postprocess Preset Debugger", analytics_enabled=False) as demo:
        gr.Markdown(
            """
            # TTD Postprocess Preset Debugger / TTD 后处理预设调试台

            上传一段音频后，可以选择两套独立流程：先做标准后处理，再叠加风格 preset；或者切到 custom 模式手工组合整条链。  
            Upload an audio clip, then use two independent stages: standard postprocess first, then style presets; or switch to custom mode to build the whole chain manually.

            **使用建议 / Tips**
            - `Preset` 模式现在分成两段：`Standard Postprocess` 和 `Style Preset`。  
              `Preset` mode now has two stages: `Standard Postprocess` and `Style Preset`.
            - `Custom` 模式适合实验底层积木；`Preset` 模式适合业务试听和定稿。  
              `Custom` mode is for low-level experiments; `Preset` mode is for business review and preset tuning.
            - `Filter Type / 滤波类型` 只在 `Custom` 模式下生效；`Preset` 模式由具体 preset 决定用哪种滤波。  
              `Filter Type` only applies in `Custom` mode; presets decide their own filtering.
            - 现在已经支持 `Reverb / 混响`，用于补空间感；`Delay` 更偏回声，`Reverb` 更偏房间尾音。  
              `Reverb` is now available for ambience; `Delay` behaves more like echo, while `Reverb` feels more like room tail.
            - 页面下方的 `处理摘要 / Processing Summary` 会告诉你当前 preset 主要响应哪些参数。  
              The summary panel tells you which controls matter most for the selected preset.
            """
        )

        with gr.Row():
            input_audio = gr.Audio(label="Input Audio / 输入音频", type="numpy", sources=["upload"])
            output_audio = gr.Audio(
                label="Output Audio / 处理后音频",
                type="numpy",
                autoplay=True,
            )

        summary = gr.Textbox(label="Processing Summary / 处理摘要", lines=14)

        with gr.Row():
            use_mono = gr.Checkbox(
                label="Convert to Mono Before Processing / 先转单声道再处理",
                value=True,
                info="某些素材是双声道，先转单声道更方便比较音色变化。 Convert stereo input to mono first for easier A/B listening.",
            )
            mode = gr.Radio(
                label="Mode / 处理模式",
                choices=MODE_CHOICES,
                value="preset",
            )
            preset_name = gr.Dropdown(
                label="Preset / 预设",
                choices=[(PRESET_LABELS[name], name) for name in PRESET_CHOICES],
                value="smart_assistant",
            )
            filter_type = gr.Dropdown(
                label="Custom Filter Type / 自定义滤波类型",
                choices=[(FILTER_LABELS[name], name) for name in FILTER_CHOICES],
                value="bandpass",
                info="Only used in Custom mode / 仅在 Custom 模式下生效",
            )

        with gr.Accordion("Standard Postprocess / 标准后处理", open=False):
            with gr.Row():
                use_standard_chain = gr.Checkbox(label="Enable Standard Postprocess / 启用标准后处理", value=True)
                trim_silence = gr.Checkbox(label="Trim Leading/Trailing Silence / 裁剪首尾静音", value=False)
                enable_eq = gr.Checkbox(label="Enable Default EQ / 启用默认 EQ", value=True)
            target_loudness = gr.Slider(label="Target LUFS / 目标 LUFS", minimum=-32.0, maximum=-12.0, value=-23.0, step=0.5)

        with gr.Accordion("Style Preset Controls / 预设风格参数", open=True):
            with gr.Row():
                low_cut_hz = gr.Slider(label="Low Cut Hz / 低切频率", minimum=20.0, maximum=4000.0, value=170.0, step=10.0)
                high_cut_hz = gr.Slider(label="High Cut Hz / 高切频率", minimum=500.0, maximum=12000.0, value=4250.0, step=50.0)
            with gr.Row():
                lowpass_hz = gr.Slider(label="Lowpass Hz / 低通频率", minimum=200.0, maximum=12000.0, value=950.0, step=50.0)
                highpass_hz = gr.Slider(label="Highpass Hz / 高通频率", minimum=20.0, maximum=4000.0, value=1740.0, step=10.0)

            with gr.Row():
                enable_saturate = gr.Checkbox(label="Enable Saturation / 启用饱和", value=True)
                drive = gr.Slider(label="Drive / 饱和驱动", minimum=1.0, maximum=3.0, value=1.45, step=0.05)
                saturate_wet = gr.Slider(label="Saturation Wet / 饱和 Wet", minimum=0.0, maximum=1.0, value=0.06, step=0.01)
                wet_ratio = gr.Slider(label="Overall Wet / 整体 Wet", minimum=0.0, maximum=1.0, value=0.95, step=0.01)

            with gr.Row():
                enable_delay = gr.Checkbox(label="Enable Delay / 启用延迟", value=True)
                delay_ms = gr.Slider(label="Delay ms / 延迟毫秒", minimum=10.0, maximum=200.0, value=41.0, step=1.0)
                decay = gr.Slider(label="Decay / 衰减", minimum=0.0, maximum=0.9, value=0.53, step=0.01)
                repeats = gr.Slider(label="Repeats / 重复次数", minimum=0, maximum=6, value=2, step=1)
                delay_wet = gr.Slider(label="Delay Wet / 延迟 Wet", minimum=0.0, maximum=1.0, value=0.87, step=0.01)

            with gr.Row():
                enable_reverb = gr.Checkbox(label="Enable Reverb / 启用混响", value=False)
                reverb_room_size = gr.Slider(label="Reverb Room Size / 混响空间大小", minimum=0.0, maximum=1.0, value=0.32, step=0.01)
                reverb_damping = gr.Slider(label="Reverb Damping / 混响阻尼", minimum=0.0, maximum=1.0, value=0.42, step=0.01)
                reverb_pre_delay_ms = gr.Slider(label="Reverb Pre-delay ms / 混响预延迟毫秒", minimum=0.0, maximum=120.0, value=16.0, step=1.0)
                reverb_wet = gr.Slider(label="Reverb Wet / 混响 Wet", minimum=0.0, maximum=1.0, value=0.0, step=0.01)

            with gr.Row():
                enable_limiter = gr.Checkbox(label="Enable Limiter / 启用限幅", value=True)
                limiter_threshold = gr.Slider(label="Limiter Threshold / 限幅阈值", minimum=0.80, maximum=1.0, value=0.98, step=0.005)

        run_button = gr.Button("Apply Effect / 应用效果", variant="primary")

        preset_name.change(
            fn=sync_preset_controls,
            inputs=[preset_name],
            outputs=[
                use_standard_chain,
                target_loudness,
                trim_silence,
                enable_eq,
                enable_saturate,
                enable_delay,
                enable_reverb,
                enable_limiter,
                low_cut_hz,
                high_cut_hz,
                lowpass_hz,
                highpass_hz,
                drive,
                saturate_wet,
                wet_ratio,
                delay_ms,
                decay,
                repeats,
                delay_wet,
                reverb_room_size,
                reverb_damping,
                reverb_pre_delay_ms,
                reverb_wet,
                limiter_threshold,
            ],
        )

        run_button.click(
            fn=process_audio,
            inputs=[
                input_audio,
                use_mono,
                mode,
                preset_name,
                use_standard_chain,
                target_loudness,
                trim_silence,
                enable_eq,
                filter_type,
                low_cut_hz,
                high_cut_hz,
                lowpass_hz,
                highpass_hz,
                enable_saturate,
                drive,
                saturate_wet,
                wet_ratio,
                enable_delay,
                delay_ms,
                decay,
                repeats,
                delay_wet,
                enable_reverb,
                reverb_room_size,
                reverb_damping,
                reverb_pre_delay_ms,
                reverb_wet,
                enable_limiter,
                limiter_threshold,
            ],
            outputs=[output_audio, summary],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="TTD postprocess/preset debug Gradio app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
