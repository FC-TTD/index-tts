import atexit
import argparse
import html
import json
import os
import sys
import threading
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
indextts_dir = os.path.join(current_dir, "indextts")
if indextts_dir not in sys.path:
    sys.path.append(indextts_dir)

import gradio as gr

from download_filename import build_download_filename
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

try:
    from ttd_fastapi_utils import SmartModel
except ImportError:
    from packages.ttd_fastapi_utils.src.ttd_fastapi_utils import SmartModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndexTTS WebUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose mode"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the web UI on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the web UI on"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./checkpoints",
        help="Model checkpoints directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 for inference if available",
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        default=False,
        help="Use DeepSpeed to accelerate if available",
    )
    parser.add_argument(
        "--cuda_kernel",
        action="store_true",
        default=False,
        help="Use CUDA kernel for inference if available",
    )
    parser.add_argument(
        "--gui_seg_tokens",
        type=int,
        default=120,
        help="GUI: Max tokens per generation segment",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_model_dir(model_dir: str) -> None:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory {model_dir} does not exist. Please download the model first."
        )

    for file_name in [
        "bpe.model",
        "gpt.pth",
        "config.yaml",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
    ]:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file {file_path} does not exist. Please download it."
            )


i18n = I18nAuto(language="zh_CN")
MODE = "local"
LANGUAGES = {"中文": "zh_CN", "English": "en_US"}
EMO_CHOICES_ALL = [
    i18n("与音色参考音频相同"),
    i18n("使用情感参考音频"),
    i18n("使用情感向量控制"),
    i18n("使用情感描述文本控制"),
]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]


class NamedAudioInput:
    def __init__(self, path: str | None, orig_name: str | None = None):
        self.path = path
        self.orig_name = orig_name


def NamedAudio(*args, **kwargs):
    component = gr.Audio(*args, **kwargs)
    base_preprocess = component.preprocess

    def preprocess(payload):
        path = base_preprocess(payload)
        orig_name = getattr(payload, "orig_name", None) if payload is not None else None
        return NamedAudioInput(path=path, orig_name=orig_name)

    component.preprocess = preprocess
    return component


def _audio_path(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return getattr(value, "path", None) or getattr(value, "name", None)


def _audio_orig_name(value):
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, dict):
        return value.get("orig_name")
    return getattr(value, "orig_name", None)


def _default_examples() -> list[list[object]]:
    examples_path = os.path.join(current_dir, "examples", "cases.jsonl")
    if not os.path.exists(examples_path):
        return []

    cases = []
    with open(examples_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            emo_audio_path = (
                os.path.join("examples", example["emo_audio"])
                if example.get("emo_audio")
                else None
            )
            cases.append(
                [
                    os.path.join(
                        "examples", example.get("prompt_audio", "sample_prompt.wav")
                    ),
                    EMO_CHOICES_ALL[example.get("emo_mode", 0)],
                    example.get("text"),
                    emo_audio_path,
                    example.get("emo_weight", 1.0),
                    example.get("emo_text", ""),
                    example.get("emo_vec_1", 0),
                    example.get("emo_vec_2", 0),
                    example.get("emo_vec_3", 0),
                    example.get("emo_vec_4", 0),
                    example.get("emo_vec_5", 0),
                    example.get("emo_vec_6", 0),
                    example.get("emo_vec_7", 0),
                    example.get("emo_vec_8", 0),
                ]
            )
    return cases


def get_example_cases(
    example_cases: list[list[object]], include_experimental: bool = False
) -> list[list[object]]:
    if include_experimental:
        return example_cases
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]


def build_demo(
    args: argparse.Namespace,
    tts_manager: SmartModel | None = None,
    tts_manager_getter=None,
) -> gr.Blocks:
    validate_model_dir(args.model_dir)
    gui_seg_tokens = int(getattr(args, "gui_seg_tokens", 120))
    verbose = bool(getattr(args, "verbose", False))
    deepspeed = bool(getattr(args, "deepspeed", getattr(args, "use_deepspeed", False)))
    cuda_kernel = bool(
        getattr(args, "cuda_kernel", getattr(args, "use_cuda_kernel", False))
    )

    created_manager = False
    allow_lazy_boot = tts_manager is None and tts_manager_getter is not None
    if tts_manager is None and tts_manager_getter is None:
        created_manager = True

        def tts_loader():
            return IndexTTS2(
                model_dir=args.model_dir,
                cfg_path=os.path.join(args.model_dir, "config.yaml"),
                use_fp16=args.fp16,
                use_deepspeed=deepspeed,
                use_cuda_kernel=cuda_kernel,
            )

        tts_manager = SmartModel(tts_loader, timeout_seconds=7200)

    if tts_manager_getter is None:
        tts_manager_getter = lambda: tts_manager

    def get_tts():
        manager = tts_manager_getter()
        if manager is None:
            raise RuntimeError("TTS manager is not ready yet")
        return manager.get()

    model_version = "2.0"
    glossary_enabled = False
    max_mel_tokens_limit = 1500
    max_text_tokens_limit = max(120, gui_seg_tokens)
    if not allow_lazy_boot:
        boot_tts = get_tts()
        model_version = boot_tts.model_version or "1.0"
        glossary_enabled = bool(boot_tts.normalizer.enable_glossary)
        max_mel_tokens_limit = int(boot_tts.cfg.gpt.max_mel_tokens)
        max_text_tokens_limit = int(boot_tts.cfg.gpt.max_text_tokens)
        del boot_tts
        if created_manager:
            tts_manager.unload()

    def cleanup_manager():
        if created_manager:
            try:
                tts_manager.unload()
            finally:
                tts_manager.stop()

    if created_manager:
        atexit.register(cleanup_manager)

    os.makedirs("outputs/tasks", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)
    example_cases = _default_examples()

    def format_glossary_markdown():
        tts = get_tts()
        if not tts.normalizer.term_glossary:
            return i18n("暂无术语")

        lines = [
            f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |",
            "|---|---|---|",
        ]
        for term, reading in tts.normalizer.term_glossary.items():
            zh = reading.get("zh", "") if isinstance(reading, dict) else reading
            en = reading.get("en", "") if isinstance(reading, dict) else reading
            lines.append(f"| {term} | {zh} | {en} |")
        return "\n".join(lines)

    def update_prompt_audio():
        return gr.update(interactive=True)

    def create_warning_message(warning_text):
        return gr.HTML(
            f'<div style="padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold">{html.escape(warning_text)}</div>'
        )

    def create_experimental_warning_message():
        return create_warning_message(
            i18n("提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。")
        )

    with gr.Blocks(title="IndexTTS Demo") as demo:
        mutex = threading.Lock()

        def gen_single(
            emo_control_method,
            prompt,
            text,
            emo_ref_path,
            emo_weight,
            vec1,
            vec2,
            vec3,
            vec4,
            vec5,
            vec6,
            vec7,
            vec8,
            emo_text,
            emo_random,
            max_text_tokens_per_segment=120,
            *advanced_args,
            progress=gr.Progress(),
        ):
            tts = get_tts()
            prompt_orig_name = _audio_orig_name(prompt)
            emo_orig_name = _audio_orig_name(emo_ref_path)
            prompt_path = _audio_path(prompt)
            emo_ref_path = _audio_path(emo_ref_path)
            source_name = prompt_orig_name or emo_orig_name or "source"
            output_path = os.path.join(
                "outputs", build_download_filename(source_name, text)
            )
            tts.gr_progress = progress
            (
                do_sample,
                top_p,
                top_k,
                temperature,
                length_penalty,
                num_beams,
                repetition_penalty,
                max_mel_tokens,
            ) = advanced_args
            kwargs = {
                "do_sample": bool(do_sample),
                "top_p": float(top_p),
                "top_k": int(top_k) if int(top_k) > 0 else None,
                "temperature": float(temperature),
                "length_penalty": float(length_penalty),
                "num_beams": num_beams,
                "repetition_penalty": float(repetition_penalty),
                "max_mel_tokens": int(max_mel_tokens),
            }
            if type(emo_control_method) is not int:
                emo_control_method = emo_control_method.value
            if emo_control_method == 0:
                emo_ref_path = None
            if emo_control_method == 2:
                vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
                vec = tts.normalize_emo_vec(vec, apply_bias=True)
            else:
                vec = None
            if emo_text == "":
                emo_text = None

            with mutex:
                output = tts.infer(
                    spk_audio_prompt=prompt_path,
                    text=text,
                    output_path=output_path,
                    emo_audio_prompt=emo_ref_path,
                    emo_alpha=emo_weight,
                    emo_vector=vec,
                    use_emo_text=(emo_control_method == 3),
                    emo_text=emo_text,
                    use_random=emo_random,
                    verbose=verbose,
                    max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                    **kwargs,
                )
            return gr.update(value=output, visible=True)

        gr.HTML(
            """
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    """
        )

        with gr.Tab(i18n("音频生成")):
            with gr.Row():
                prompt_audio = NamedAudio(
                    label=i18n("音色参考音频"),
                    key="prompt_audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                with gr.Column():
                    input_text_single = gr.TextArea(
                        label=i18n("文本"),
                        key="input_text_single",
                        placeholder=i18n("请输入目标文本"),
                        info=f"{i18n('当前模型版本')}{model_version}",
                    )
                    gen_button = gr.Button(
                        i18n("生成语音"), key="gen_button", interactive=True
                    )
                output_audio = gr.Audio(
                    label=i18n("生成结果"), visible=True, key="output_audio"
                )

            with gr.Row():
                experimental_checkbox = gr.Checkbox(
                    label=i18n("显示实验功能"), value=False
                )
                glossary_checkbox = gr.Checkbox(
                    label=i18n("开启术语词汇读音"), value=glossary_enabled
                )
            with gr.Accordion(i18n("功能设置")):
                with gr.Row():
                    emo_control_method = gr.Radio(
                        choices=EMO_CHOICES_OFFICIAL,
                        type="index",
                        value=EMO_CHOICES_OFFICIAL[0],
                        label=i18n("情感控制方式"),
                    )
                    emo_control_method_all = gr.Radio(
                        choices=EMO_CHOICES_ALL,
                        type="index",
                        value=EMO_CHOICES_ALL[0],
                        label=i18n("情感控制方式"),
                        visible=False,
                    )

            with gr.Group(visible=False) as emotion_reference_group:
                with gr.Row():
                    emo_upload = NamedAudio(
                        label=i18n("上传情感参考音频"), type="filepath"
                    )

            with gr.Row(visible=False) as emotion_randomize_group:
                emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

            with gr.Group(visible=False) as emotion_vector_group:
                with gr.Row():
                    with gr.Column():
                        vec1 = gr.Slider(
                            label=i18n("喜"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                        vec2 = gr.Slider(
                            label=i18n("怒"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                        vec3 = gr.Slider(
                            label=i18n("哀"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                        vec4 = gr.Slider(
                            label=i18n("惧"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                    with gr.Column():
                        vec5 = gr.Slider(
                            label=i18n("厌恶"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                        vec6 = gr.Slider(
                            label=i18n("低落"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                        vec7 = gr.Slider(
                            label=i18n("惊喜"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )
                        vec8 = gr.Slider(
                            label=i18n("平静"),
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )

            with gr.Group(visible=False) as emo_text_group:
                create_experimental_warning_message()
                with gr.Row():
                    emo_text = gr.Textbox(
                        label=i18n("情感描述文本"),
                        placeholder=i18n(
                            "请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"
                        ),
                        value="",
                        info=i18n("例如：委屈巴巴、危险在悄悄逼近"),
                    )

            with gr.Row(visible=False) as emo_weight_group:
                emo_weight = gr.Slider(
                    label=i18n("情感权重"),
                    minimum=0.0,
                    maximum=1.0,
                    value=0.65,
                    step=0.01,
                )

            with gr.Accordion(
                i18n("自定义术语词汇读音"), open=False, visible=glossary_enabled
            ) as glossary_accordion:
                gr.Markdown(i18n("自定义个别专业术语的读音"))
                with gr.Row():
                    with gr.Column(scale=1):
                        glossary_term = gr.Textbox(
                            label=i18n("术语"), placeholder="IndexTTS2"
                        )
                        glossary_reading_zh = gr.Textbox(
                            label=i18n("中文读法"), placeholder="Index T-T-S 二"
                        )
                        glossary_reading_en = gr.Textbox(
                            label=i18n("英文读法"), placeholder="Index T-T-S two"
                        )
                        btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                    with gr.Column(scale=2):
                        glossary_table = gr.Markdown(value=i18n("暂无术语"))

            with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._"
                        )
                        with gr.Row():
                            do_sample = gr.Checkbox(
                                label="do_sample", value=True, info=i18n("是否进行采样")
                            )
                            temperature = gr.Slider(
                                label="temperature",
                                minimum=0.1,
                                maximum=2.0,
                                value=0.8,
                                step=0.1,
                            )
                        with gr.Row():
                            top_p = gr.Slider(
                                label="top_p",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.01,
                            )
                            top_k = gr.Slider(
                                label="top_k", minimum=0, maximum=100, value=30, step=1
                            )
                            num_beams = gr.Slider(
                                label="num_beams",
                                value=3,
                                minimum=1,
                                maximum=10,
                                step=1,
                            )
                        with gr.Row():
                            repetition_penalty = gr.Number(
                                label="repetition_penalty",
                                precision=None,
                                value=10.0,
                                minimum=0.1,
                                maximum=20.0,
                                step=0.1,
                            )
                            length_penalty = gr.Number(
                                label="length_penalty",
                                precision=None,
                                value=0.0,
                                minimum=-2.0,
                                maximum=2.0,
                                step=0.1,
                            )
                        max_mel_tokens = gr.Slider(
                            label="max_mel_tokens",
                            value=1500,
                            minimum=50,
                            maximum=max_mel_tokens_limit,
                            step=10,
                            info=i18n("生成Token最大数量，过小导致音频被截断"),
                            key="max_mel_tokens",
                        )
                    with gr.Column(scale=2):
                        gr.Markdown(
                            f"**{i18n('分句设置')}** _{i18n('参数会影响音频质量和生成速度')}_"
                        )
                        with gr.Row():
                            initial_value = max(
                                20, min(max_text_tokens_limit, gui_seg_tokens)
                            )
                            max_text_tokens_per_segment = gr.Slider(
                                label=i18n("分句最大Token数"),
                                value=initial_value,
                                minimum=20,
                                maximum=max_text_tokens_limit,
                                step=2,
                                key="max_text_tokens_per_segment",
                                info=i18n(
                                    "建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"
                                ),
                            )
                        with gr.Accordion(i18n("预览分句结果"), open=True):
                            segments_preview = gr.Dataframe(
                                headers=[
                                    i18n("序号"),
                                    i18n("分句内容"),
                                    i18n("Token数"),
                                ],
                                key="segments_preview",
                                wrap=True,
                            )

                advanced_params = [
                    do_sample,
                    top_p,
                    top_k,
                    temperature,
                    length_penalty,
                    num_beams,
                    repetition_penalty,
                    max_mel_tokens,
                ]

            example_table = gr.Dataset(
                label="Examples",
                samples_per_page=20,
                samples=get_example_cases(example_cases, include_experimental=False),
                type="values",
                components=[
                    prompt_audio,
                    emo_control_method_all,
                    input_text_single,
                    emo_upload,
                    emo_weight,
                    emo_text,
                    vec1,
                    vec2,
                    vec3,
                    vec4,
                    vec5,
                    vec6,
                    vec7,
                    vec8,
                ],
            )

        def on_example_click(example):
            return tuple(gr.update(value=example[idx]) for idx in range(14))

        def on_input_text_change(text, max_text_tokens_per_segment):
            tts = get_tts()
            if text and len(text) > 0:
                text_tokens_list = tts.tokenizer.tokenize(text)
                segments = tts.tokenizer.split_segments(
                    text_tokens_list,
                    max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                )
                data = []
                for idx, segment in enumerate(segments):
                    data.append([idx, "".join(segment), len(segment)])
                return {
                    segments_preview: gr.update(value=data, visible=True, type="array")
                }

            df = pd.DataFrame(
                [], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")]
            )
            return {segments_preview: gr.update(value=df)}

        def on_add_glossary_term(term, reading_zh, reading_en):
            tts = get_tts()
            term = term.rstrip()
            reading_zh = reading_zh.rstrip()
            reading_en = reading_en.rstrip()
            if not term:
                gr.Warning(i18n("请输入术语"))
                return gr.update()
            if not reading_zh and not reading_en:
                gr.Warning(i18n("请至少输入一种读法"))
                return gr.update()

            if reading_zh and reading_en:
                reading = {"zh": reading_zh, "en": reading_en}
            elif reading_zh:
                reading = {"zh": reading_zh}
            elif reading_en:
                reading = {"en": reading_en}
            else:
                reading = reading_zh or reading_en

            tts.normalizer.term_glossary[term] = reading
            try:
                tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
                gr.Info(i18n("词汇表已更新"), duration=1)
            except Exception as exc:
                gr.Error(i18n("保存词汇表时出错"))
                print(f"Error details: {exc}")
                return gr.update()
            return gr.update(value=format_glossary_markdown())

        def on_method_change(emo_control_method):
            if emo_control_method == 1:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            if emo_control_method == 2:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            if emo_control_method == 3:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def on_experimental_change(is_experimental, current_mode_index):
            new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
            new_index = (
                current_mode_index if current_mode_index < len(new_choices) else 0
            )
            return gr.update(
                choices=new_choices, value=new_choices[new_index]
            ), gr.update(
                samples=get_example_cases(
                    example_cases, include_experimental=is_experimental
                )
            )

        def on_glossary_checkbox_change(is_enabled):
            tts = get_tts()
            tts.normalizer.enable_glossary = is_enabled
            return gr.update(visible=is_enabled)

        def on_demo_load():
            tts = get_tts()
            try:
                tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
            except Exception as exc:
                gr.Error(i18n("加载词汇表时出错"))
                print(f"Failed to reload glossary on page load: {exc}")
            return gr.update(value=format_glossary_markdown())

        example_table.click(
            on_example_click,
            inputs=[example_table],
            outputs=[
                prompt_audio,
                emo_control_method,
                input_text_single,
                emo_upload,
                emo_weight,
                emo_text,
                vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6,
                vec7,
                vec8,
            ],
        )
        emo_control_method.change(
            on_method_change,
            inputs=[emo_control_method],
            outputs=[
                emotion_reference_group,
                emotion_randomize_group,
                emotion_vector_group,
                emo_text_group,
                emo_weight_group,
            ],
        )
        experimental_checkbox.change(
            on_experimental_change,
            inputs=[experimental_checkbox, emo_control_method],
            outputs=[emo_control_method, example_table],
        )
        glossary_checkbox.change(
            on_glossary_checkbox_change,
            inputs=[glossary_checkbox],
            outputs=[glossary_accordion],
        )
        input_text_single.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_segment],
            outputs=[segments_preview],
        )
        max_text_tokens_per_segment.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_segment],
            outputs=[segments_preview],
        )
        prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])
        btn_add_term.click(
            on_add_glossary_term,
            inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
            outputs=[glossary_table],
        )
        demo.load(on_demo_load, inputs=[], outputs=[glossary_table])
        gen_button.click(
            gen_single,
            inputs=[
                emo_control_method,
                prompt_audio,
                input_text_single,
                emo_upload,
                emo_weight,
                vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6,
                vec7,
                vec8,
                emo_text,
                emo_random,
                max_text_tokens_per_segment,
                *advanced_params,
            ],
            outputs=[output_audio],
        )

    demo.queue(20)
    return demo


if __name__ == "__main__":
    cli_args = parse_args()
    demo = build_demo(cli_args)
    demo.launch(server_name=cli_args.host, server_port=cli_args.port)
