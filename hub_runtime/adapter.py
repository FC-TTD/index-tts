"""Index-specific hooks; no changes to upstream inference or device placement."""

from contextlib import ExitStack
from contextvars import ContextVar
import gc
import json
import os
from pathlib import Path
import threading
from functools import wraps
from fastapi import UploadFile
from download_filename import build_download_filename
from .parameters import DEFAULTS
from .startup import _ensure_runtime_cache_env

_progress = ContextVar("index_gradio_progress", default=None)


def load_model(args):
    _ensure_runtime_cache_env()
    from indextts.infer_v2_5 import IndexTTS2

    model = IndexTTS2(
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        model_dir=args.model_dir,
        use_bf16=bool(args.bf16),
        device=args.device,
        use_cuda_kernel=bool(args.use_cuda_kernel),
        use_deepspeed=bool(args.use_deepspeed),
        use_accel=bool(args.use_accel),
        use_torch_compile=bool(args.use_torch_compile),
        use_qwen_emo=True,
    )
    glossary = os.getenv("HUB_GLOSSARY_PATH")
    if glossary:
        model.glossary_path = glossary
    native_infer = model.infer
    lock = threading.RLock()

    @wraps(native_infer)
    def infer(*positional, **keywords):
        with lock:
            previous = getattr(model, "gr_progress", None)
            model.gr_progress = _progress.get()
            try:
                return native_infer(*positional, **keywords)
            finally:
                model.gr_progress = previous

    model.infer = infer
    return model


def completion(model):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def generate_ui(
    runtime,
    args,
    method,
    prompt_path,
    prompt_name,
    text,
    language,
    emotion_path,
    emotion_weight,
    vector,
    emotion_text,
    random_emotion,
    segment_tokens,
    advanced,
    progress,
):
    from .service import synthesize

    method = int(getattr(method, "value", method))
    values = dict(DEFAULTS)
    names = (
        "do_sample",
        "top_p",
        "top_k",
        "temperature",
        "length_penalty",
        "num_beams",
        "repetition_penalty",
        "max_mel_tokens",
    )
    values.update(zip(names, advanced))
    values.update(
        text=text,
        language=language,
        emo_alpha=float(emotion_weight),
        emo_vector=json.dumps(vector) if method == 2 else None,
        use_emo_text=method == 3,
        emo_text=emotion_text or None,
        use_random=bool(random_emotion),
        max_text_tokens_per_sentence=int(segment_tokens),
    )
    for name in ("top_k", "num_beams", "max_mel_tokens"):
        values[name] = int(values[name])
    source_name = prompt_name or "source.wav"
    with ExitStack() as files:
        values["prompt_speech"] = UploadFile(
            files.enter_context(open(prompt_path, "rb")), filename=source_name
        )
        values["emo_audio_prompt"] = (
            UploadFile(
                files.enter_context(open(emotion_path, "rb")),
                filename=Path(emotion_path).name,
            )
            if emotion_path and method != 0
            else None
        )
        token = _progress.set(progress)
        try:
            response = synthesize(runtime, args, **values)
        finally:
            _progress.reset(token)
    output = Path("outputs") / build_download_filename(source_name, text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.body)
    return str(output)
