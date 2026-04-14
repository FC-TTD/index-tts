#!/usr/bin/env python3
import argparse
from contextlib import asynccontextmanager
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置当前目录和路径（需在依赖导入前完成）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
indextts_dir = os.path.join(current_dir, "indextts")
if indextts_dir not in sys.path:
    sys.path.append(indextts_dir)


def _ensure_runtime_cache_env() -> None:
    hf_home = Path(os.getenv("HF_HOME", "/opt/hf_cache"))
    hf_hub_cache = Path(os.getenv("HF_HUB_CACHE", str(hf_home / "hub")))
    transformers_cache = Path(
        os.getenv("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    )
    modelscope_cache = Path(os.getenv("MODELSCOPE_CACHE", str(hf_home / "modelscope")))
    torch_extensions_dir = Path(
        os.getenv("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    )
    torchinductor_cache_dir = Path(
        os.getenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")
    )

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("MODELSCOPE_CACHE", str(modelscope_cache))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_extensions_dir))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache_dir))

    for path in (
        hf_home,
        hf_hub_cache,
        transformers_cache,
        modelscope_cache,
        torch_extensions_dir,
        torchinductor_cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


_ensure_runtime_cache_env()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import librosa
import soundfile as sf
import uvicorn

from indextts.infer_v2 import IndexTTS2

try:
    from ttd_fastapi_utils import (
        SmartModel,
        apply_postprocess,
        setup_cuda_health,
        trim_silence,
    )
except ImportError:
    from packages.ttd_fastapi_utils.src.ttd_fastapi_utils import (
        SmartModel,
        apply_postprocess,
        setup_cuda_health,
        trim_silence,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("index-tts-api")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndexTTS API")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose mode"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the API on"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="Model checkpoints directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 to reduce memory and speed up on CUDA",
    )
    parser.add_argument(
        "--use_cuda_kernel",
        action="store_true",
        default=True,
        help="Use BigVGAN custom CUDA kernel (CUDA only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on, e.g., 'cuda:0', 'cpu'",
    )
    parser.add_argument(
        "--use_deepspeed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use DeepSpeed if available (default: disabled)",
    )
    parser.add_argument(
        "--use_accel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Accelerate for multi-GPU if available (default: disabled)",
    )
    parser.add_argument(
        "--use_torch_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.compile for inference (default: disabled)",
    )
    parser.add_argument(
        "--preload_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload the model at startup so runtime requests do not trigger external downloads",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_model_dir(model_dir: str) -> None:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在，请先下载模型。")


def create_tts_manager(args: argparse.Namespace) -> SmartModel:
    def loader():
        return IndexTTS2(
            cfg_path=os.path.join(args.model_dir, "config.yaml"),
            model_dir=args.model_dir,
            use_fp16=bool(args.fp16),
            device=args.device,
            use_cuda_kernel=bool(args.use_cuda_kernel),
            use_deepspeed=bool(args.use_deepspeed),
            use_accel=bool(args.use_accel),
            use_torch_compile=bool(args.use_torch_compile),
        )

    return SmartModel(loader, timeout_seconds=7200)


def create_app(args: argparse.Namespace) -> FastAPI:
    validate_model_dir(args.model_dir)
    model_state = {"ready": False}
    tts_manager: SmartModel | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal tts_manager

        os.makedirs("outputs", exist_ok=True)

        logger.info(
            "正在初始化 IndexTTS2 模型管理器... (fp16=%s, use_cuda_kernel=%s, device=%s, use_deepspeed=%s)",
            args.fp16,
            args.use_cuda_kernel,
            args.device,
            args.use_deepspeed,
        )

        try:
            tts_manager = create_tts_manager(args)
            if args.preload_model:
                logger.info(
                    "启动阶段预热 IndexTTS2 模型，等待权重、缓存和 CUDA 扩展全部就绪..."
                )
                tts_manager.get()
                logger.info("IndexTTS2 模型预热完成")
            app.state.tts_manager = tts_manager
            app.state.cmd_args = args
            model_state["ready"] = True
            logger.info("IndexTTS2 模型管理器初始化完成")
            yield
        finally:
            logger.info("正在关闭 IndexTTS2 模型管理器...")
            model_state["ready"] = False
            if tts_manager:
                tts_manager.stop()
            app.state.tts_manager = None

    app = FastAPI(
        title="IndexTTS API",
        description="IndexTTS2 语音合成 API 服务",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.cmd_args = args
    app.state.tts_manager = None

    setup_cuda_health(
        app,
        path="/health",
        ready_predicate=lambda: model_state["ready"],
        enable_default_home=False,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/generate")
    async def generate_audio(
        text: str = Form(...),
        prompt_speech: UploadFile = File(...),
        emo_audio_prompt: UploadFile | None = File(None),
        emo_alpha: float = Form(1.0),
        emo_vector: str | None = Form(None),
        use_emo_text: bool = Form(False),
        emo_text: str | None = Form(None),
        use_random: bool = Form(False),
        interval_silence: int = Form(200),
        max_text_tokens_per_sentence: int = Form(120),
        do_sample: bool = Form(True),
        top_p: float = Form(0.8),
        top_k: int = Form(30),
        temperature: float = Form(0.8),
        length_penalty: float = Form(0.0),
        num_beams: int = Form(3),
        repetition_penalty: float = Form(10.0),
        max_mel_tokens: int = Form(1500),
        remove_silence: bool = Form(True),
        postprocess: bool = Form(True),
        lufs: float = Form(-23.0),
        expected_duration: float | None = Form(None),
        speed: float = Form(1.0),
        pitch: float = Form(0.0),
    ):
        if app.state.tts_manager is None:
            raise HTTPException(status_code=503, detail="模型未初始化")

        try:
            temp_path = None
            emo_temp_path = None
            try:
                contents = await prompt_speech.read()
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(contents)

                if emo_audio_prompt is not None:
                    emo_contents = await emo_audio_prompt.read()
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as emo_file:
                        emo_temp_path = emo_file.name
                        emo_file.write(emo_contents)

                emo_vector_list = None
                if emo_vector:
                    try:
                        parsed = json.loads(emo_vector)
                        if isinstance(parsed, list) and all(
                            isinstance(x, (int, float)) for x in parsed
                        ):
                            emo_vector_list = [float(x) for x in parsed]
                    except Exception:
                        logger.warning("emo_vector 解析失败，已忽略")

                kwargs = {
                    "do_sample": bool(do_sample),
                    "top_p": float(top_p),
                    "temperature": float(temperature),
                    "length_penalty": float(length_penalty),
                    "num_beams": num_beams,
                    "repetition_penalty": float(repetition_penalty),
                    "max_mel_tokens": int(max_mel_tokens),
                }
                if int(top_k) > 0:
                    kwargs["top_k"] = int(top_k)

                output_path = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ).name
                tts = app.state.tts_manager.get()

                max_retries = 2
                tolerance = 0.05
                current_speed = float(speed)
                current_expected = None
                try:
                    if expected_duration is not None:
                        current_expected = float(expected_duration)
                except Exception:
                    current_expected = None

                wav = None
                sr = None
                for attempt in range(max_retries + 1):
                    logger.info(
                        "开始生成语音（v2），文本长度: %s, speed=%s, expected_duration=%s, attempt=%s/%s",
                        len(text),
                        current_speed,
                        current_expected,
                        attempt + 1,
                        max_retries + 1,
                    )

                    wav_path = tts.infer(
                        spk_audio_prompt=temp_path,
                        text=text,
                        output_path=output_path,
                        emo_audio_prompt=emo_temp_path,
                        emo_alpha=float(emo_alpha),
                        emo_vector=emo_vector_list,
                        use_emo_text=bool(use_emo_text),
                        emo_text=emo_text,
                        use_random=bool(use_random),
                        interval_silence=int(interval_silence),
                        duration_ratio=float(current_speed),
                        verbose=args.verbose,
                        max_text_tokens_per_segment=int(max_text_tokens_per_sentence),
                        **kwargs,
                    )

                    logger.info("语音生成完成，保存到: %s", wav_path)
                    wav, sr = sf.read(wav_path, dtype="float32")

                    if current_expected is None or current_expected <= 0:
                        break

                    try:
                        aligned_wav = trim_silence(
                            wav, int(sr), min_silence_duration_ms=0
                        )
                        measured = (
                            float(len(aligned_wav)) / float(sr)
                            if sr and len(aligned_wav)
                            else 0.0
                        )
                    except Exception:
                        logger.exception(
                            "expected_duration: 时长测量失败，已跳过自适应"
                        )
                        break

                    if measured <= 0:
                        break

                    rel_err = abs(measured - current_expected) / current_expected
                    if rel_err <= tolerance or attempt >= max_retries:
                        break

                    new_speed = current_speed * (measured / current_expected)
                    if not (new_speed > 0):
                        break
                    logger.info(
                        "expected_duration: measured=%.3fs expected=%.3fs rel_err=%.2f%%, update speed %.4f -> %.4f and retry",
                        measured,
                        current_expected,
                        rel_err * 100.0,
                        current_speed,
                        new_speed,
                    )
                    current_speed = float(new_speed)

                if wav is None or sr is None:
                    raise HTTPException(
                        status_code=500, detail="语音生成失败：未获得有效音频"
                    )

                if remove_silence:
                    try:
                        wav = trim_silence(wav, sr)
                    except Exception:
                        logger.exception("静音切除失败，已跳过")

                try:
                    if float(pitch) != 0 and abs(float(pitch)) > 1e-6:
                        wav = librosa.effects.pitch_shift(
                            wav, sr=sr, n_steps=float(pitch)
                        )
                except Exception:
                    logger.exception("音高移调失败，已跳过移调")

                if postprocess:
                    wav = apply_postprocess(
                        wav, sr, target_loudness=float(lufs), enable=True
                    )

                buffer = BytesIO()
                sf.write(buffer, wav, sr, format="WAV")
                buffer.seek(0)

                return Response(
                    headers={
                        "Content-Disposition": "attachment; filename=generated.wav"
                    },
                    content=buffer.read(),
                    media_type="audio/wav",
                )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                if emo_temp_path and os.path.exists(emo_temp_path):
                    os.remove(emo_temp_path)
                if (
                    "output_path" in locals()
                    and output_path
                    and os.path.exists(output_path)
                ):
                    os.remove(output_path)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("语音生成处理错误: %s", exc)
            raise HTTPException(status_code=500, detail=f"语音生成处理错误: {str(exc)}")

    return app


def run_app(args: argparse.Namespace) -> None:
    logger.info("启动 IndexTTS API 服务，端口: %s，主机: %s", args.port, args.host)
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    run_app(parse_args())
