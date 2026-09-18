"""Container startup settings; model residency is owned by Hub and Runtime."""

import argparse
import os
from pathlib import Path


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndexTTS Hub Runtime (API and native UI)"
    )
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
        "--bf16",
        action="store_true",
        default=False,
        help="Use BF16 to reduce memory and speed up on CUDA",
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
        default=False,
        help="Use Accelerate for multi-GPU if available (default: disabled)",
    )
    parser.add_argument(
        "--use_torch_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile for inference (default: disabled)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_model_dir(model_dir: str) -> None:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在，请先下载模型。")
    required = (
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "codec.pth",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "wav2vec2bert_stats.pt",
        "qwen0.6bemo4-merge",
    )
    missing = [
        name for name in required if not os.path.exists(os.path.join(model_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"模型目录 {model_dir} 缺少 IndexTTS 2.5 文件: {', '.join(missing)}"
        )
