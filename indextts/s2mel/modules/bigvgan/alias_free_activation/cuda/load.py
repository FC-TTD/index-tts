# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib

import torch
from torch.utils import cpp_extension

"""
Setting this param to a list has a problem of generating different compilation commands (with diferent order of architectures) and leading to recompilation of fused kernels. 
Set it to empty stringo avoid recompilation and assign arch flags explicity in extra_cuda_cflags below
"""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def _get_cuda_cache_root() -> pathlib.Path:
    cache_root = os.getenv("TORCH_EXTENSIONS_DIR")
    if cache_root:
        return pathlib.Path(cache_root) / "bigvgan_cuda"
    return pathlib.Path("/tmp/torch_extensions") / "bigvgan_cuda"


def _get_runtime_cc_flag():
    if not torch.cuda.is_available():
        return []

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return [
        "-gencode",
        f"arch=compute_{arch},code=sm_{arch}",
    ]


def load():
    cc_flag = _get_runtime_cc_flag()

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = _get_cuda_cache_root() / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ]
            + extra_cuda_flags
            + cc_flag,
            verbose=True,
        )

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    anti_alias_activation_cuda = _cpp_extention_load_helper(
        "anti_alias_activation_cuda", sources, extra_cuda_flags
    )

    return anti_alias_activation_cuda


def _create_build_dir(buildpath):
    try:
        os.makedirs(buildpath, exist_ok=True)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
