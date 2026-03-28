from .cuda_health import CudaHealthMonitor
from .fastapi_entry import check_health, init_cuda_health_plugin, mount_health_route, setup_cuda_health
from . import postprocess
from . import preset
from .postprocess import (
    apply_postprocess,
    bandpass,
    butter_filter,
    delay,
    delay_tail,
    eq,
    highpass,
    limiter,
    lowpass,
    loudnorm,
    mix,
    reverb,
    saturate,
    trim_silence,
)
from .speed_control import apply_speed_to_wav_list, time_stretch_wav
from .ttd_notify import notifier
try:
    from .model_lifecycle import SmartModel
except Exception:  # pragma: no cover - optional heavy dependency path
    SmartModel = None

__all__ = [
    # cuda_health (lazy)
    "CudaHealthMonitor",
    "init_cuda_health_plugin",
    "check_health",
    "mount_health_route",
    "setup_cuda_health",
    # postprocess
    "postprocess",
    "preset",
    "loudnorm",
    "eq",
    "butter_filter",
    "lowpass",
    "highpass",
    "bandpass",
    "delay",
    "delay_tail",
    "reverb",
    "saturate",
    "mix",
    "limiter",
    "apply_postprocess",
    "trim_silence",
    # speed_control
    "time_stretch_wav",
    "apply_speed_to_wav_list",
    # notify
    "notifier",
    # lifecycle
    "SmartModel",
]
