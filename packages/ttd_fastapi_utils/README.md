# ttd-fastapi-utils

TTD FastAPI Utils: a small utility collection for FastAPI services.

Includes five modules:

- `cuda_health`: Global CUDA-aware health check (Plan B) that tracks recent failures.
- `home_probe`: Default home probe payload (app/build/container info) for reverse proxies.
- `ttd_notify`: A simple notifier helper (default webhook URL can be overridden via env).
- `postprocess`: Audio post-processing helpers, including LUFS-based loudness normalization with short-audio padding and a simple EQ.
- `preset`: Ready-to-use audio style presets built on top of `postprocess`.
- `speed_control`: Pitch-preserving time-stretch helpers via external tools (SoX/FFmpeg).

## Repository Status

`ttd-fastapi-utils` is currently colocated inside the `index-tts` repository only for fast integration testing.
It is maintained as an independent package with its own packaging and publication path, and it should not be treated as part of the `index-tts` service runtime or deployment lifecycle.

Practical implications:

- Release this package through `packages/pub.yml`, not through the service Docker/Swarm deployment flow.
- Validate package changes in the package context first.
- Do not assume the repository root `.venv` is the correct environment for this package.
- When running tests from the repository root, make sure imports resolve to `packages/ttd_fastapi_utils/src` instead of an already installed wheel or site-package copy.

## Install (monorepo)

This repository uses the src layout. You may import directly in the monorepo or install the wheel built from this package.

```python
from ttd_fastapi_utils import setup_cuda_health, apply_postprocess
```

If running outside the monorepo, build and install the wheel:

```bash
# from packages/ttd_fastapi_utils
python -m build  # or `hatch build`
pip install dist/ttd_fastapi_utils-*.whl
```

## Package-Local Development

Treat this package as an independent Python project even when working from the monorepo.

Recommended local workflow:

```bash
# from packages/ttd_fastapi_utils
uv venv
PYTHONPATH=src uv run pytest tests/test_home_probe.py tests/test_fastapi_integration.py
```

If you prefer running from the repository root, keep imports pinned to the local package source:

```bash
# from repository root
PYTHONPATH="packages/ttd_fastapi_utils/src" uv run pytest \
  packages/ttd_fastapi_utils/tests/test_home_probe.py \
  packages/ttd_fastapi_utils/tests/test_fastapi_integration.py
```

Notes:

- Prefer the package-local environment under `packages/ttd_fastapi_utils/` for package validation.
- Do not assume the repository root `.venv` matches this package's dependency set.
- `PYTHONPATH=src` or `PYTHONPATH="packages/ttd_fastapi_utils/src"` avoids accidentally testing an already installed wheel.

## Pre-Publish Routing

Before publishing `ttd-fastapi-utils`, validate the package through the package path, not the service deployment path.

Release routing checklist:

- Run package tests against the local development source.
- Run at least one FastAPI integration test that mounts the plugin into a real app and exercises real HTTP requests.
- Include one real audio HTTP integration check for `postprocess` or `preset`, so publication is not gated only by health-route coverage.
- Include FastAPI integration coverage for the remaining exported runtime helpers that affect service behavior, including `SmartModel`, `speed_control`, and notifier wiring.
- Publish through `packages/pub.yml`.
- Do not use `ansible/site.yml`, Docker Compose, or Swarm deployment as the default validation path for this package.

## Quick Start

```python
from fastapi import FastAPI
from ttd_fastapi_utils import postprocess, setup_cuda_health

app = FastAPI()

# Optional readiness predicate
is_ready = False

def ready_predicate():
    return is_ready

# 1) CUDA health
monitor = setup_cuda_health(
    app,
    path="/health",
    ready_predicate=ready_predicate,
    # When unhealthy, send SIGTERM to self to trigger container restart (default True)
    terminate_on_unhealthy=True,
)

# 2) Postprocess (audio)
import soundfile as sf

@app.post("/generate")
async def generate():
    wav, sr = sf.read("/path/to.wav", dtype="float32")
    wav = postprocess.apply_postprocess(
        wav,
        sr,
        target_loudness=-23.0,
        enable=True,
        trim_silence=False,
    )
    return {"ok": True}
```

## Components

### CUDA Health Monitor

- Global detection (Plan B):
  - HTTPException with status >= 500 whose detail contains CUDA keywords is treated as a CUDA failure
  - Uncaught exceptions with CUDA keywords are treated as CUDA failures
  - Non-CUDA 5xx responses are recorded as healthy to avoid sticky unhealthy
- Suppresses `uvicorn.access` logs for `/health` and `/docs` by default
- Optional default home probe for reverse proxies
  - If enabled, GET/HEAD `/` returns 200 when healthy, or 503 when unhealthy
  - Response payload includes app name, build info, and container info
- Provides a notifier hook (`Notify`) for one-shot alerting when flipping unhealthy

Public API:

- `CudaHealthMonitor`
- `init_cuda_health_plugin(app, ...)`
- `check_health(app, is_ready: Optional[bool] = None, default_unhealthy_detail: str = "CUDA 错误导致不健康")`
- `mount_health_route(app, path: str = "/health")`
- `setup_cuda_health(app, ...)`

Parameters (selected):

- `terminate_on_unhealthy: bool = True`
  When health flips to unhealthy (the last N tracked results are CUDA failures),
  a background thread sends SIGTERM to the current process (after a short delay) to trigger container restart. If the process does not exit after grace period, the thread calls `os._exit(1)` as a last resort. Set to `False` in local development if you prefer to keep the process running for debugging.

- `enable_default_home: bool = True`
  When enabled, installs a middleware that serves default home probe response at `/` if you have not defined your own GET/HEAD `/` route.

- `suppress_access_paths: Optional[Iterable[str]] = None`
  If `None`, defaults to `("/health", "/docs")`, and also includes `"/"` when `enable_default_home=True`.

Environment variables:

- `CONSECUTIVE_FAIL_LIMIT` (default `3`)
- `TTD_WEBHOOK_URL` (optional) for default notifier
- `APP_NAME` (optional) for default home probe payload when `FastAPI(title=...)` is not set
- `BUILDINFO_PATH` (optional) buildinfo json path
- `BUILD_COMMIT`, `BUILD_TIME` (optional) fallback build metadata when buildinfo file is not present

### Home Probe

- `build_home_payload(app_name, healthy, unhealthy_detail)`
  - Includes build info from `BUILDINFO_PATH` / `buildinfo.json` (fallback to `BUILD_COMMIT` / `BUILD_TIME`)
  - Includes container info from env and `/proc/self/cgroup`

### Notify

- `notifier`
  - Default webhook URL can be overridden via `TTD_WEBHOOK_URL`

### Postprocess

Recommended import style:

```python
from ttd_fastapi_utils import postprocess

wav = postprocess.delay(wav, sr, delay_ms=80.0, decay=0.35, repeats=2)
wav = postprocess.bandpass(wav, sr, low_cut_hz=300.0, high_cut_hz=3400.0)
```

- `loudnorm(wav, sr, target_loudness=-23, threshold=0.99, block_sec=0.4)`
  - Pads short audio to `block_sec` for LUFS calculation, then applies gain to original length
  - Falls back to peak limiting when LUFS fails
  - Multi-channel: supports both `(T, C)` and `(C, T)` shapes (heuristic); only the first channel is used and returned
- `limiter(data, threshold=0.99)`
  - A simple peak limiter that scales samples above `threshold` down to `threshold` (element-wise)
  - Used by `loudnorm(...)` internally via its `threshold` parameter
- `butter_filter(audio, sr, btype=..., cutoff=..., order=4)`
  - 通用 Butterworth 滤波原语，适合业务层自行封装 preset
- `lowpass(wav, sr, cutoff_hz, order=4)`
- `highpass(wav, sr, cutoff_hz, order=4)`
- `bandpass(wav, sr, low_cut_hz, high_cut_hz, order=4)`
  - 基础滤波积木，适合组合电话、广播、朦胧、回忆等效果
- `delay(wav, sr, delay_ms, decay, repeats)`
  - 多次衰减延迟；返回包含原始信号的 composite signal
- `delay_tail(wav, sr, delay_ms, decay, repeats)`
  - 多次衰减延迟；仅返回 wet-only echo tail，不包含原始信号
- `reverb(wav, sr, room_size=0.45, damping=0.35, pre_delay_ms=18.0)`
  - 轻量 Schroeder 风格混响原语，适合业务层自己叠加房间感、尾音和空间感
- `saturate(wav, drive=1.3)`
  - 基于 `tanh` 的软削波/饱和
- `mix(dry, wet, wet_ratio=0.5)`
  - 干湿混合，`wet_ratio` 越大效果越重
- `eq(wav, sr)`
  - Simple high-band enhancement with safe Butterworth band-pass
- `apply_postprocess(wav, sr, target_loudness=-23.0, enable=True, trim_silence=False, enable_eq=True)`
  - Convenience wrapper with exception safety (optional `trim_silence` + `loudnorm(target_loudness=...)` + optional `eq`)
- `trim_silence(wav, sr, threshold_db=-40.0, min_silence_duration_ms=200, min_segment_ms=50, ignore_trailing_gap_ms=300, fade_ms=10)`
  - Trims leading and trailing silence using librosa.effects.split.
  - Multi-channel: supports both `(T, C)` and `(C, T)` shapes (heuristic)
  - `threshold_db`: Silence threshold relative to peak (default -40dB).
  - `min_silence_duration_ms`: Silence padding to keep (default 200ms).
  - `min_segment_ms`: Minimum length of non-silent segment to keep (default 50ms).
  - `ignore_trailing_gap_ms`: If last segment is far from previous (gap > this) and short, drop it (default 300ms).
  - `fade_ms`: Fade in/out duration (default 10ms).

Example composition in business layer:

```python
from ttd_fastapi_utils import postprocess

filtered = postprocess.bandpass(wav, sr, low_cut_hz=300.0, high_cut_hz=3400.0)
colored = postprocess.saturate(filtered, drive=1.5)
echo = postprocess.delay(colored, sr, delay_ms=80.0, decay=0.35, repeats=2)
room = postprocess.reverb(colored, sr, room_size=0.35, damping=0.45, pre_delay_ms=16.0)
wet = postprocess.mix(echo, room, wet_ratio=0.4)
wav_out = postprocess.limiter(postprocess.mix(colored, wet, wet_ratio=0.25), threshold=0.98)
```

Recommended separation:

```python
from ttd_fastapi_utils import postprocess, preset

# 1) Standard postprocess for cleanup / consistency
wav = postprocess.apply_postprocess(
    wav,
    sr,
    target_loudness=-23.0,
    enable=True,
    trim_silence=False,
    enable_eq=True,
)

# 2) Style preset for character / sound design
wav = preset.smart_assistant(wav, sr)
```

Implementation guidance:

- 不要把 `Standard Postprocess` 和 `Style Preset` 混成一条隐式链路。
- `Standard Postprocess` 更适合给 TTS / ASR / 语音生成结果做统一扫尾，例如：
  - `trim_silence`
  - `loudnorm`
  - `eq`
- `Style Preset` 应该视为独立的音色设计步骤，按需单独 `apply`：
  - 电话音 / 智能助手 / 心声 / 广播 / 对讲机
- 推荐业务侧显式串联，而不是让 preset 内部偷偷带标准链：

```python
wav = postprocess.apply_postprocess(wav, sr, ...)
wav = preset.apply_preset("smart_assistant", wav, sr)
```

更完整的实现建议见：

- [`PRESET_UI_GUIDE.md`](./PRESET_UI_GUIDE.md)

### Preset

Recommended import style:

```python
from ttd_fastapi_utils import preset

wav = preset.apply_preset("telephone", wav, sr)
wav2 = preset.inner_monologue(wav, sr, delay_ms=95.0, wet_ratio=0.35)
```

- `list_presets()`
  - 当前内置：`telephone`、`smart_assistant`、`inner_monologue`、`radio`、`intercom`
- `preset_metadata(name=None)`
  - 返回 preset 元数据，适合 UI 层读取显示名、主要参数和推荐标准链默认值
- `apply_preset(name, wav, sr, **kwargs)`
  - 按名称应用 preset，支持中英文别名，例如 `电话`、`智能语音`、`心声`
- `telephone(wav, sr, ...)`
  - 电话音：窄带 + 轻饱和
- `smart_assistant(wav, sr, ...)`
  - 模拟智能语音：更干净、清晰、略带数字感，并带一点受控空间感
  - 建议业务侧先跑 `postprocess.apply_postprocess(...)`，再叠加该 preset
- `inner_monologue(wav, sr, ...)`
  - 心声独白：柔和低通 + 短回声 + 轻混响
- `radio(wav, sr, ...)`
  - 收音机/广播：更强中频和一点箱体空间感
- `intercom(wav, sr, ...)`
  - 对讲机：更窄、更硬的中频质感

Debug app:

```bash
uv run python packages/ttd_fastapi_utils/examples/postprocess_debug_app.py --host 0.0.0.0 --port 7861
```

- 上传音频后，可切换 `preset` / `custom` 两种模式
- `preset` 模式下可直接试听电话音、模拟智能语音、心声独白、广播、对讲机
- `custom` 模式下可手动调 `标准链 / 滤波 / 饱和 / 延迟 / 混响 / 限幅`
- `preset` 模式下，`Standard Postprocess` 与 `Style Preset` 现在是两套独立流程

### Speed Control

- `time_stretch_wav(wav, sr, speed, allow_passthrough_on_failure=True)`
  - Pitch-preserving time-stretch via SoX (`sox tempo -s`).
  - Requires SoX to be installed on the system.
  - Set environment variable `TTD_SPEED_CONTROL_BYPASS_SOX` to any non-empty value to bypass SoX and passthrough the original audio.
- `apply_speed_to_wav_list(wavs, sr, speed, allow_passthrough_on_failure=True)`

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
