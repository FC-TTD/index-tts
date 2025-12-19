# ttd-fastapi-utils

TTD FastAPI Utils: a small utility collection for FastAPI services.

Includes four modules:

- `cuda_health`: Global CUDA-aware health check (Plan B) that tracks recent failures.
- `home_probe`: Default home probe payload (app/build/container info) for reverse proxies.
- `ttd_notify`: A simple notifier helper (default webhook URL can be overridden via env).
- `postprocess`: Audio post-processing helpers, including LUFS-based loudness normalization with short-audio padding and a simple EQ.

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

## Quick Start

```python
from fastapi import FastAPI
from ttd_fastapi_utils import setup_cuda_health, apply_postprocess

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
    wav = apply_postprocess(wav, sr, target_loudness=-23.0, enable=True, trim_silence=False)
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

- `loudnorm(wav, sr, target_loudness=-23, threshold=0.99, block_sec=0.4)`
  - Pads short audio to `block_sec` for LUFS calculation, then applies gain to original length
  - Falls back to peak limiting when LUFS fails
  - Multi-channel: supports both `(T, C)` and `(C, T)` shapes (heuristic); only the first channel is used and returned
- `limiter(data, threshold=0.99)`
  - A simple peak limiter that scales samples above `threshold` down to `threshold` (element-wise)
  - Used by `loudnorm(...)` internally via its `threshold` parameter
- `eq(wav, sr)`
  - Simple high-band enhancement with safe Butterworth band-pass
- `apply_postprocess(wav, sr, target_loudness=-23.0, enable=True, trim_silence=False)`
  - Convenience wrapper with exception safety (optional `trim_silence` + `loudnorm(target_loudness=...)` + `eq`, so limiter is applied via `loudnorm`)
- `trim_silence(wav, sr, threshold_db=-40.0, min_silence_duration_ms=200, min_segment_ms=50, ignore_trailing_gap_ms=300, fade_ms=10)`
  - Trims leading and trailing silence using librosa.effects.split.
  - Multi-channel: supports both `(T, C)` and `(C, T)` shapes (heuristic)
  - `threshold_db`: Silence threshold relative to peak (default -40dB).
  - `min_silence_duration_ms`: Silence padding to keep (default 200ms).
  - `min_segment_ms`: Minimum length of non-silent segment to keep (default 50ms).
  - `ignore_trailing_gap_ms`: If last segment is far from previous (gap > this) and short, drop it (default 300ms).
  - `fade_ms`: Fade in/out duration (default 10ms).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
