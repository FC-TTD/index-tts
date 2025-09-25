# ttd-fastapi-utils

TTD FastAPI Utils: a small utility collection for FastAPI services.

Includes two components:

- CUDA Health Monitor: Global CUDA-aware health check (Plan B) that tracks recent failures and mounts `/health`.
- Postprocess: Audio post-processing helpers, including LUFS-based loudness normalization with short-audio padding and a simple EQ.

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
    wav = apply_postprocess(wav, sr, enable=True)
    return {"ok": True}
```

## Components

### CUDA Health Monitor

- Global detection (Plan B):
  - HTTPException with status >= 500 whose detail contains CUDA keywords is treated as a CUDA failure
  - Uncaught exceptions with CUDA keywords are treated as CUDA failures
  - Non-CUDA 5xx responses are recorded as healthy to avoid sticky unhealthy
- Suppresses `uvicorn.access` logs for `/health` and `/docs` by default
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

Environment variables:

- `CONSECUTIVE_FAIL_LIMIT` (default `3`)
- `TTD_WEBHOOK_URL` (optional) for default notifier

### Postprocess

- `loudnorm(wav, sr, target_loudness=-23, threshold=0.99, block_sec=0.4)`
  - Pads short audio to `block_sec` for LUFS calculation, then applies gain to original length
  - Falls back to peak limiting when LUFS fails
  - Multi-channel: only the first channel is used and returned
- `eq(wav, sr)`
  - Simple high-band enhancement with safe Butterworth band-pass
- `apply_postprocess(wav, sr, enable=True)`
  - Convenience wrapper with exception safety

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
