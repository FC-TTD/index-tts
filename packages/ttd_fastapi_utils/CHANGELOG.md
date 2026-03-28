# Changelog

## 0.3.2 (2026-03-28)

- postprocess
  - Clarify `delay(...)` semantics: it returns a composite signal that already includes the direct signal
  - Add `delay_tail(...)` for explicit wet-only echo tails
- preset
  - Add `preset_metadata(...)` so UI/documentation can consume stable preset labels, summaries, primary controls, and recommended standard-chain defaults
- debug app / docs
  - Make preset labels, help text, and recommended standard postprocess defaults derive from preset metadata instead of duplicated UI-side hardcoded tables
  - Add a dedicated `PRESET_UI_GUIDE.md` to document the recommended split between `Preset` implementation and `Preset UI`

## 0.3.1 (2026-03-28)

- postprocess / preset
  - Add reusable filter/effect primitives: `butter_filter`, `lowpass`, `highpass`, `bandpass`, `delay`, `reverb`, `saturate`, `mix`, `limiter`
  - Add style presets: `telephone`, `smart_assistant`, `inner_monologue`, `radio`, `intercom`
  - Separate `Standard Postprocess` from `Style Preset` so business-side cleanup and style design can be applied independently
  - Keep `apply_postprocess(...)` focused on cleanup: optional `trim_silence`, `loudnorm`, and optional `eq`
- debug app
  - Add single-file Gradio preset debugger with bilingual UI, auto-play output, logging, and independent `Standard Postprocess` / `Style Preset` controls
  - Improve preset sync so preset defaults and switch states are reflected correctly in the UI
- package
  - Export `postprocess` and `preset` modules from top-level package
  - Make `SmartModel` import optional to avoid forcing heavy dependencies in lightweight postprocess-only environments

## 0.3.0 (2026-01-27)

- speed_control
  - Add `time_stretch_wav` and `apply_speed_to_wav_list` for pitch-preserving time-stretch via SoX (`sox tempo -s`).
  - SoX is required by default; if not found, a `FileNotFoundError` is raised.
  - Set environment variable `TTD_SPEED_CONTROL_BYPASS_SOX` to bypass SoX and passthrough the original audio.

## 0.2.4 (2026-01-24)

- SmartModel: New model lifecycle management wrapper
  - Added `SmartModel` class for lazy loading and auto-unloading of models
  - Supports configurable timeout-based auto-unloading (TTL) with background monitoring
  - Includes manual `unload()` method and automatic GPU memory cleanup
  - Thread-safe implementation with proper locking mechanisms
  - Updated package description to include SmartModel functionality

## 0.2.3 (2025-12-20)

- Compatibility improvements
  - Lowered Python version requirement from >=3.9 to >=3.8
  - Relaxed core dependency version constraints for broader compatibility:
    - fastapi: >=0.100 → >=0.95
    - numpy: >=1.22 → >=1.19
    - scipy: >=1.10 → >=1.5
    - pyloudnorm: >=0.1.1 → >=0.1.0
    - requests: >=2.31 → >=2.0
  - Added `from __future__ import annotations` to cuda_health.py for Python 3.8 type annotation support
  - Removed deprecated cuda_health_legacy.py file

## 0.2.2 (2025-12-19)

- Postprocess
  - `apply_postprocess(wav, sr, target_loudness=..., enable=True)` now accepts `target_loudness` and forwards it to `loudnorm(...)`
  - `apply_postprocess(...)` now accepts `trim_silence` (default False) to optionally run `trim_silence(wav, sr)` before `loudnorm + eq`
  - Improved multi-channel handling heuristics for `(T, C)` and `(C, T)` shaped arrays in `loudnorm(...)` and `trim_silence(...)`

## 0.2.1 (2025-12-19)

- Postprocess: add `limiter(data, threshold=...)`
  - Used by `loudnorm(...)` internally via its `threshold` parameter

## 0.2.0 (2025-12-19)

- CUDA Health Monitor: refactored module layout
  - Core CUDA health monitor logic lives in `ttd_fastapi_utils.cuda_health`
  - FastAPI integration entry lives in `ttd_fastapi_utils.fastapi_entry`
  - Default home probe payload lives in `ttd_fastapi_utils.home_probe`
  - Notifier helper lives in `ttd_fastapi_utils.ttd_notify`
- Default home probe (GET/HEAD `/`)
  - Enabled by default via `setup_cuda_health(..., enable_default_home=True)`
  - Returns 200 when healthy, 503 when unhealthy
  - Payload includes app name, build info, and container info
- Access log suppression
  - `suppress_access_paths` is now `None` by default in `setup_cuda_health`
  - When `enable_default_home=True` and `suppress_access_paths` is not provided, `/` is suppressed by default (along with `/health` and `/docs`)
  - Suppression for `/` uses exact match to avoid filtering all paths
  
## 0.1.2 (2025-11-26)

- Enhanced health check log filtering and notification configuration
- Improved access log filter to identify and block uptime-kuma monitoring requests by User-Agent and message content
- Added exception handling to ensure robustness of log filtering
- Added type annotations for `check_health` function's monitor variable
- Updated default webhook URL configuration

## 0.1.1 (2025-09-24)

- CUDA Health Monitor: add parameter `terminate_on_unhealthy` (default True)
- When unhealthy (last N results are CUDA failures), a background thread sends SIGTERM to self after a short delay; if the process still runs after grace period, force exit with code 1
- README updated to reflect the new parameter name and behavior

## 0.1.0 (2025-09-24)

- Initial release of ttd-fastapi-utils
- Component 1: CUDA Health Monitor (migrated from fastapi-cuda-health, Plan B global detection)
- Component 2: Postprocess (audio loudness normalization with short-audio padding and EQ)
- Added README, LICENSE, tests, and Notify helper
