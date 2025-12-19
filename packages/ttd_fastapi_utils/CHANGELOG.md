# Changelog

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
