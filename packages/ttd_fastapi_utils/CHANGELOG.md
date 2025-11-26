# Changelog

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
