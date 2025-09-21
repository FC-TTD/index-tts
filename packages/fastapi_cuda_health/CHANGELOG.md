# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-21

- Switch to Plan B (global detection) design:
  - Count as CUDA failure when HTTPException (>=500) detail contains CUDA keywords
  - Count as CUDA failure for uncaught exceptions whose message contains CUDA keywords
  - Treat non-CUDA 5xx as healthy to avoid sticky-unhealthy windows
- Add automatic `x-cuda-error: 1` response header for CUDA 5xx HTTPException
- Keep readiness predicate and unified `/health` mounting helper
- Add optional TRACE on `/health` with light diagnostics
- Suppress `uvicorn.access` logs for `/health` and `/docs`
- Improve module/class/function docstrings and README to reflect Plan B
- Update packaging metadata and documentation; bump version to 0.2.0

## [0.1.x]

- Initial package skeleton and decorator-based tracking prototype
