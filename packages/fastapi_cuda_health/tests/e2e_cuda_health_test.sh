#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8099}
APP="tests.mock_cuda_app:app"

echo "[E2E] Using PORT=${PORT}"

python3 -V >/dev/null
pip3 -V >/dev/null || true

# Ensure fastapi & uvicorn installed (user site)
python3 - <<'PY'
import importlib, sys, subprocess
try:
    importlib.import_module('fastapi'); importlib.import_module('uvicorn')
    print('[E2E] fastapi/uvicorn already installed')
except Exception:
    print('[E2E] Installing fastapi/uvicorn to user site...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'fastapi==0.115.0', 'uvicorn==0.30.0'])
PY

# Start server
python3 -m uvicorn "${APP}" --host 127.0.0.1 --port "${PORT}" --log-level info &
PID=$!
trap 'kill ${PID} >/dev/null 2>&1 || true' EXIT

# Wait until server ready
for i in {1..50}; do
  if curl -s "http://127.0.0.1:${PORT}/" >/dev/null; then
    echo "[E2E] Server is up"
    break
  fi
  sleep 0.2
  if [ "$i" -eq 50 ]; then
    echo "[E2E] Server failed to start in time" >&2
    exit 1
  fi
done

# 1) HTTPException CUDA x3
echo "[E2E] Case1: HTTPException CUDA x3"
for i in 1 2 3; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/mock_http_cuda")
  echo "[E2E] /mock_http_cuda -> ${code} (expected 500)"; sleep 0.2
done
curl -s -D /tmp/e2e-headers1.out -o /dev/null -X POST "http://127.0.0.1:${PORT}/mock_http_cuda" || true
echo "[E2E] Headers (http cuda):"; sed -n '1,20p' /tmp/e2e-headers1.out
grep -iq '^x-cuda-error: *1' /tmp/e2e-headers1.out
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health")
echo "[E2E] /health -> ${code} (expected 503)"; [ "$code" = "503" ]

# 2) raw CUDA x2
echo "[E2E] Case2: raw CUDA exception x2"
for i in 1 2; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/mock_raw_cuda")
  echo "[E2E] /mock_raw_cuda -> ${code} (expected 500)"; sleep 0.2
done

# 3) non-CUDA 500
echo "[E2E] Case3: non-CUDA 500"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/mock_non_cuda")
echo "[E2E] /mock_non_cuda -> ${code} (expected 500)"

# 4) success 200
echo "[E2E] Case4: success 200"
curl -s "http://127.0.0.1:${PORT}/mock_ok" >/dev/null

# 再触发 3 次 HTTPException CUDA，/health 应再次变 503
echo "[E2E] Case5: HTTPException CUDA again x3"
for i in 1 2 3; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://127.0.0.1:${PORT}/mock_http_cuda" >/dev/null
  sleep 0.2
done
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health")
echo "[E2E] /health -> ${code} (expected 503)"; [ "$code" = "503" ]

echo "[E2E] PASSED"
