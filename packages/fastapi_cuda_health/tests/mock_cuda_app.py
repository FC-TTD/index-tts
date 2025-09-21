import os
import sys
import logging
from fastapi import FastAPI, HTTPException

# 将项目根目录加入 sys.path，以便导入 tools.health_plugin
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 让本地包 src 可被导入（未发布/未安装场景）
sys.path.append(os.path.join(PROJECT_ROOT, "packages", "fastapi-cuda-health", "src"))
from fastapi_cuda_health.plugin import setup_cuda_health  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock-cuda-app")

app = FastAPI(title="Mock CUDA Health App")

# 安装插件（Plan B：全局判定 CUDA 错误，不依赖端点标记）
setup_cuda_health(app, path="/health", ready_predicate=lambda: True)


@app.post("/mock_http_cuda")
async def mock_http_cuda():
    """通过 HTTPException 触发 CUDA 错误（包含 CUDA 关键词）"""
    detail = (
        "语音生成处理错误: CUDA error: an illegal memory access was encountered\n"
        "CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\n"
        "For debugging consider passing CUDA_LAUNCH_BLOCKING=1\n"
        "Compile with `TORCH_USE_CUDA_DSA`  to enable device-side assertions.\n"
    )
    raise HTTPException(status_code=500, detail=detail)


@app.post("/mock_raw_cuda")
async def mock_raw_cuda():
    """直接抛出原生异常，包含 CUDA 关键词（不通过 HTTPException）"""
    raise RuntimeError("CUDA error: an illegal memory access was encountered")


@app.post("/mock_non_cuda")
async def mock_non_cuda():
    """返回 500 但不包含 CUDA 关键词，应该不会计入失败窗口。"""
    raise HTTPException(status_code=500, detail="some transient server error")


@app.get("/")
async def root():
    return {"ok": True}


@app.get("/mock_ok")
async def mock_ok():
    return {"ok": True}
