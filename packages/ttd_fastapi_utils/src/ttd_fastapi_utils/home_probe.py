import json
import os
import re
from typing import Any, Optional


def read_buildinfo() -> dict[str, Any]:
    buildinfo_path = os.getenv("BUILDINFO_PATH", "").strip()
    candidates: list[str] = []
    if buildinfo_path:
        candidates.append(buildinfo_path)
    candidates.extend([
        os.path.join(os.getcwd(), "buildinfo.json"),
        "/app/buildinfo.json",
    ])

    for p in candidates:
        try:
            if not p:
                continue
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                v = json.load(f)
            if isinstance(v, dict):
                v = dict(v)
                v["path"] = p
                return v
        except Exception:
            continue

    env_build = {
        "build_commit": os.getenv("BUILD_COMMIT", ""),
        "build_time": os.getenv("BUILD_TIME", ""),
    }
    return {k: v for k, v in env_build.items() if v}


def read_container_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": os.getenv("HOSTNAME", ""),
        "pid": os.getpid(),
        "in_container": bool(os.path.exists("/.dockerenv")),
    }

    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            cgroup = f.read()
        m = re.search(r"([0-9a-f]{12,64})", cgroup)
        if m:
            info["container_id"] = m.group(1)
    except Exception:
        pass

    v = os.getenv("NVIDIA_VISIBLE_DEVICES", "")
    if v:
        info["nvidia_visible_devices"] = v

    return info


def build_home_payload(
    *,
    app_name: str,
    healthy: bool,
    unhealthy_detail: Optional[object],
) -> dict[str, Any]:
    buildinfo = read_buildinfo()
    container = read_container_info()

    message = f"欢迎使用 {app_name}"
    if isinstance(buildinfo, dict) and (buildinfo.get("build_commit") or buildinfo.get("build_time")):
        bc = str(buildinfo.get("build_commit") or "")
        bt = str(buildinfo.get("build_time") or "")
        if bc and bt:
            message = f"欢迎使用 {app_name} (build {bc} at {bt})"
        elif bc:
            message = f"欢迎使用 {app_name} (build {bc})"
        elif bt:
            message = f"欢迎使用 {app_name} (build at {bt})"

    payload: dict[str, Any] = {
        "message": message,
        "app": {"name": app_name},
        "status": "healthy" if healthy else "unhealthy",
        "build": buildinfo,
        "container": container,
    }

    if not healthy:
        payload["detail"] = unhealthy_detail

    return payload
