"""Opt-in service integration; the Hub SDK owns durable execution admission."""

import gc
import inspect
import os
import threading
from functools import wraps


def managed_enabled():
    return os.getenv("HUB_RUNTIME_ENABLED") == "1"


# Set this before importing ttd_fastapi_utils, whose notifier is created on import.
if managed_enabled():
    os.environ["TTD_WEBHOOK_URL"] = ""


def managed_call(getter):
    if not managed_enabled():
        return lambda function: function
    from ttd_hub_runtime import managed_call as sdk_managed_call

    return sdk_managed_call(getter)


def managed_api_call(getter):
    """Translate control failures at the API boundary without changing its schema."""
    def decorate(function):
        if not managed_enabled():
            return function
        from fastapi import HTTPException
        from ttd_hub_runtime import HubError

        admitted = managed_call(getter)(function)

        def http_error(error):
            code = error.code
            status = {
                "conflict": 409, "duplicate": 409, "cancelled": 409,
                "single_gpu_capacity_exceeded": 422,
                "invalid_control_request": 400, "not_found": 404,
            }.get(code, 503)
            return HTTPException(status_code=status, detail=code)

        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapped(*args, **kwargs):
                try:
                    return await admitted(*args, **kwargs)
                except HubError as error:
                    raise http_error(error) from error
            return async_wrapped

        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return admitted(*args, **kwargs)
            except HubError as error:
                raise http_error(error) from error
        return wrapped
    return decorate


def add_managed_middleware(app):
    if managed_enabled():
        from ttd_hub_runtime import HubContextMiddleware

        app.add_middleware(HubContextMiddleware)


def cleanup_cuda():
    """Called only after the SDK drops its last model reference and drains work."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def synchronize_cuda():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_managed_model(loader):
    from ttd_hub_runtime import ManagedModel

    def checked_loader():
        model = loader()
        # Wrap this instance at the integration boundary; leave upstream source intact.
        model.infer = serialize_managed_inference(model.infer)
        observe_model_devices(model)
        if os.getenv("HUB_GLOSSARY_PATH"):
            model.glossary_path = os.environ["HUB_GLOSSARY_PATH"]
        return model

    return ManagedModel.from_env(checked_loader, cleanup=cleanup_cuda, synchronize=synchronize_cuda)


def observe_model_devices(model):
    """Record native placement without changing or rejecting CPU/GPU/offload choices."""
    if not managed_enabled():
        return
    summary = {"components": {}}

    def record(name, tensors, placement=None):
        devices, dtypes = set(), set()
        count = 0
        for _, tensor in tensors:
            devices.add(str(tensor.device))
            dtypes.add(str(tensor.dtype))
            count += 1
        result = {"devices": sorted(devices), "dtypes": sorted(dtypes), "tensor_count": count}
        if placement is not None:
            result["device_map"] = {str(k): str(v) for k, v in placement.items()}
        summary["components"][name] = result

    # Observability is best-effort; native model placement remains authoritative.
    for name in ("gpt", "semantic_model", "semantic_codec", "s2mel", "campplus_model", "bigvgan", "qwen_emo.model"):
        component = model
        for part in name.split("."):
            component = getattr(component, part, None)
        if component is None:
            continue
        try:
            record(name, list(component.named_parameters()) + list(component.named_buffers()),
                   getattr(component, "hf_device_map", None))
        except Exception:
            summary["components"][name] = {"observation": "unavailable"}
    for name in ("semantic_mean", "semantic_std", "emo_matrix", "spk_matrix"):
        value = getattr(model, name, None)
        if value is None:
            continue
        tensors = value if isinstance(value, (list, tuple)) else (value,)
        try:
            record(name, ((str(index), tensor) for index, tensor in enumerate(tensors)))
        except Exception:
            summary["components"][name] = {"observation": "unavailable"}
    model.__hub_device_summary__ = summary


# Index caches speaker/emotion tensors on the model. Its UI already serializes
# inference; use the same model-local execution rule for API and UI in managed mode.
_infer_lock = threading.RLock()


def serialize_managed_inference(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not managed_enabled():
            return function(*args, **kwargs)
        with _infer_lock:
            return function(*args, **kwargs)

    return wrapped
