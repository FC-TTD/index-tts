"""Record native placement for smoke evidence without moving or rejecting tensors."""

from itertools import chain


def observe_devices(model):
    summary = {
        "model_device": str(getattr(model, "device", "unknown")),
        "components": {},
    }

    def record(name, tensors, placement=None):
        devices, dtypes, count = set(), set(), 0
        for tensor in tensors:
            devices.add(str(tensor.device))
            dtypes.add(str(tensor.dtype))
            count += 1
        item = {
            "devices": sorted(devices),
            "dtypes": sorted(dtypes),
            "tensor_count": count,
        }
        if placement is not None:
            item["device_map"] = {
                str(key): str(value) for key, value in placement.items()
            }
        summary["components"][name] = item

    for name in (
        "gpt",
        "semantic_model",
        "semantic_codec",
        "s2mel",
        "campplus_model",
        "bigvgan",
        "qwen_emo",
    ):
        component = getattr(model, name, None)
        if name == "qwen_emo":
            component = getattr(component, "model", None)
        if component is not None:
            record(
                name,
                chain(component.parameters(), component.buffers()),
                getattr(component, "hf_device_map", None),
            )
    for name in ("semantic_mean", "semantic_std", "emo_matrix", "spk_matrix"):
        value = getattr(model, name, None)
        if value is not None:
            record(name, value if isinstance(value, (list, tuple)) else (value,))
    return summary
