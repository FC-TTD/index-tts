import os
import re
import time
from urllib.parse import quote


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]+')
_UNDERSCORES = re.compile(r"_+")


def sanitize_filename_part(
    value: object, fallback: str, max_chars: int, *, strip_extension: bool = False
) -> str:
    """Return a safe filename segment while preserving non-ASCII text."""
    text = "" if value is None else str(value)
    if strip_extension:
        text = os.path.basename(text)
        text = os.path.splitext(text)[0]
    text = _INVALID_FILENAME_CHARS.sub("_", text)
    text = _UNDERSCORES.sub("_", text).strip(" ._")
    if not text:
        text = fallback
    return text[:max_chars]


def build_download_filename(
    source_name: object,
    inference_text: object,
    *,
    epoch_ms: int | None = None,
) -> str:
    if epoch_ms is None:
        epoch_ms = int(time.time() * 1000)
    source = sanitize_filename_part(source_name, "source", 48, strip_extension=True)
    text = sanitize_filename_part(inference_text, "text", 40)
    return f"{source}-{text}-{int(epoch_ms)}.wav"


def build_content_disposition(filename: str) -> str:
    ascii_filename = filename.encode("ascii", "replace").decode("ascii").replace('"', "_")
    encoded_filename = quote(filename, safe="")
    return (
        f'attachment; filename="{ascii_filename}"; '
        f"filename*=UTF-8''{encoded_filename}"
    )
