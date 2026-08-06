"""Shared multimodal image helpers used by VL family renderers.

Family-specific geometry (Qwen ``smart_resize``, Kimi MoonViT layout, …) stays
in the family modules. This module owns the cross-family bits: content-part
detection, ``file://`` asset loading, Pillow dimension / PIL decode helpers,
and resolving the checkpoint name layout knobs are sourced from.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def is_image_part(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    t = item.get("type")
    if t in ("image", "image_url"):
        return True
    if t is not None:
        return False
    # Untyped fallback for loosely-shaped image parts. Require a truthy
    # value: HF Arrow schema unification (Dataset.from_list over a list of
    # heterogeneous content dicts) fills missing keys with None, so any
    # text part round-tripped through a Dataset will have ``image_url: None``
    # as a key. Mere key presence isn't enough.
    return bool(item.get("image")) or bool(item.get("image_url"))


def is_video_part(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    t = item.get("type")
    if t in ("video", "video_url"):
        return True
    if t is not None:
        return False
    return bool(item.get("video")) or bool(item.get("video_url"))


def image_source(item: dict[str, Any]) -> Any:
    if "image" in item:
        return item["image"]
    if "image_url" in item:
        image_url = item.get("image_url")
        return image_url.get("url") if isinstance(image_url, dict) else image_url
    return item.get("url") or item.get("path")


def offloaded_image_path(source: Any) -> Path:
    """The one accepted raw-mode image source: an offloaded ``file://`` URL."""
    if isinstance(source, str):
        parsed = urlparse(source)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path)).resolve()
    raise ValueError(
        "v1 multimodal image rendering requires offloaded file:// image assets"
    )


def load_pil_image(item: dict[str, Any]):
    """Resolve an ImagePart to a PIL Image for processed multimodal output.

    Accepted sources (local only — the renderer is not an HTTP client):

    - a preloaded ``PIL.Image.Image``
    - a filesystem path or ``file://`` URL
    - a ``data:image/...;base64,...`` URI
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Processed multimodal rendering requires Pillow. Install "
            "`renderers[vision]` or provide Pillow in the caller environment."
        ) from exc

    raw = image_source(item)
    if isinstance(raw, Image.Image):
        return raw.convert("RGB") if raw.mode != "RGB" else raw

    if not isinstance(raw, str):
        raise TypeError(
            f"Unsupported image source {type(raw).__name__!r}; expected PIL "
            "Image, local path, file:// URL, or data: URI."
        )

    if raw.startswith("data:image/") and ";base64," in raw:
        _, _, payload = raw.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if raw.startswith("data:"):
        raise ValueError(
            "Processed multimodal rendering only accepts data:image/...;base64,... "
            f"URIs, got {raw.split(',', 1)[0]!r}"
        )

    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        raise ValueError(
            "Processed multimodal rendering does not fetch remote images; "
            "pass a PIL Image, local path, file:// URL, or data: URI "
            f"(got {parsed.scheme}://...)."
        )
    if parsed.scheme == "file":
        return Image.open(unquote(parsed.path)).convert("RGB")
    if parsed.scheme == "":
        return Image.open(raw).convert("RGB")

    raise ValueError(f"Unsupported image URL scheme: {parsed.scheme!r} in {raw!r}")


def load_image_asset(part: dict[str, Any]) -> tuple[Path, bytes]:
    """Resolve a part's offloaded image source and read it once."""
    path = offloaded_image_path(image_source(part))
    return path, path.read_bytes()


def image_dimensions(raw: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to read image dimensions for multimodal rendering."
        ) from exc

    with Image.open(io.BytesIO(raw)) as image:
        return image.height, image.width


def pil_image_hash(pil_image) -> str:
    h = hashlib.sha256()
    h.update(pil_image.tobytes())
    h.update(f"{pil_image.size}".encode())
    return h.hexdigest()[:32]


def layout_model_name(tokenizer, renderer_name: str) -> str:
    """The checkpoint name the raw layout knobs are sourced from."""
    name = getattr(tokenizer, "name_or_path", None)
    if not name:
        raise RuntimeError(
            f"{renderer_name} needs the checkpoint name to resolve raw image "
            "layout knobs. Load the tokenizer with a known name_or_path."
        )
    return name
