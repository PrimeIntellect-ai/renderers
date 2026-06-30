"""Run-scoped image asset helpers for multimodal rendering.

The default renderer multimodal mode does not ship processed image features.
Images are written once into the run output tree and messages carry ``file://``
URLs to those files. Renderers then emit lightweight image refs for vLLM only
when the engine needs to process an image.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

# Contract: must match prime_rl.utils.run_assets.IMAGE_OFFLOAD_DIR_ENV.
IMAGE_OFFLOAD_DIR_ENV = "VF_RENDERER_IMAGE_OFFLOAD_DIR"

IMAGE_REF_PREFIX = "mmraw"
RAW_MM_ITEM_KIND = "prime_raw_mm_item"

_SAFE = {
    "multimodal family": re.compile(r"^[A-Za-z0-9_.-]+$"),
    "raw multimodal modality": re.compile(r"^[A-Za-z0-9_.-]+$"),
    "image layout fingerprint": re.compile(r"^[a-f0-9]{16,64}$"),
    "image hash": re.compile(r"^[a-f0-9]{16,128}$"),
    "raw multimodal ref payload segment": re.compile(r"^[A-Za-z0-9_-]*$"),
}

_MEDIA_TYPE_EXT = {
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "png": ".png",
    "webp": ".webp",
    "gif": ".gif",
}


def _ensure_safe(label: str, value: str) -> str:
    if not _SAFE[label].fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def run_image_dir() -> Path:
    """Resolve the directory for raw image assets for a run."""
    explicit = os.getenv(IMAGE_OFFLOAD_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).resolve()
    raise RuntimeError(
        f"Set {IMAGE_OFFLOAD_DIR_ENV} before resolving raw image assets."
    )


def _media_type_ext(media_type: str) -> str:
    subtype = media_type.split("/", 1)[-1].split(";", 1)[0].strip().lower()
    return _MEDIA_TYPE_EXT.get(subtype, ".img")


def offload_image_to_run_assets(
    url: object, image_dir: Path | None = None
) -> str | None:
    """Decode a base64 data image into the run image assets directory.

    Returns a ``file://`` URL when ``url`` was rewritten and ``None`` for
    non-data-image values. Writes are content-addressed and atomic.
    """
    if not isinstance(url, str) or not url.startswith("data:image/"):
        return None
    marker = ";base64,"
    if marker not in url:
        return None

    header, b64 = url.split(marker, 1)
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None

    root = (image_dir or run_image_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()[:16]
    path = root / f"{digest}{_media_type_ext(header[len('data:') :])}"
    if not path.exists():
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    else:
        try:
            path.touch()
        except OSError:
            pass
    return path.as_uri()


def _json_fingerprint_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _encode_ref_payload(payload: dict[str, object] | None) -> str:
    raw = json.dumps(payload or {}, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_ref_payload(encoded: str) -> dict[str, object]:
    _ensure_safe("raw multimodal ref payload segment", encoded)
    padded = encoded + "=" * (-len(encoded) % 4)
    payload = json.loads(
        base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("Raw multimodal ref payload must decode to a dict")
    return payload


def image_layout_fingerprint(*, family: str, **values: object) -> str:
    """Stable adapter-owned fingerprint for raw multimodal layout contracts."""
    _ensure_safe("multimodal family", family)
    encoded_values = ":".join(
        f"{key}={_json_fingerprint_value(values[key])}" for key in sorted(values)
    )
    raw = f"image-layout:{family}:{encoded_values}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def raw_mm_item(
    *,
    modality: str,
    family: str,
    layout_fingerprint: str,
    payload: dict[str, object],
    raw_image_uri: str,
    vllm_modality: str | None = None,
) -> dict[str, object]:
    """Build the JSON-safe raw multimodal descriptor envelope.

    ``payload`` is intentionally adapter-owned. Shared consumers may route by
    ``family`` and validate the common envelope, but must not inspect adapter
    payload keys.
    """
    _ensure_safe("multimodal family", family)
    _ensure_safe("raw multimodal modality", modality)
    _ensure_safe("image layout fingerprint", layout_fingerprint)
    out: dict[str, object] = {
        "kind": RAW_MM_ITEM_KIND,
        "modality": modality,
        "family": family,
        "layout_fingerprint": layout_fingerprint,
        "payload": payload,
    }
    if vllm_modality is not None:
        out["vllm_modality"] = vllm_modality
    out["raw_image_uri"] = raw_image_uri
    return out


@dataclass(frozen=True)
class RawMMRef:
    family: str
    fingerprint: str
    modality: str
    mm_hash: str
    payload: dict[str, object]
    raw_image_uri: str


def raw_mm_ref(
    *,
    family: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    raw_image_uri: str,
    payload: dict[str, object] | None = None,
) -> str:
    """Generic raw multimodal asset ref.

    Adapter-owned details stay in the descriptor payload so refs can serve
    future families without baking shape names into the wire id.
    """
    _ensure_safe("multimodal family", family)
    _ensure_safe("image layout fingerprint", fingerprint)
    _ensure_safe("raw multimodal modality", modality)
    _ensure_safe("image hash", mm_hash)

    ref_payload: dict[str, object] = {
        "family": family,
        "fingerprint": fingerprint,
        "modality": modality,
        "mm_hash": mm_hash,
        "payload": payload or {},
        "raw_image_uri": raw_image_uri,
    }

    return f"{IMAGE_REF_PREFIX}:{_encode_ref_payload(ref_payload)}"


def split_raw_mm_ref(ref: str) -> RawMMRef:
    parts = ref.split(":")
    if len(parts) != 2 or parts[0] != IMAGE_REF_PREFIX:
        raise ValueError(f"Invalid raw multimodal ref shape: {ref!r}")

    payload = _decode_ref_payload(parts[1])
    family = payload.get("family")
    fingerprint = payload.get("fingerprint")
    modality = payload.get("modality")
    mm_hash = payload.get("mm_hash")
    raw_image_uri = payload.get("raw_image_uri")
    item_payload = payload.get("payload")

    if not isinstance(family, str):
        raise ValueError("Raw multimodal ref is missing family")
    if not isinstance(fingerprint, str):
        raise ValueError("Raw multimodal ref is missing fingerprint")
    if not isinstance(modality, str):
        raise ValueError("Raw multimodal ref is missing modality")
    if not isinstance(mm_hash, str):
        raise ValueError("Raw multimodal ref is missing mm_hash")
    if not isinstance(raw_image_uri, str):
        raise ValueError("Raw multimodal ref is missing raw_image_uri")
    if not isinstance(item_payload, dict):
        raise ValueError("Raw multimodal ref payload must be a dict")

    return RawMMRef(
        family=_ensure_safe("multimodal family", family),
        fingerprint=_ensure_safe("image layout fingerprint", fingerprint),
        modality=_ensure_safe("raw multimodal modality", modality),
        mm_hash=_ensure_safe("image hash", mm_hash),
        payload=item_payload,
        raw_image_uri=raw_image_uri,
    )


def is_raw_mm_ref(ref: object) -> bool:
    return isinstance(ref, str) and ref.startswith(f"{IMAGE_REF_PREFIX}:")
