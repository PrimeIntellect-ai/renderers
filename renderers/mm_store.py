"""Run-scoped image asset helpers for multimodal rendering.

The renderer stack does not ship processed multimodal features. Images are
written once into the run output tree and messages carry ``file://`` URLs to
those files. Renderers then emit lightweight image refs for vLLM only when the
engine needs to process an image.
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

RUN_OUTPUT_ROOT = Path("/data/outputs")

IMAGE_OFFLOAD_DIR_ENV = "VF_RENDERER_IMAGE_OFFLOAD_DIR"
IMAGE_STORAGE_ENV = "PRIME_RL_MM_IMAGE_STORAGE"
RUN_DIR_ENV = "PRIME_RL_RUN_DIR"
RUN_ID_ENV = "RUN_ID"

IMAGE_STORAGE_OFFLOAD = "offload"
IMAGE_STORAGE_INLINE = "inline"
IMAGE_STORAGE_MODES = {IMAGE_STORAGE_OFFLOAD, IMAGE_STORAGE_INLINE}

IMAGE_ASSET_SUBDIR = Path("assets/images")
IMAGE_REF_PREFIX = "mmraw"
IMAGE_REF_V2_PREFIX = "mmraw:v2"
IMAGE_REF_VERSION = "v3"
IMAGE_REF_PAYLOAD_KEY = "_prime_rl_image_ref"
IMAGE_REF_PAYLOAD_VALUE = "raw_image"
RAW_MM_ITEM_KIND = "prime_raw_mm_item"
RAW_MM_ITEM_VERSION = 1

_SAFE = {
    "run id": re.compile(r"^[A-Za-z0-9_.-]+$"),
    "multimodal family": re.compile(r"^[A-Za-z0-9_.-]+$"),
    "raw multimodal modality": re.compile(r"^[A-Za-z0-9_.-]+$"),
    "image layout fingerprint": re.compile(r"^[a-f0-9]{16,64}$"),
    "image hash": re.compile(r"^[a-f0-9]{16,128}$"),
    "raw image id": re.compile(r"^[A-Za-z0-9_.-]+$"),
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


def image_storage_mode() -> str:
    mode = os.getenv(IMAGE_STORAGE_ENV, IMAGE_STORAGE_OFFLOAD).strip().lower()
    if mode not in IMAGE_STORAGE_MODES:
        raise ValueError(
            f"{IMAGE_STORAGE_ENV} must be one of {sorted(IMAGE_STORAGE_MODES)}, got {mode!r}"
        )
    return mode


def normalize_run_id(run_id: str) -> str:
    """Return the canonical run id, without the directory's ``run_`` prefix."""
    value = run_id.strip()
    if value.startswith("run_"):
        value = value[len("run_") :]
    if not value:
        raise ValueError(f"Invalid run id: {run_id!r}")
    return _ensure_safe("run id", value)


def run_dir_name(run_id: str) -> str:
    return f"run_{normalize_run_id(run_id)}"


def current_run_id() -> str:
    """Best-effort run id for refs emitted by this process."""
    raw = os.getenv(RUN_ID_ENV, "").strip()
    if raw:
        return normalize_run_id(raw)

    run_dir = os.getenv(RUN_DIR_ENV, "").strip()
    if run_dir:
        return normalize_run_id(Path(run_dir).name)

    image_dir = os.getenv(IMAGE_OFFLOAD_DIR_ENV, "").strip()
    if image_dir:
        # Expected shape is <run_dir>/assets/images. If callers pass another
        # explicit directory, the ref's run segment is only a stable label; the
        # path resolver will use the explicit directory in every process.
        path = Path(image_dir).resolve()
        if path.name == "images" and path.parent.name == "assets":
            try:
                return normalize_run_id(path.parent.parent.name)
            except ValueError:
                pass
        return "explicit"

    if image_storage_mode() == IMAGE_STORAGE_INLINE:
        return "inline"

    raise RuntimeError(
        f"Set {IMAGE_OFFLOAD_DIR_ENV}, {RUN_DIR_ENV}, or {RUN_ID_ENV} before emitting image refs."
    )


def run_dir(run_id: str | None = None) -> Path:
    """Resolve the run output directory.

    Resolution order:
    1. ``PRIME_RL_RUN_DIR`` as an exact run directory.
    2. ``RUN_ID`` or explicit ``run_id`` under ``/data/outputs/run_<id>``.
    """
    explicit = os.getenv(RUN_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).resolve()

    value = run_id or os.getenv(RUN_ID_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"Set {RUN_DIR_ENV} or {RUN_ID_ENV} before resolving a run directory."
        )
    return (RUN_OUTPUT_ROOT / run_dir_name(value)).resolve()


def run_image_dir(run_id: str | None = None) -> Path:
    """Resolve the directory for raw image assets for a run."""
    explicit = os.getenv(IMAGE_OFFLOAD_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).resolve()
    return (run_dir(run_id) / IMAGE_ASSET_SUBDIR).resolve()


def _media_type_ext(media_type: str) -> str:
    subtype = media_type.split("/", 1)[-1].split(";", 1)[0].strip().lower()
    return _MEDIA_TYPE_EXT.get(subtype, ".img")


def offload_image_to_run_assets(
    url: object, image_dir: Path | None = None
) -> tuple[str, int] | None:
    """Decode a base64 data image into the run image assets directory.

    Returns ``(file_url, byte_count)`` when ``url`` was rewritten and ``None``
    for non-data-image values. Writes are content-addressed and atomic.
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
    return path.as_uri(), len(raw)


def raw_image_path(*, run_id: str, raw_image_id: str) -> Path:
    _ensure_safe("raw image id", raw_image_id)
    root = run_image_dir(run_id)
    path = (root / raw_image_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Raw image path escaped root: {path}")
    return path


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
    raw = f"image-layout:v1:{family}:{encoded_values}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def raw_mm_item(
    *,
    modality: str,
    family: str,
    layout_fingerprint: str,
    payload: dict[str, object],
    raw_uri: str | None = None,
    raw_image_id: str | None = None,
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
        "version": RAW_MM_ITEM_VERSION,
        "modality": modality,
        "family": family,
        "layout_fingerprint": layout_fingerprint,
        "payload": payload,
    }
    if vllm_modality is not None:
        out["vllm_modality"] = vllm_modality
    if raw_uri is not None:
        out["raw_uri"] = raw_uri
        out[IMAGE_REF_PAYLOAD_KEY] = IMAGE_REF_PAYLOAD_VALUE
    if raw_image_id is not None:
        out["raw_image_id"] = raw_image_id
        out[IMAGE_REF_PAYLOAD_KEY] = IMAGE_REF_PAYLOAD_VALUE
    return out


@dataclass(frozen=True)
class RawMMRef:
    run_id: str
    family: str
    fingerprint: str
    modality: str
    mm_hash: str
    payload: dict[str, object]
    raw_uri: str | None = None
    raw_image_id: str | None = None


def raw_mm_ref(
    *,
    run_id: str,
    family: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    raw_image_id: str | None = None,
    raw_uri: str | None = None,
    payload: dict[str, object] | None = None,
) -> str:
    """Generic raw multimodal asset ref.

    Adapter-owned details stay in the descriptor payload so refs can serve
    future families without baking shape names into the wire id.
    """
    run_id = normalize_run_id(run_id)
    _ensure_safe("multimodal family", family)
    _ensure_safe("image layout fingerprint", fingerprint)
    _ensure_safe("raw multimodal modality", modality)
    _ensure_safe("image hash", mm_hash)
    if raw_image_id is None and raw_uri is None:
        raise ValueError("raw multimodal refs require raw_image_id or raw_uri")
    if raw_image_id is not None:
        raw_image_path(run_id=run_id, raw_image_id=raw_image_id)
    if raw_uri is not None and not raw_uri:
        raise ValueError("raw_uri must be non-empty when set")

    ref_payload: dict[str, object] = {
        "run_id": run_id,
        "family": family,
        "fingerprint": fingerprint,
        "modality": modality,
        "mm_hash": mm_hash,
        "payload": payload or {},
    }
    if raw_image_id is not None:
        ref_payload["raw_image_id"] = raw_image_id
    if raw_uri is not None:
        ref_payload["raw_uri"] = raw_uri

    return f"{IMAGE_REF_PREFIX}:{IMAGE_REF_VERSION}:{_encode_ref_payload(ref_payload)}"


def split_raw_mm_ref(ref: str) -> RawMMRef:
    parts = ref.split(":")
    if parts[:2] == ["mmraw", "v2"] and len(parts) == 9:
        run_id, family, fingerprint, modality, mm_hash, raw_image_id, encoded_payload = (
            parts[2:]
        )
        return RawMMRef(
            run_id=normalize_run_id(run_id),
            family=_ensure_safe("multimodal family", family),
            fingerprint=_ensure_safe("image layout fingerprint", fingerprint),
            modality=_ensure_safe("raw multimodal modality", modality),
            mm_hash=_ensure_safe("image hash", mm_hash),
            payload=_decode_ref_payload(encoded_payload),
            raw_image_id=_ensure_safe("raw image id", raw_image_id),
        )

    if parts[:2] != ["mmraw", IMAGE_REF_VERSION] or len(parts) != 3:
        raise ValueError(f"Invalid raw multimodal ref shape: {ref!r}")

    payload = _decode_ref_payload(parts[2])
    run_id = payload.get("run_id")
    family = payload.get("family")
    fingerprint = payload.get("fingerprint")
    modality = payload.get("modality")
    mm_hash = payload.get("mm_hash")
    raw_uri = payload.get("raw_uri")
    raw_image_id = payload.get("raw_image_id")
    item_payload = payload.get("payload")

    if not isinstance(run_id, str):
        raise ValueError("Raw multimodal ref is missing run_id")
    if not isinstance(family, str):
        raise ValueError("Raw multimodal ref is missing family")
    if not isinstance(fingerprint, str):
        raise ValueError("Raw multimodal ref is missing fingerprint")
    if not isinstance(modality, str):
        raise ValueError("Raw multimodal ref is missing modality")
    if not isinstance(mm_hash, str):
        raise ValueError("Raw multimodal ref is missing mm_hash")
    if raw_uri is not None and not isinstance(raw_uri, str):
        raise ValueError("Raw multimodal ref raw_uri must be a string")
    if raw_image_id is not None and not isinstance(raw_image_id, str):
        raise ValueError("Raw multimodal ref raw_image_id must be a string")
    if raw_uri is None and raw_image_id is None:
        raise ValueError("Raw multimodal ref is missing an image source")
    if not isinstance(item_payload, dict):
        raise ValueError("Raw multimodal ref payload must be a dict")

    return RawMMRef(
        run_id=normalize_run_id(run_id),
        family=_ensure_safe("multimodal family", family),
        fingerprint=_ensure_safe("image layout fingerprint", fingerprint),
        modality=_ensure_safe("raw multimodal modality", modality),
        mm_hash=_ensure_safe("image hash", mm_hash),
        payload=item_payload,
        raw_uri=raw_uri,
        raw_image_id=_ensure_safe("raw image id", raw_image_id) if raw_image_id is not None else None,
    )


def is_raw_mm_ref(ref: object) -> bool:
    return isinstance(ref, str) and ref.startswith(f"{IMAGE_REF_PREFIX}:")
