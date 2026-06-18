"""Run-scoped image asset helpers for multimodal rendering.

The renderer stack does not ship processed multimodal features. Images are
written once into the run output tree and messages carry ``file://`` URLs to
those files. Renderers then emit lightweight image refs for vLLM only when the
engine needs to process an image.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
from pathlib import Path

RUN_OUTPUT_ROOT = Path("/data/outputs")

IMAGE_OFFLOAD_DIR_ENV = "VF_RENDERER_IMAGE_OFFLOAD_DIR"
RUN_DIR_ENV = "PRIME_RL_RUN_DIR"
RUN_ID_ENV = "RUN_ID"

IMAGE_ASSET_SUBDIR = Path("assets/images")
IMAGE_REF_PREFIX = "mmraw:v1"
IMAGE_REF_PAYLOAD_KEY = "_prime_rl_image_ref"
IMAGE_REF_PAYLOAD_VALUE = "raw_image"

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{16,64}$")
_SAFE_MM_HASH_RE = re.compile(r"^[a-f0-9]{16,128}$")
_SAFE_IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_GRID_THW_RE = re.compile(r"^[0-9]+x[0-9]+x[0-9]+$")

_MEDIA_TYPE_EXT = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp", "gif": ".gif"}


def normalize_run_id(run_id: str) -> str:
    """Return the canonical run id, without the directory's ``run_`` prefix."""
    value = run_id.strip()
    if value.startswith("run_"):
        value = value[len("run_") :]
    if not value or not _SAFE_RUN_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return value


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
        raise RuntimeError(f"Set {RUN_DIR_ENV} or {RUN_ID_ENV} before resolving a run directory.")
    return (RUN_OUTPUT_ROOT / run_dir_name(value)).resolve()


def run_image_dir(run_id: str | None = None) -> Path:
    """Resolve the directory for raw image assets for a run."""
    explicit = os.getenv(IMAGE_OFFLOAD_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).resolve()
    return (run_dir(run_id) / IMAGE_ASSET_SUBDIR).resolve()


def image_asset_dir(run_id: str | None = None) -> Path:
    """Alias for callers that already use the assets terminology."""
    return run_image_dir(run_id)


def _media_type_ext(media_type: str) -> str:
    subtype = media_type.split("/", 1)[-1].split(";", 1)[0].strip().lower()
    return _MEDIA_TYPE_EXT.get(subtype, ".img")


def offload_image_to_run_assets(url: object, image_dir: Path | None = None) -> tuple[str, int] | None:
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
    if not _SAFE_IMAGE_ID_RE.fullmatch(raw_image_id):
        raise ValueError(f"Invalid raw image id: {raw_image_id!r}")
    root = run_image_dir(run_id)
    path = (root / raw_image_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Raw image path escaped root: {path}")
    return path


def image_layout_fingerprint(
    *,
    family: str,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    min_pixels: int,
    max_pixels: int,
) -> str:
    raw = (
        f"image-layout:v1:{family}:{int(patch_size)}:{int(merge_size)}:"
        f"{int(temporal_patch_size)}:{int(min_pixels)}:{int(max_pixels)}"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _grid_to_ref(grid_thw: object) -> str:
    data = grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, (list, tuple)) or len(data) != 3:
        raise ValueError(f"Invalid image grid_thw for image ref: {grid_thw!r}")
    return "x".join(str(int(v)) for v in data)


def _grid_from_ref(value: str) -> list[int]:
    if not _SAFE_GRID_THW_RE.fullmatch(value):
        raise ValueError(f"Invalid image grid_thw ref segment: {value!r}")
    return [int(v) for v in value.split("x")]


def image_ref(
    *,
    run_id: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    raw_image_id: str,
    grid_thw: object,
) -> str:
    run_id = normalize_run_id(run_id)
    if not _SAFE_FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError(f"Invalid image layout fingerprint: {fingerprint!r}")
    if modality != "image":
        raise ValueError(f"Unsupported image ref modality: {modality!r}")
    if not _SAFE_MM_HASH_RE.fullmatch(mm_hash):
        raise ValueError(f"Invalid image hash: {mm_hash!r}")
    raw_image_path(run_id=run_id, raw_image_id=raw_image_id)
    return f"{IMAGE_REF_PREFIX}:{run_id}:{fingerprint}:{modality}:{mm_hash}:{raw_image_id}:{_grid_to_ref(grid_thw)}"


def split_image_ref(ref: str) -> tuple[str, str, str, str, str, list[int]]:
    parts = ref.split(":")
    if parts[:2] != ["mmraw", "v1"] or len(parts) != 8:
        raise ValueError(f"Invalid image ref shape: {ref!r}")
    return normalize_run_id(parts[2]), parts[3], parts[4], parts[5], parts[6], _grid_from_ref(parts[7])


# Backwards-compatible names for consumers that already speak the mmraw wire format.
MMRAW_PREFIX = IMAGE_REF_PREFIX
MM_RAW_PAYLOAD_KEY = IMAGE_REF_PAYLOAD_KEY
MM_RAW_PAYLOAD_VALUE = IMAGE_REF_PAYLOAD_VALUE
mmraw_ref = image_ref
split_mmraw_ref = split_image_ref
