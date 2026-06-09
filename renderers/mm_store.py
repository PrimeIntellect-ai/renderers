"""Shared run-scoped artifact store for offloaded multimodal data.

Two subsystems offload heavy multimodal data to ``/data`` during a rollout,
ship a cheap reference, and re-load it on the consumer:

1. **Image offload** — raw images written to
   ``<run_dir>/assets/images/<hash>.jpg`` and shipped as ``file://`` refs.
2. **MM-feature offload** — processed vLLM ``MultiModalKwargsItem`` payloads
   written to
   ``<run_dir>/assets/mm_features/v1/vllm-mmitem/<fingerprint>/<modality>/<hash[:2]>/<hash>.msgpack``
   and shipped as ``mmfile:v1:<run_id>:<fingerprint>:<modality>:<mm_hash>``
   tuple refs.
3. **Raw-image inference refs** — raw images already written under
   ``<run_dir>/assets/images`` and shipped as compact
   ``mmraw:v1:<run_id>:<fingerprint>:<modality>:<mm_hash>:<raw_image_id>:<grid>``
   refs. vLLM loads the raw image and runs its own processor.

This module is the single source of truth for the on-disk layout, the
fingerprint, the ref strings, and the msgpack envelope. It lives in
``renderers`` because that is the lowest common dependency of both the
verifiers env-worker client (writer of images), the renderers generate client
(writer of features), and prime-rl (reader of both). The writer/reader
file-I/O halves stay in their respective consumers; only the shared contract
lives here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# Root of every run's output tree. ``/data/outputs/run_<run_id>`` in prod.
RUN_OUTPUT_ROOT = Path("/data/outputs")
MM_FEATURE_ROOT_ENV = "PRIME_RL_MM_FEATURE_ROOT"

MMFILE_PREFIX = "mmfile:v1"
MMRAW_PREFIX = "mmraw:v1"
MM_PAYLOAD_MODE_ENV = "RENDERERS_MM_FEATURE_STORE_MODE"
MM_RAW_PAYLOAD_KEY = "_prime_rl_mm_payload"
MM_RAW_PAYLOAD_VALUE = "raw"

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{16,64}$")
_SAFE_MM_HASH_RE = re.compile(r"^[a-f0-9]{16,128}$")
_SAFE_RAW_IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_GRID_THW_RE = re.compile(r"^[0-9]+x[0-9]+x[0-9]+$")

_MM_FEATURE_SCHEMA_VERSION = 1
_MM_FEATURE_KIND = "vllm.MultiModalKwargsItem"
_MM_RAW_SCHEMA_VERSION = 1

# Run-dir-relative asset subdirs, for callers that already hold a run dir.
IMAGE_ASSET_SUBDIR = Path("assets/images")
FEATURE_ASSET_SUBDIR = Path("assets/mm_features")


def run_id_from_env() -> str:
    """Return the safe ``RUN_ID`` from the environment.

    The platform injects ``RUN_ID`` into every container (env worker,
    orchestrator, inference) so the run-scoped artifact dir can be derived
    consistently across pods that don't share other env.
    """
    run_id = os.environ.get("RUN_ID", "").strip()
    if not run_id or not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise RuntimeError("RUN_ID must be set to a safe run id before writing multimodal feature artifacts.")
    return run_id


def run_dir(run_id: str) -> Path:
    """``<root>/run_<run_id>`` (resolved root from env or ``RUN_OUTPUT_ROOT``)."""
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"Invalid multimodal feature run id: {run_id!r}")
    root = Path(os.environ.get(MM_FEATURE_ROOT_ENV, str(RUN_OUTPUT_ROOT)))
    return root / f"run_{run_id}"


def image_asset_dir(run_id: str) -> Path:
    """``<run_dir>/assets/images``, resolved."""
    return (run_dir(run_id) / IMAGE_ASSET_SUBDIR).resolve()


def feature_asset_dir(run_id: str) -> Path:
    """``<run_dir>/assets/mm_features``, resolved.

    This is the root that ``mm_feature_path`` builds under and that the
    traversal guard checks against.
    """
    return (run_dir(run_id) / FEATURE_ASSET_SUBDIR).resolve()


def mm_payload_mode() -> str:
    """Return the inference multimodal payload mode.

    Explicit env wins:

    - ``raw``: send only cache-only ``None`` slots or raw-image ``mmraw`` refs.
    - ``on`` / ``processed``: legacy processed ``mmfile`` feature artifacts.
    - ``off`` / ``inline``: legacy inline base64 processed payloads.

    With no explicit env, hosted runs (``RUN_ID`` present) default to ``raw`` so
    env workers avoid the image processor. Local/dev processes without
    ``RUN_ID`` keep the old processed-artifact behavior.
    """
    raw = os.getenv(MM_PAYLOAD_MODE_ENV)
    if raw is None:
        return "raw" if os.getenv("RUN_ID", "").strip() else "processed"
    mode = raw.strip().lower()
    if mode in {"raw", "raw-ref", "raw_refs", "mmraw"}:
        return "raw"
    if mode in {"", "1", "true", "on", "enabled", "yes", "processed", "mmfile"}:
        return "processed"
    if mode in {"0", "false", "off", "disabled", "none", "no", "inline"}:
        return "inline"
    raise ValueError(f"Invalid {MM_PAYLOAD_MODE_ENV}={mode!r}; expected raw, on/processed, or off/inline.")


def mm_feature_fingerprint(*, family: str, spatial_merge_size: int) -> str:
    import importlib.metadata

    parts = {
        "schema_version": _MM_FEATURE_SCHEMA_VERSION,
        "kind": _MM_FEATURE_KIND,
        "family": family,
        "spatial_merge_size": spatial_merge_size,
        "vllm": importlib.metadata.version("vllm"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
    }
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def mm_processor_fingerprint(
    *,
    family: str,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    min_pixels: int,
    max_pixels: int,
) -> str:
    import importlib.metadata

    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "missing"

    parts = {
        "schema_version": _MM_RAW_SCHEMA_VERSION,
        "kind": "raw-image-processor-layout",
        "family": family,
        "patch_size": int(patch_size),
        "merge_size": int(merge_size),
        "temporal_patch_size": int(temporal_patch_size),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "transformers": transformers_version,
    }
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def mm_feature_path(*, run_id: str, fingerprint: str, modality: str, mm_hash: str) -> Path:
    if not _SAFE_FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError(f"Invalid multimodal feature fingerprint: {fingerprint!r}")
    if modality != "image":
        raise ValueError(f"Unsupported multimodal feature modality: {modality!r}")
    if not _SAFE_MM_HASH_RE.fullmatch(mm_hash):
        raise ValueError(f"Invalid multimodal feature hash: {mm_hash!r}")

    root = feature_asset_dir(run_id)
    path = (root / "v1" / "vllm-mmitem" / fingerprint / modality / mm_hash[:2] / f"{mm_hash}.msgpack").resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Multimodal feature path escaped root: {path}")
    return path


def mmfile_ref(*, run_id: str, fingerprint: str, modality: str, mm_hash: str) -> str:
    return f"{MMFILE_PREFIX}:{run_id}:{fingerprint}:{modality}:{mm_hash}"


def split_mmfile_ref(ref: str) -> "tuple[str | None, str, str, str]":
    """Inverse of :func:`mmfile_ref`: parse the ref SHAPE into
    ``(run_id_or_None, fingerprint, modality, mm_hash)``. ``run_id`` is ``None``
    for the legacy 5-part form (the caller supplies it from its own context).
    Raises ``ValueError`` on a bad prefix/version/arity. The ref field order
    lives here, next to the emitter — so emit and parse can't drift apart.
    Field-level validation (safe regexes, slot/fingerprint matching) is the
    caller's responsibility."""
    parts = ref.split(":")
    if parts[:2] != ["mmfile", "v1"] or len(parts) not in {5, 6}:
        raise ValueError(f"Invalid mmfile ref shape: {ref!r}")
    if len(parts) == 6:
        return parts[2], parts[3], parts[4], parts[5]
    return None, parts[2], parts[3], parts[4]


def _grid_to_ref(grid_thw: object) -> str:
    data = grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, (list, tuple)) or len(data) != 3:
        raise ValueError(f"Invalid image grid_thw for raw ref: {grid_thw!r}")
    return "x".join(str(int(v)) for v in data)


def _grid_from_ref(value: str) -> list[int]:
    if not _SAFE_GRID_THW_RE.fullmatch(value):
        raise ValueError(f"Invalid image grid_thw ref segment: {value!r}")
    return [int(v) for v in value.split("x")]


def raw_image_path(*, run_id: str, raw_image_id: str) -> Path:
    if not _SAFE_RAW_IMAGE_ID_RE.fullmatch(raw_image_id):
        raise ValueError(f"Invalid raw image id: {raw_image_id!r}")
    root = image_asset_dir(run_id)
    path = (root / raw_image_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Raw image path escaped root: {path}")
    return path


def mmraw_ref(
    *,
    run_id: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    raw_image_id: str,
    grid_thw: object,
) -> str:
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"Invalid raw multimodal run id: {run_id!r}")
    if not _SAFE_FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError(f"Invalid raw multimodal fingerprint: {fingerprint!r}")
    if modality != "image":
        raise ValueError(f"Unsupported raw multimodal modality: {modality!r}")
    if not _SAFE_MM_HASH_RE.fullmatch(mm_hash):
        raise ValueError(f"Invalid raw multimodal hash: {mm_hash!r}")
    raw_image_path(run_id=run_id, raw_image_id=raw_image_id)
    return f"{MMRAW_PREFIX}:{run_id}:{fingerprint}:{modality}:{mm_hash}:{raw_image_id}:{_grid_to_ref(grid_thw)}"


def split_mmraw_ref(ref: str) -> tuple[str, str, str, str, str, list[int]]:
    """Parse a run-scoped raw-image ref into
    ``(run_id, fingerprint, modality, mm_hash, raw_image_id, grid_thw)``.
    Field-level safe-regex checks live in the reader, but the grid segment is
    parsed here so emit/parse cannot drift.
    """
    parts = ref.split(":")
    if parts[:2] != ["mmraw", "v1"] or len(parts) != 8:
        raise ValueError(f"Invalid mmraw ref shape: {ref!r}")
    return parts[2], parts[3], parts[4], parts[5], parts[6], _grid_from_ref(parts[7])


def build_mm_feature_envelope(
    *,
    run_id: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    payload: bytes,
    placeholder_length: int,
) -> dict:
    """Envelope dict the writer packs (with the payload) into the msgpack file."""
    return {
        "schema_version": _MM_FEATURE_SCHEMA_VERSION,
        "kind": _MM_FEATURE_KIND,
        "run_id": run_id,
        "fingerprint": fingerprint,
        "modality": modality,
        "mm_hash": mm_hash,
        "placeholder_length": int(placeholder_length),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def mm_feature_envelope_matches(
    envelope: dict,
    *,
    run_id: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    payload: bytes,
    require_run_id: bool = True,
) -> bool:
    """Validate a parsed envelope against the requested artifact identity.

    ``require_run_id=False`` mirrors the reader's tolerance for envelopes that
    predate the ``run_id`` field (``envelope.get("run_id", run_id)``).
    """
    envelope_run_id = envelope.get("run_id") if require_run_id else envelope.get("run_id", run_id)
    return (
        envelope.get("schema_version") == _MM_FEATURE_SCHEMA_VERSION
        and envelope.get("kind") == _MM_FEATURE_KIND
        and envelope_run_id == run_id
        and envelope.get("fingerprint") == fingerprint
        and envelope.get("modality") == modality
        and envelope.get("mm_hash") == mm_hash
        and envelope.get("payload_sha256") == hashlib.sha256(payload).hexdigest()
    )


def sweep_stale_artifacts(run_dir: Path, ttl_seconds: float) -> int:
    """Delete stale ``assets/mm_features`` artifacts (the expensive processed
    ``MultiModalKwargsItem`` payloads, ~tens of MB each) whose mtime is older than
    ttl_seconds. Returns count deleted.

    Features ONLY — ``assets/images`` are never swept here. Features are a
    regenerable cache: the trainer rebuilds pixels from the source image
    (``materialize_pixels``) and never reads these files, and the env-worker
    rewrites any missing feature on demand (``force_full_pixels`` repair retry +
    write-if-missing). Source images, by contrast, are terminal browser output
    with no regeneration path, so they are retained for the whole run as the
    recoverable source of truth. Over-eviction of a feature is therefore safe
    (it just forces a reprocess); over-eviction of an image is NOT, which is why
    this sweep deliberately excludes ``IMAGE_ASSET_SUBDIR``.

    ttl_seconds only needs to exceed the write->vLLM-admit window (seconds), so
    any horizon of minutes leaves a huge safety margin against racing in-flight
    reads. No-op if the dir doesn't exist; ignore per-file errors (a file may be
    mid-write). Walk files only; leave dir structure."""
    import time

    cutoff = time.time() - ttl_seconds
    deleted = 0
    base = run_dir / FEATURE_ASSET_SUBDIR
    if not base.is_dir():
        return 0
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            # File may be mid-write or already gone; over-eviction is safe (the
            # feature is regenerable), so ignore and continue.
            continue
    return deleted
