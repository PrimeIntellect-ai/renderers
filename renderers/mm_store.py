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

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{16,64}$")
_SAFE_MM_HASH_RE = re.compile(r"^[a-f0-9]{16,128}$")

_MM_FEATURE_SCHEMA_VERSION = 1
_MM_FEATURE_KIND = "vllm.MultiModalKwargsItem"

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
    traversal guard checks against. Identical to ``mm_feature_run_root``.
    """
    return (run_dir(run_id) / FEATURE_ASSET_SUBDIR).resolve()


# Alias kept for symmetry with the feature-format function names; both resolve
# to ``<root>/run_<id>/assets/mm_features``.
mm_feature_run_root = feature_asset_dir


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


def mm_feature_path(*, run_id: str, fingerprint: str, modality: str, mm_hash: str) -> Path:
    if not _SAFE_FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError(f"Invalid multimodal feature fingerprint: {fingerprint!r}")
    if modality != "image":
        raise ValueError(f"Unsupported multimodal feature modality: {modality!r}")
    if not _SAFE_MM_HASH_RE.fullmatch(mm_hash):
        raise ValueError(f"Invalid multimodal feature hash: {mm_hash!r}")

    root = mm_feature_run_root(run_id)
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
    """Delete artifact files under run_dir/{assets/images, assets/mm_features} whose
    mtime is older than ttl_seconds. Returns count deleted. Safe by construction:
    artifacts are content-addressed and re-writable, so over-eviction just triggers
    re-materialization, never corruption. No-op if the dirs don't exist; ignore
    per-file errors (a file may be mid-write). Walk files only; leave dir structure."""
    import time

    cutoff = time.time() - ttl_seconds
    deleted = 0
    for subdir in (IMAGE_ASSET_SUBDIR, FEATURE_ASSET_SUBDIR):
        base = run_dir / subdir
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                # File may be mid-write or already gone; over/under-eviction is safe.
                continue
    return deleted
