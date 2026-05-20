"""Routing layer between the pure-Python renderers and the Rust port.

Loaded by each family shim (currently ``renderers.qwen3``). Resolves
whether the native module is available and, if so, whether the caller
opted into it for this family via the ``RENDERERS_NATIVE`` env var.

The env-var accepts:

- ``0`` / empty / unset — use the pure-Python implementation (default).
- ``1`` / ``all`` — route every supported family to the native module.
- comma-separated list of family names, e.g. ``qwen3`` or
  ``qwen3,qwen35`` — route only those families.

Family detection is opt-in per family so callers can roll out the
native path one model at a time; everything else falls back to Python
verbatim.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("renderers._native_router")

_NATIVE_MODULE: Any | None = None
_NATIVE_LOAD_ATTEMPTED = False
_ALL_EXCLUDED = {"default"}


def native_enabled(family: str) -> bool:
    """Should *family* route to the native module?"""
    raw = os.environ.get("RENDERERS_NATIVE", "").strip()
    if not raw or raw == "0":
        return False
    if raw in {"1", "all"}:
        return family not in _ALL_EXCLUDED
    return family in {part.strip() for part in raw.split(",") if part.strip()}


def load_native() -> Any | None:
    """Import ``renderers_native`` lazily. Returns ``None`` if the
    extension module is not installed (caller falls back to Python).

    Kept as a top-level distribution (rather than `renderers._native`)
    so the maturin-built wheel doesn't collide with the hatchling-built
    `renderers` wheel at install time.
    """
    global _NATIVE_MODULE, _NATIVE_LOAD_ATTEMPTED
    if _NATIVE_LOAD_ATTEMPTED:
        return _NATIVE_MODULE
    _NATIVE_LOAD_ATTEMPTED = True
    try:
        import renderers_native  # type: ignore[import-not-found]

        _NATIVE_MODULE = renderers_native
    except ImportError as exc:
        logger.info(
            "RENDERERS_NATIVE is set but the native extension is not "
            "available (%s); falling back to pure Python. Build it with "
            "`maturin develop --manifest-path crates/renderers-py/Cargo.toml`.",
            exc,
        )
        _NATIVE_MODULE = None
    return _NATIVE_MODULE


def resolve_tokenizer_path(tokenizer: Any) -> str:
    """Return a filesystem path to ``tokenizer.json`` for *tokenizer*.

    Accepts either:

    - a string (already a path / HF model id) — the caller is
      responsible for snapshotting the model first if it's a remote id.
    - a HuggingFace ``PreTrainedTokenizerBase`` — pulls
      ``name_or_path`` and locates ``tokenizer.json`` next to it.
    """
    if isinstance(tokenizer, (str, os.PathLike)):
        path = Path(tokenizer)
        if path.is_dir():
            return str(path / "tokenizer.json")
        return str(path)

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        data = backend.to_str()
        digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
        cache_dir = Path(tempfile.gettempdir()) / "renderers-tokenizers"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{digest}.json"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(path)
        return str(path)

    name_or_path = getattr(tokenizer, "name_or_path", None)
    if not name_or_path:
        raise ValueError(
            "Cannot determine tokenizer.json path: tokenizer has no "
            "name_or_path attribute. Pass an explicit path string instead."
        )

    candidate = Path(name_or_path)
    if candidate.is_dir():
        path = candidate / "tokenizer.json"
        if path.exists():
            return str(path)

    # HF cache fallback: <HF cache>/models--name--with--slashes/snapshots/<sha>/tokenizer.json
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
    except ImportError:
        raise ValueError(
            f"tokenizer.json not found near {name_or_path}; install "
            "huggingface_hub or pass an explicit path."
        )

    cached = try_to_load_from_cache(repo_id=name_or_path, filename="tokenizer.json")
    if not isinstance(cached, (str, os.PathLike)):
        raise ValueError(
            f"tokenizer.json not available in the local HF cache for {name_or_path}. "
            "Run `snapshot_download` first or pass an explicit path."
        )
    return str(cached)


def try_resolve_tokenizer_path(tokenizer: Any, family: str) -> str | None:
    """Best-effort tokenizer resolution for optional native routing."""
    try:
        return resolve_tokenizer_path(tokenizer)
    except ValueError as exc:
        logger.info(
            "RENDERERS_NATIVE selected %s but no native tokenizer path was "
            "available (%s); falling back to pure Python.",
            family,
            exc,
        )
        return None
