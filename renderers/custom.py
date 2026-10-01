"""Load custom renderers from importable modules or local Python files.

An import path is ``my_module.Name`` (dotted, like any ``import``) or
``path/to/file.py:Name``. File paths are resolved against the working
directory and imported once per process under a stable module name, so every
caller gets the same class objects.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def load_target(import_path: str) -> Any:
    """Return the object that ``import_path`` names."""
    module_ref, sep, attr = import_path.rpartition(":")
    if not sep and not _is_file_ref(import_path):
        module_ref, sep, attr = import_path.rpartition(".")
    if not sep or not module_ref or not attr:
        raise ValueError(
            f"Import path {import_path!r} must look like 'my_module.Name' or 'path/to/file.py:Name'"
        )
    module = load_module(module_ref)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ValueError(f"Module {module_ref!r} has no attribute {attr!r}") from None


def load_module(module_ref: str) -> ModuleType:
    """Import a module by dotted name or by file path."""
    if _is_file_ref(module_ref):
        return _load_file_module(str(Path(module_ref).expanduser().resolve()))
    return importlib.import_module(module_ref)


def _is_file_ref(ref: str) -> bool:
    return ref.endswith(".py") or "/" in ref


@lru_cache(maxsize=None)
def _load_file_module(path: str) -> ModuleType:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Custom renderer file {path!r} does not exist")
    # A path-derived name keeps two files with the same name apart.
    digest = hashlib.sha1(path.encode()).hexdigest()[:8]
    name = f"renderers_custom_{file.stem}_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import custom renderer file {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_custom_renderer(import_path: str) -> Any:
    """Return the renderer class that ``import_path`` names.

    The class must set ``config_class`` to its own
    :class:`~renderers.configs.BaseRendererConfig` subclass.
    """
    from renderers.configs import BaseRendererConfig

    renderer_cls = load_target(import_path)
    config_cls = getattr(renderer_cls, "config_class", None)
    if not (
        isinstance(config_cls, type) and issubclass(config_cls, BaseRendererConfig)
    ):
        raise TypeError(
            f"Custom renderer {import_path!r} must set config_class to a BaseRendererConfig subclass"
        )
    return renderer_cls
