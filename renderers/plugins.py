"""Load renderer plugins from importable modules or local Python files.

A plugin target is ``package.module:Name`` or ``path/to/file.py:Name``. File
targets are resolved against the working directory and imported once per
process under a stable module name, so every caller gets the same class
objects.
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


def load_plugin_object(target: str) -> Any:
    """Return the object that ``target`` names."""
    module_ref, sep, attr = target.rpartition(":")
    if not sep or not module_ref or not attr:
        raise ValueError(
            f"Plugin target {target!r} must look like 'package.module:Name' or 'path/to/file.py:Name'"
        )
    module = load_plugin_module(module_ref)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ValueError(
            f"Plugin module {module_ref!r} has no attribute {attr!r}"
        ) from None


def load_plugin_module(module_ref: str) -> ModuleType:
    """Import a module by dotted name or by file path."""
    if module_ref.endswith(".py") or "/" in module_ref:
        return _load_file_module(str(Path(module_ref).expanduser().resolve()))
    return importlib.import_module(module_ref)


@lru_cache(maxsize=None)
def _load_file_module(path: str) -> ModuleType:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Renderer plugin file {path!r} does not exist")
    # A path-derived name keeps two plugins with the same file name apart.
    digest = hashlib.sha1(path.encode()).hexdigest()[:8]
    name = f"renderers_plugin_{file.stem}_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import renderer plugin file {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_plugin_renderer(target: str) -> Any:
    """Return the renderer class that ``target`` names.

    The class must set ``config_class`` to its own
    :class:`~renderers.configs.BaseRendererConfig` subclass.
    """
    from renderers.configs import BaseRendererConfig

    renderer_cls = load_plugin_object(target)
    config_cls = getattr(renderer_cls, "config_class", None)
    if not (
        isinstance(config_cls, type) and issubclass(config_cls, BaseRendererConfig)
    ):
        raise TypeError(
            f"Renderer plugin {target!r} must set config_class to a BaseRendererConfig subclass"
        )
    return renderer_cls
