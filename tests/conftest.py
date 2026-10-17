"""Shared fixtures for renderer tests.

Each (model_name, renderer_name) pair gets a tokenizer + renderer.
The same barrage of tests runs against every pair.

Modules with their own model caches release them in ``teardown_module``.
The module-scoped fixture below releases shared models and cyclic references.
"""

import gc
from pathlib import Path

import pytest
from parity import models_for
from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import config_from_name

# Backwards-compatible view used by documentation and a few invariant tests.
RENDERER_MODELS = [(case.model, case.renderer) for case in models_for("shared")]

_cache: dict[str, tuple] = {}


@pytest.fixture(scope="module", autouse=True)
def release_shared_models():
    """Keep model caches local to a test module, including cyclic HF objects."""
    yield
    _cache.clear()
    gc.collect()


def _load(model_name: str, renderer_name: str):
    key = f"{model_name}:{renderer_name}"
    if key not in _cache:
        tokenizer = load_tokenizer(model_name)
        renderer = create_renderer(tokenizer, config_from_name(renderer_name))
        _cache[key] = (tokenizer, renderer)
    return _cache[key]


def pytest_generate_tests(metafunc):
    if "model_name" in metafunc.fixturenames:
        filename = Path(str(metafunc.definition.path)).name
        if filename == "test_build_helpers.py":
            suite = "build-helpers"
        elif filename in {
            "test_parse_response.py",
            "test_parse_response_robustness.py",
        }:
            suite = "plain-parser"
        else:
            suite = "shared"
        cases = models_for(suite)
        metafunc.parametrize(
            "model_name,renderer_name",
            [(case.model, case.renderer) for case in cases],
            ids=[case.model for case in cases],
        )


@pytest.fixture
def tokenizer(model_name, renderer_name):
    t, _ = _load(model_name, renderer_name)
    return t


@pytest.fixture
def renderer(model_name, renderer_name):
    _, r = _load(model_name, renderer_name)
    return r
