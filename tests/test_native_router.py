"""Unit tests for the Python/native routing layer.

These are isolated from the inference engines and don't require a
network connection — they exercise just the env-var parsing, the
lazy import, and (where the wheel is built) the native module's
class surface.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from renderers import _native_router as router


def test_native_disabled_by_default():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not router.native_enabled("qwen3")


@pytest.mark.parametrize("value", ["", "0"])
def test_native_off_values(value):
    with mock.patch.dict(os.environ, {"RENDERERS_NATIVE": value}, clear=True):
        assert not router.native_enabled("qwen3")


@pytest.mark.parametrize("value", ["1", "all"])
def test_native_on_global(value):
    with mock.patch.dict(os.environ, {"RENDERERS_NATIVE": value}, clear=True):
        assert router.native_enabled("qwen3")
        assert router.native_enabled("qwen35")
        assert router.native_enabled("glm5")


def test_native_csv_specific_families():
    with mock.patch.dict(
        os.environ, {"RENDERERS_NATIVE": "qwen3,glm5"}, clear=True
    ):
        assert router.native_enabled("qwen3")
        assert router.native_enabled("glm5")
        assert not router.native_enabled("qwen35")


def test_native_csv_whitespace_tolerant():
    with mock.patch.dict(
        os.environ, {"RENDERERS_NATIVE": " qwen3 , glm5 "}, clear=True
    ):
        assert router.native_enabled("qwen3")
        assert router.native_enabled("glm5")


def test_load_native_caches_result():
    # Reset the loader cache for the test.
    router._NATIVE_MODULE = None
    router._NATIVE_LOAD_ATTEMPTED = False
    first = router.load_native()
    second = router.load_native()
    assert first is second  # cached


def test_resolve_tokenizer_path_from_string(tmp_path):
    # Pass a directory containing tokenizer.json — get the file path back.
    (tmp_path / "tokenizer.json").write_text("{}")
    assert router.resolve_tokenizer_path(str(tmp_path)).endswith("tokenizer.json")


def test_resolve_tokenizer_path_from_exact_file(tmp_path):
    f = tmp_path / "tokenizer.json"
    f.write_text("{}")
    # Pass a file path directly — return as-is.
    assert router.resolve_tokenizer_path(str(f)) == str(f)


# ── Native module surface (only runs when the wheel is built) ────────


@pytest.fixture
def native():
    mod = router.load_native()
    if mod is None:
        pytest.skip("renderers_native not built; run `maturin develop`")
    return mod


def test_native_exports(native):
    # The five classes the Python shim relies on.
    for name in (
        "Renderer",
        "RenderedTokens",
        "ParsedResponse",
        "ParsedToolCall",
        "ToolCallParseStatus",
    ):
        assert hasattr(native, name), f"missing {name}"


def test_native_status_constants(native):
    s = native.ToolCallParseStatus
    assert s.OK == "ok"
    assert s.INVALID_JSON == "invalid_json"
    assert s.UNCLOSED_BLOCK == "unclosed_block"
    assert s.MISSING_NAME == "missing_name"
    assert s.MALFORMED_STRUCTURE == "malformed_structure"
