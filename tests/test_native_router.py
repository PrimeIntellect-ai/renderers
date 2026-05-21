"""Unit tests for the Python/native routing layer.

These are isolated from the inference engines and don't require a
network connection — they exercise just the env-var parsing, the
lazy import, and (where the wheel is built) the native module's
class surface.
"""

from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace
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


def test_resolve_tokenizer_path_rejects_hf_missing_sentinel(monkeypatch):
    tokenizer = SimpleNamespace(name_or_path="org/custom-tokenizer")
    fake_hf = SimpleNamespace(try_to_load_from_cache=lambda **_kwargs: object())
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(ValueError, match="tokenizer.json not available"):
        router.resolve_tokenizer_path(tokenizer)


def test_resolve_tokenizer_path_uses_tiktoken_export(monkeypatch, tmp_path):
    tokenizer = SimpleNamespace(name_or_path="moonshotai/Kimi-K2-Instruct")
    fake_hf = SimpleNamespace(try_to_load_from_cache=lambda **_kwargs: object())
    exported = tmp_path / "tokenizer.json"
    exported.write_text("{}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(
        router,
        "_export_tiktoken_tokenizer_json",
        lambda repo_id, _loader: str(exported)
        if repo_id == "moonshotai/Kimi-K2-Instruct"
        else None,
    )

    assert router.resolve_tokenizer_path(tokenizer) == str(exported)


def test_kimi_k2_constructor_falls_back_without_tokenizer_path(monkeypatch):
    from renderers.kimi_k2 import KimiK2Renderer

    fake_native = mock.Mock()
    monkeypatch.setattr("renderers.kimi_k2.native_enabled", lambda _family: True)
    monkeypatch.setattr("renderers.kimi_k2.load_native", lambda: fake_native)
    monkeypatch.setattr(
        "renderers.kimi_k2.try_resolve_tokenizer_path",
        lambda _tokenizer, _family: None,
    )

    inst = KimiK2Renderer.__new__(KimiK2Renderer, object())

    assert isinstance(inst, KimiK2Renderer)
    fake_native.Renderer.kimi_k2.assert_not_called()


def test_kimi_k25_constructor_does_not_route_eagerly(monkeypatch):
    from renderers.kimi_k25 import KimiK25Renderer

    fake_native = mock.Mock()
    monkeypatch.setattr("renderers.kimi_k25.native_enabled", lambda _family: True)
    monkeypatch.setattr("renderers.kimi_k25.load_native", lambda: fake_native)

    inst = KimiK25Renderer.__new__(KimiK25Renderer, object(), processor=None)

    assert isinstance(inst, KimiK25Renderer)
    fake_native.Renderer.kimi_k25.assert_not_called()


def test_kimi_k25_native_delegate_rejects_render_time_tools():
    from renderers.kimi_k25 import KimiK25Renderer

    inst = object.__new__(KimiK25Renderer)
    inst._native_renderer = object()

    assert inst._can_use_native([{"role": "user", "content": "hi"}], tools=None)
    assert not inst._can_use_native(
        [{"role": "user", "content": "hi"}],
        tools=[{"name": "echo", "parameters": {}}],
    )


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


def test_native_base_api_surface(native):
    renderer_methods = [
        "render",
        "render_ids",
        "parse_response",
        "get_stop_token_ids",
        "bridge_to_next_turn",
    ]
    rendered_tokens_attrs = [
        "token_ids",
        "message_indices",
        "sampled_mask",
        "is_content",
        "message_roles",
        "multi_modal_data",
        "tokens_per_message",
        "message_token_spans",
        "role_token_spans",
        "tokens_by_role",
        "content_token_spans_by_role",
        "content_mask_for_roles",
    ]

    for name in renderer_methods:
        assert hasattr(native.Renderer, name), f"missing Renderer.{name}"
    for name in rendered_tokens_attrs:
        assert hasattr(native.RenderedTokens, name), f"missing RenderedTokens.{name}"

    assert "tools" in inspect.signature(native.Renderer.parse_response).parameters
