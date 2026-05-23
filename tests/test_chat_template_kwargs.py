from types import SimpleNamespace

import pytest

from renderers import base


class _FakeRenderer:
    CHAT_TEMPLATE_KWARGS = frozenset({"enable_thinking"})

    def __init__(
        self,
        tokenizer,
        *,
        enable_thinking: bool = True,
        preserve_all_thinking: bool = False,
        preserve_thinking_between_tool_calls: bool = False,
    ):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.preserve_all_thinking = preserve_all_thinking
        self.preserve_thinking_between_tool_calls = (
            preserve_thinking_between_tool_calls
        )


def test_create_renderer_forwards_model_chat_template_kwargs(monkeypatch):
    monkeypatch.setitem(base.RENDERER_REGISTRY, "fake-qwen", _FakeRenderer)

    renderer = base.create_renderer(
        SimpleNamespace(name_or_path="unused"),
        renderer="fake-qwen",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert renderer.enable_thinking is False


def test_create_renderer_rejects_unsupported_model_chat_template_kwargs(monkeypatch):
    monkeypatch.setitem(base.RENDERER_REGISTRY, "fake-qwen", _FakeRenderer)

    with pytest.raises(ValueError, match="reasoning_effort"):
        base.create_renderer(
            SimpleNamespace(name_or_path="unused"),
            renderer="fake-qwen",
            chat_template_kwargs={"reasoning_effort": "high"},
        )


def test_create_renderer_auto_forwards_model_chat_template_kwargs(monkeypatch):
    monkeypatch.setitem(base.RENDERER_REGISTRY, "fake-qwen", _FakeRenderer)
    monkeypatch.setitem(base.MODEL_RENDERER_MAP, "fake/model", "fake-qwen")

    renderer = base.create_renderer(
        SimpleNamespace(name_or_path="fake/model"),
        chat_template_kwargs={"enable_thinking": False},
    )

    assert renderer.enable_thinking is False
