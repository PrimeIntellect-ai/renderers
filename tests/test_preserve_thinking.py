"""Targeted coverage for the ``thinking_retention`` bridge-policy flag.

Generic ``thinking_retention`` controls whether a renderer may append via
``bridge_to_next_turn`` or should fall back to a full re-render. It is not a
chat-template kwarg, so the only full-render contract here is that explicit
generic values do not change render output.
"""

from __future__ import annotations

import pytest

from renderers import create_renderer
from renderers.base import MODEL_RENDERER_MAP
from renderers.configs import _config_class_for


CONVERSATION = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "I should call the weather tool for Paris.",
        "content": "Let me check.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "name": "get_weather", "content": "Sunny, 22C"},
    {
        "role": "assistant",
        "reasoning_content": "The tool returned the weather.",
        "content": "Sunny, 22C in Paris.",
    },
    {"role": "user", "content": "And Berlin?"},
]


def _make(tokenizer, renderer_name, **flags):
    """Build a fresh renderer with the given construction-time flags."""
    if renderer_name == "auto":
        renderer_name = MODEL_RENDERER_MAP.get(
            getattr(tokenizer, "name_or_path", ""), "default"
        )
    config = _config_class_for(renderer_name)(**flags)
    return create_renderer(tokenizer, config)


def test_generic_thinking_retention_does_not_change_full_render(
    model_name, tokenizer, renderer_name, renderer
):
    """Generic retention is bridge policy only, not a render override."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on explicit retention — covered separately")

    default = renderer.render_ids(CONVERSATION)
    for retention in ("tool_cycle", "all"):
        explicit = _make(
            tokenizer,
            renderer_name,
            thinking_retention=retention,
        ).render_ids(CONVERSATION)
        assert explicit == default, (
            f"{model_name}: thinking_retention={retention!r} changed full render"
        )


def test_no_thinking_knob_implies_all_bridge_policy(
    model_name, renderer_name, tokenizer
):
    """No-thinking generation config means there is no thinking to evict."""
    from renderers.default import DefaultRenderer

    bare = _make(tokenizer, renderer_name)
    if isinstance(bare, DefaultRenderer):
        pytest.skip("DefaultRenderer has no typed no-thinking bridge policy")

    cfg_cls = _config_class_for(renderer_name)
    template_fields = cfg_cls.template_field_names()
    if "enable_thinking" in template_fields:
        no_thinking = {"enable_thinking": False}
    elif "thinking" in template_fields:
        no_thinking = {"thinking": False}
    else:
        pytest.skip(f"{model_name}: no no-thinking generation knob")

    all_on = _make(tokenizer, renderer_name, **no_thinking)
    assert all_on.effective_thinking_retention == "all"

    conservative = _make(
        tokenizer,
        renderer_name,
        **no_thinking,
        thinking_retention="tool_cycle",
    )
    assert conservative.effective_thinking_retention == "tool_cycle"
