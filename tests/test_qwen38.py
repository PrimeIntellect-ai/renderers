"""Qwen3.8 renderer coverage.

Locks in (a) the Qwen3.8 entries in ``MODEL_RENDERER_MAP`` and
``MULTIMODAL_MODELS``, (b) the thinking-polarity default, and (c) byte
parity of ``Qwen38Renderer`` against the model's own
``apply_chat_template`` across the template's new knobs —
``reasoning_effort`` (xhigh / medium / low) and ``preserve_thinking`` —
on top of the Qwen3.5 surface the shared barrage already exercises.
"""

from __future__ import annotations

import pytest

from renderers import Qwen38Renderer, Qwen38RendererConfig, create_renderer
from renderers.base import MODEL_RENDERER_MAP, MULTIMODAL_MODELS, load_tokenizer

QWEN38 = "Qwen/Qwen3.8-27B"

HISTORY = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
    {"role": "user", "content": "q2"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_map_routes_qwen38_to_qwen38_renderer():
    assert MODEL_RENDERER_MAP.get(QWEN38) == "qwen3.8"


def test_qwen38_registered_as_multimodal():
    assert QWEN38 in MULTIMODAL_MODELS


def test_qwen38_thinking_default_on():
    """Qwen3.8 ships thinking on by default (open `` thinking\n`` at the
    gen-prompt boundary)."""
    tok = load_tokenizer(QWEN38)
    renderer = create_renderer(tok, Qwen38RendererConfig())
    assert isinstance(renderer, Qwen38Renderer)
    assert renderer.config.enable_thinking is True


def test_config_defaults_match_template():
    cfg = Qwen38RendererConfig()
    assert cfg.preserve_thinking is True
    assert cfg.reasoning_effort is None  # template default xhigh


# ---------------------------------------------------------------------------
# Byte parity against apply_chat_template
# ---------------------------------------------------------------------------


def _render_and_compare(tok, kwargs, messages, tools=None, add_generation_prompt=True):
    config = Qwen38RendererConfig(**kwargs)
    renderer = create_renderer(tok, config)
    rendered = renderer.render_ids(
        messages, tools=tools, add_generation_prompt=add_generation_prompt
    )
    template_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    reference = tok.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **template_kwargs,
    )["input_ids"]
    assert rendered == reference


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"reasoning_effort": "low"},
        {"reasoning_effort": "medium"},
        {"reasoning_effort": "xhigh"},
        {"preserve_thinking": False},
        {"enable_thinking": False},
        {"enable_thinking": False, "reasoning_effort": "low"},
        {"enable_thinking": False, "preserve_thinking": False},
    ],
)
def test_render_parity_knobs(kwargs):
    tok = load_tokenizer(QWEN38)
    _render_and_compare(tok, kwargs, HISTORY)


def test_render_parity_tools_cycle():
    tok = load_tokenizer(QWEN38)
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "reasoning_content": "let me check",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "Sunny, 22C"},
        {
            "role": "assistant",
            "reasoning_content": "done",
            "content": "Sunny in Paris.",
        },
    ]
    _render_and_compare(tok, {}, messages, tools=TOOLS)


def test_render_parity_system_message():
    tok = load_tokenizer(QWEN38)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    for kwargs in (
        {},
        {"reasoning_effort": "low"},
        {"reasoning_effort": "medium"},
        {"reasoning_effort": "xhigh"},
    ):
        _render_and_compare(tok, kwargs, messages)
