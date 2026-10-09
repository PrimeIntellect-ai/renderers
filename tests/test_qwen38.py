"""Focused Qwen3.8 registration and upstream-template parity coverage."""

from __future__ import annotations

from functools import lru_cache
from copy import deepcopy

import pytest
from pydantic import TypeAdapter

from renderers import (
    Qwen38Renderer,
    Qwen38RendererConfig,
    RendererConfig,
    create_renderer,
)
from renderers.base import MODEL_RENDERER_MAP, MULTIMODAL_MODELS, load_tokenizer


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

QWEN38_MODELS = ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next"]


@lru_cache(maxsize=None)
def _qwen38(model_name):
    tokenizer = load_tokenizer(model_name)
    return tokenizer, create_renderer(tokenizer)


def _expected(tokenizer, messages, **kwargs):
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        **kwargs,
    )
    return list(result)


@pytest.mark.parametrize("qwen38_model", QWEN38_MODELS)
def test_qwen38_is_registered_with_native_defaults(qwen38_model):
    tokenizer, renderer = _qwen38(qwen38_model)

    assert tokenizer.name_or_path == qwen38_model
    assert MODEL_RENDERER_MAP[tokenizer.name_or_path] == "qwen3.8"
    assert MULTIMODAL_MODELS[tokenizer.name_or_path] == {"image"}
    assert isinstance(renderer, Qwen38Renderer)
    assert renderer.config.enable_thinking is True
    assert renderer.config.reasoning_effort == "xhigh"
    assert renderer.config.preserve_thinking is True
    assert renderer.effective_thinking_retention == "all"


def test_qwen38_config_discriminator():
    parsed = TypeAdapter(RendererConfig).validate_python(
        {
            "name": "qwen3.8",
            "reasoning_effort": "low",
            "preserve_thinking": False,
        }
    )

    assert isinstance(parsed, Qwen38RendererConfig)
    assert parsed.reasoning_effort == "low"
    assert parsed.preserve_thinking is False


@pytest.mark.parametrize(
    "config_kwargs",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"reasoning_effort": "xhigh"}, id="xhigh"),
        pytest.param({"reasoning_effort": "medium"}, id="medium"),
        pytest.param({"reasoning_effort": "low"}, id="low"),
        pytest.param({"enable_thinking": False}, id="thinking-disabled"),
        pytest.param({"preserve_thinking": False}, id="drop-history-thinking"),
    ],
)
@pytest.mark.parametrize("qwen38_model", QWEN38_MODELS)
def test_qwen38_text_and_tool_parity(config_kwargs, qwen38_model):
    tokenizer, _ = _qwen38(qwen38_model)
    renderer = Qwen38Renderer(tokenizer, Qwen38RendererConfig(**config_kwargs))
    cases = [
        (
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello."},
            ],
            {"add_generation_prompt": True},
        ),
        (
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Hello."},
            ],
            {"add_generation_prompt": True},
        ),
        (
            [
                {"role": "user", "content": "First question."},
                {
                    "role": "assistant",
                    "reasoning_content": "First thought.",
                    "content": "First answer.",
                },
                {"role": "user", "content": "Second question."},
                {
                    "role": "assistant",
                    "reasoning_content": "Second thought.",
                    "content": "Second answer.",
                },
            ],
            {},
        ),
        (
            [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris", "fresh": True},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature": 20}'},
            ],
            {"tools": TOOLS, "add_generation_prompt": True},
        ),
    ]

    for messages, render_kwargs in cases:
        expected = _expected(
            tokenizer,
            messages,
            **config_kwargs,
            **render_kwargs,
        )
        assert renderer.render_ids(messages, **render_kwargs) == expected


@pytest.mark.parametrize("qwen38_model", QWEN38_MODELS)
def test_qwen38_rejects_inline_think_markup(qwen38_model):
    _, renderer = _qwen38(qwen38_model)
    messages = [
        {"role": "user", "content": "Echo this."},
        {
            "role": "assistant",
            "content": "<think>literal markup</think>visible content",
        },
    ]

    with pytest.raises(ValueError, match="normalize legacy.*reasoning_content"):
        renderer.render_ids(messages)


@pytest.mark.parametrize("qwen38_model", QWEN38_MODELS)
def test_qwen38_requires_a_real_user_query(qwen38_model):
    _, renderer = _qwen38(qwen38_model)

    with pytest.raises(ValueError, match="No user query found"):
        renderer.render_ids([{"role": "system", "content": "System only."}])


@pytest.mark.parametrize(
    "role,with_image",
    [
        ("system", False),
        ("user", False),
        ("assistant", False),
        ("tool", False),
        ("user", True),
        ("tool", True),
    ],
)
def test_qwen38_structured_content_render_and_bridge(role, with_image):
    _, renderer = _qwen38(QWEN38_MODELS[0])
    parts = ["Hello", {"type": "text", "text": " world"}]
    if with_image:
        parts.append({"type": "image", "image": "unused", "text": 123})
        parts.append({"type": "text", "text": "After image"})
    noisy_parts = [
        {"type": "unsupported"},
        *parts,
        {"type": "text", "text": None},
        {"type": "text", "text": 123},
        None,
    ]
    prior = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    clean = {"role": role, "content": parts}
    noisy = {"role": role, "content": noisy_parts}
    original = deepcopy(noisy)
    messages = [noisy, *prior] if role == "system" else [*prior, noisy]
    expected_messages = [clean, *prior] if role == "system" else [*prior, clean]
    assert renderer.render(messages, process_multimodal=False) == renderer.render(
        expected_messages, process_multimodal=False
    )
    if role in {"user", "tool"}:
        previous = renderer.render_ids(prior)
        actual = renderer.bridge_to_next_turn(
            previous, [], [noisy], process_multimodal=False
        )
        assert actual is not None
        assert actual == renderer.bridge_to_next_turn(
            previous, [], [clean], process_multimodal=False
        )
    assert noisy == original


def test_qwen38_structured_tool_response_keeps_current_reasoning():
    tokenizer, _ = _qwen38(QWEN38_MODELS[0])
    renderer = Qwen38Renderer(tokenizer, Qwen38RendererConfig(preserve_thinking=False))
    prior = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "reasoning_content": "Keep me", "content": ""},
    ]
    content = [
        {"type": "text", "text": "  <tool_response>"},
        {"type": "unsupported"},
        {"type": "text", "text": None},
        "result",
        {"type": "text", "text": "</tool_response>  "},
    ]
    messages = [*prior, {"role": "user", "content": content}]
    expected = [
        *prior,
        {"role": "user", "content": "<tool_response>result</tool_response>"},
    ]
    assert renderer.render(messages) == renderer.render(expected)
    assert "Keep me" in tokenizer.decode(renderer.render_ids(messages))
    with pytest.raises(ValueError, match="No user query found"):
        renderer.render([messages[-1]])
