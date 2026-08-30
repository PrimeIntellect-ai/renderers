"""Renderer-wide contract tests for BYO tokenizers without character offsets."""

from __future__ import annotations

from typing import Any

import pytest

from renderers import create_renderer
from renderers.base import load_tokenizer


class OffsetlessTokenizer:
    """Delegate the basic tokenizer protocol without exposing ``__call__``."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        if name == "__call__":
            raise AttributeError(name)
        return getattr(self._tokenizer, name)

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        return self._tokenizer.encode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return self._tokenizer.decode(*args, **kwargs)

    def convert_tokens_to_ids(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer.apply_chat_template(*args, **kwargs)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


CASES = [
    pytest.param(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello!"},
        ],
        None,
        True,
        id="system-generation-prompt",
    ),
    pytest.param(
        [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "Weather?"},
        ],
        TOOLS,
        False,
        id="system-and-tools",
    ),
    pytest.param(
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Simple arithmetic",
                "content": "4",
            },
            {"role": "user", "content": "And 3+3?"},
        ],
        None,
        True,
        id="reasoning-history",
    ),
    pytest.param(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "assistant", "content": "It is 20 degrees."},
        ],
        TOOLS,
        False,
        id="full-tool-cycle",
    ),
]


def _offsetless_renderer(renderer: Any) -> Any:
    return type(renderer)(OffsetlessTokenizer(renderer._tokenizer), renderer.config)


def _assert_offsetless_contract(expected: Any, actual: Any, renderer: Any) -> None:
    assert actual.token_ids == expected.token_ids
    assert actual.message_indices == expected.message_indices
    if type(renderer).__name__ == "PrimeQwen3Renderer":
        assert actual.sampled_mask == []
    else:
        assert actual.sampled_mask == expected.sampled_mask
    assert actual.is_content == []
    assert actual.message_roles == expected.message_roles
    assert actual.message_tool_names == expected.message_tool_names
    assert actual.multi_modal_data == expected.multi_modal_data


@pytest.mark.parametrize("messages,tools,add_generation_prompt", CASES)
def test_offsetless_contract_across_renderer_matrix(
    model_name: str,
    renderer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
) -> None:
    """Every configured renderer preserves tokens and non-content metadata."""

    expected = renderer.render(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
    )
    offsetless = _offsetless_renderer(renderer)
    actual = offsetless.render(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
    )

    _assert_offsetless_contract(expected, actual, renderer)


def test_offsetless_contract_across_renderer_bridges(
    model_name: str,
    renderer: Any,
) -> None:
    """Bridge paths preserve their tokens and metadata without offsets."""

    prior_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    new_messages = [{"role": "user", "content": "Tell me more."}]
    prior = renderer.render(prior_messages)
    offsetless = _offsetless_renderer(renderer)

    expected = renderer.bridge_to_next_turn(
        prior.token_ids,
        [],
        new_messages,
    )
    actual = offsetless.bridge_to_next_turn(
        prior.token_ids,
        [],
        new_messages,
    )

    assert (actual is None) == (expected is None)
    if expected is not None and actual is not None:
        _assert_offsetless_contract(expected, actual, renderer)


def test_hy3_offsetless_preserves_multiple_system_message_indices() -> None:
    """Hy3's joined system header must not erase either caller message."""

    tokenizer = load_tokenizer("tencent/Hy3")
    renderer = create_renderer(tokenizer)
    messages = [
        {"role": "system", "content": "First instruction."},
        {"role": "system", "content": "Second instruction."},
        {"role": "user", "content": "Hello"},
    ]

    expected = renderer.render(messages, tools=TOOLS)
    actual = _offsetless_renderer(renderer).render(messages, tools=TOOLS)

    _assert_offsetless_contract(expected, actual, renderer)
    assert {0, 1}.issubset(actual.message_indices)


@pytest.mark.parametrize(
    "deepseek_model",
    ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"],
)
def test_deepseek_offsetless_contract(deepseek_model: str) -> None:
    """DeepSeek variants live outside the shared parity matrix."""

    tokenizer = load_tokenizer(deepseek_model)
    renderer = create_renderer(tokenizer)
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]

    expected = renderer.render(messages, add_generation_prompt=True)
    actual = _offsetless_renderer(renderer).render(
        messages,
        add_generation_prompt=True,
    )

    _assert_offsetless_contract(expected, actual, renderer)
