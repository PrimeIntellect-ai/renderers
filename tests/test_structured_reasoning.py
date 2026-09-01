"""Structured reasoning is an input-schema field, not inline content markup."""

from __future__ import annotations

from functools import lru_cache

import pytest

from renderers import create_renderer
from renderers.base import (
    get_structured_reasoning,
    load_tokenizer,
    validate_canonical_messages,
)


_INLINE_CONTENT = "<think>INLINE_REASONING_SENTINEL</think>VISIBLE_ANSWER_SENTINEL"

# The DeepSeek V4 PR target plus one checkpoint for every renderer
# implementation that previously promoted inline string content into its
# structured reasoning channel. Inherited variants (Qwen3.6, GLM-5.1, Kimi
# K2.6) use the same implementation.
_REASONING_MODELS = (
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-9B",
    "THUDM/GLM-4.5-Air",
    "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/Kimi-K2.5",
    "poolside/Laguna-M.1",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
)

_UNSUPPORTED_MODELS = (
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-VL-4B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
)


@lru_cache(maxsize=None)
def _load(model: str):
    tokenizer = load_tokenizer(model)
    return tokenizer, create_renderer(tokenizer)


def test_structured_reasoning_helper_never_reads_content():
    assert get_structured_reasoning({"content": _INLINE_CONTENT}) == ""
    assert (
        get_structured_reasoning(
            {
                "content": _INLINE_CONTENT,
                "reasoning_content": "structured reasoning",
            }
        )
        == "structured reasoning"
    )


@pytest.mark.parametrize("model", _REASONING_MODELS)
def test_inline_think_markup_is_rejected(model: str):
    _, renderer = _load(model)
    assert renderer.supports_reasoning_content is True
    with pytest.raises(ValueError, match="normalize legacy.*reasoning_content"):
        renderer.render(
            [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": _INLINE_CONTENT},
            ]
        )


@pytest.mark.parametrize("model", _UNSUPPORTED_MODELS)
def test_non_reasoning_renderers_reject_structured_reasoning(model: str):
    _, renderer = _load(model)
    assert renderer.supports_reasoning_content is False
    with pytest.raises(ValueError, match="does not support structured reasoning"):
        renderer.render(
            [
                {"role": "user", "content": "Question."},
                {
                    "role": "assistant",
                    "reasoning_content": "private",
                    "content": "Answer.",
                },
            ]
        )


@pytest.mark.parametrize(
    "message,match",
    [
        (
            {"role": "assistant", "reasoning": "legacy", "content": "answer"},
            "non-canonical 'reasoning' field",
        ),
        (
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "legacy"},
                    {"type": "text", "text": "answer"},
                ],
            },
            "non-canonical 'thinking' content part",
        ),
    ],
)
def test_non_canonical_structured_shapes_are_rejected(message, match):
    with pytest.raises(ValueError, match=match):
        validate_canonical_messages(
            [message],
            supports_reasoning_content=True,
            renderer_name="TestRenderer",
        )
