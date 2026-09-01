"""Structured reasoning is an input-schema field, not inline content markup."""

from __future__ import annotations

from functools import lru_cache

import pytest

from renderers import create_renderer
from renderers.base import get_structured_reasoning, load_tokenizer


_INLINE_CONTENT = "<think>INLINE_REASONING_SENTINEL</think>VISIBLE_ANSWER_SENTINEL"

# The DeepSeek V4 PR target plus one checkpoint for every renderer
# implementation that previously promoted inline string content into its
# structured reasoning channel. Inherited variants (Qwen3.6, GLM-5.1, Kimi
# K2.6) use the same implementation.
_MODELS = (
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-9B",
    "THUDM/GLM-4.5-Air",
    "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/Kimi-K2.5",
    "poolside/Laguna-M.1",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
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

    # Model-specific field precedence includes an explicit empty string.
    assert (
        get_structured_reasoning(
            {
                "reasoning": "",
                "reasoning_content": "lower-priority reasoning",
            },
            "reasoning",
            "reasoning_content",
        )
        == ""
    )


@pytest.mark.parametrize("model", _MODELS)
def test_inline_think_markup_stays_in_assistant_content(model: str):
    tokenizer, renderer = _load(model)
    rendered = renderer.render(
        [
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": _INLINE_CONTENT},
            {"role": "user", "content": "Second question."},
        ]
    )

    assistant_ids = [
        token_id
        for token_id, message_index in zip(
            rendered.token_ids, rendered.message_indices, strict=True
        )
        if message_index == 1
    ]
    assistant_text = tokenizer.decode(assistant_ids, skip_special_tokens=False)

    assert _INLINE_CONTENT in assistant_text, (
        f"{model} promoted or removed inline think markup instead of keeping "
        f"content opaque: {assistant_text!r}"
    )
