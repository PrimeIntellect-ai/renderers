from __future__ import annotations

import pytest

from renderers import (
    FULLY_STABLE,
    OPAQUE,
    STABLE_IN_TOOL_CYCLE,
    create_renderer,
)
from renderers.base import Message
from renderers.glm5 import GLM5Renderer
from renderers.qwen3 import Qwen3Renderer
from renderers.qwen3_vl import Qwen3VLRenderer
from renderers.qwen35 import Qwen35Renderer
from renderers.qwen36 import Qwen36Renderer
from renderers.stability import Boundary

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

BOUNDARY_CASES: dict[Boundary, tuple[list[Message], Message]] = {
    "user": (
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Simple arithmetic.",
                "content": "4",
            },
        ],
        {"role": "user", "content": "Now answer 3+5."},
    ),
    "assistant": (
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        {
            "role": "assistant",
            "reasoning_content": "Simple arithmetic.",
            "content": "4",
        },
    ),
    "tool": (
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
        ],
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": '{"temp": 20}',
        },
    ),
}


def _has_known_dynamic_stability(renderer) -> bool:
    return isinstance(
        renderer,
        (Qwen3Renderer, Qwen3VLRenderer, Qwen35Renderer, GLM5Renderer),
    ) and not isinstance(renderer, Qwen36Renderer)


def _assert_declared_boundaries_preserve_prefix(renderer) -> None:
    for boundary in sorted(renderer.stability.preserves_through):
        messages, appended = BOUNDARY_CASES[boundary]
        before = renderer.render(messages, tools=TOOLS).token_ids
        after = renderer.render(messages + [appended], tools=TOOLS).token_ids
        assert after[: len(before)] == before, (
            f"{renderer.__class__.__name__}: appending {boundary!r} did not "
            "preserve the rendered prefix"
        )


def test_renderer_default_stability_values(renderer):
    if _has_known_dynamic_stability(renderer):
        assert renderer.stability == STABLE_IN_TOOL_CYCLE
    else:
        assert renderer.stability == OPAQUE


def test_preserve_all_thinking_makes_known_renderers_fully_stable(
    tokenizer, renderer_name, renderer
):
    if not _has_known_dynamic_stability(renderer):
        pytest.skip("renderer does not declare preserve_all_thinking stability yet")

    all_thinking_renderer = create_renderer(
        tokenizer,
        renderer=renderer_name,
        preserve_all_thinking=True,
    )

    assert all_thinking_renderer.stability == FULLY_STABLE


def test_declared_stability_boundaries_preserve_prefix(renderer):
    if not renderer.stability.preserves_through:
        pytest.skip("renderer declares opaque stability")

    _assert_declared_boundaries_preserve_prefix(renderer)


def test_preserve_all_declared_boundaries_preserve_prefix(
    tokenizer, renderer_name, renderer
):
    if not _has_known_dynamic_stability(renderer):
        pytest.skip("renderer does not declare preserve_all_thinking stability yet")

    all_thinking_renderer = create_renderer(
        tokenizer,
        renderer=renderer_name,
        preserve_all_thinking=True,
    )

    _assert_declared_boundaries_preserve_prefix(all_thinking_renderer)
