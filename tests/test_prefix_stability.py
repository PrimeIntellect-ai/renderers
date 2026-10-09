"""Declared full-render stability must agree with token prefixes."""

from functools import lru_cache

import pytest
from parity import TOOLS, models_for
from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import _config_class_for


_STABLE_DEFAULTS = {
    "prime-qwen3",
    "qwen3-vl",
    "qwen3.8",
    "glm-5.3",
    "inkling",
    "kimi-k2",
    "laguna-xs.2",
    "laguna-m.1",
    "laguna-xs-2.1",
    "laguna-s-2.1",
    "llama-3",
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-v4",
}
_VARIANTS = {
    "qwen3": [
        ({"enable_thinking": False}, False),
        ({"thinking_retention": "all"}, False),
    ],
    "qwen3.5": [({"enable_thinking": False}, False)],
    "qwen3.6": [({"preserve_thinking": True}, True)],
    "qwen3.8": [({"preserve_thinking": False}, False)],
    "glm-5": [({"clear_thinking": False}, True), ({"enable_thinking": False}, False)],
    "glm-5.1": [
        ({"clear_thinking": False}, False),
        ({"clear_thinking": False, "enable_thinking": False}, True),
    ],
    "glm-5.3": [({"clear_thinking": True}, False)],
    "gpt-oss": [({"auto_drop_analysis": False}, False)],
    "gemma4": [
        ({"preserve_thinking": True}, False),
        ({"enable_thinking": False}, False),
    ],
    "hy3": [
        ({"preserved_thinking": True}, True),
        ({"is_training": True}, True),
        ({"preserved_thinking": True, "raw_last_assistant": True}, False),
    ],
    "kimi-k2.5": [({"thinking": False}, False)],
    "deepseek-v4": [
        ({"enable_thinking": True}, False),
        ({"enable_thinking": True, "drop_thinking": False}, True),
    ],
    "nemotron-3": [
        ({"truncate_history_thinking": False}, True),
        ({"enable_thinking": False}, False),
    ],
    "nemotron-3-ultra": [
        ({"truncate_history_thinking": False}, True),
        ({"truncate_history_thinking": False, "medium_effort": True}, False),
    ],
    "nemotron-3.5": [({"truncate_history_thinking": False}, True)],
}


def _cases():
    for case in models_for("shared"):
        name = case.resolved_renderer
        variants = [({}, name in _STABLE_DEFAULTS), *_VARIANTS.get(name, [])]
        if "Nemotron-3-Super" in case.model:
            variants.append(
                ({"truncate_history_thinking": False, "low_effort": True}, False)
            )
        for kwargs, stable in variants:
            yield pytest.param(
                case.model, name, kwargs, stable, id=f"{case.model}-{kwargs}"
            )


@lru_cache(maxsize=None)
def _tokenizer(model):
    return load_tokenizer(model)


def _conversations():
    for reasoning in ("First reasoning.", ""):
        assistant = {
            "role": "assistant",
            "content": "First answer.",
            "reasoning_content": reasoning,
        }
        yield (
            [
                {"role": "user", "content": "First question?"},
                assistant,
                {"role": "user", "content": "Second question?"},
                {
                    "role": "assistant",
                    "content": "Second answer.",
                    "reasoning_content": "Second reasoning.",
                },
            ],
            None,
        )
        yield (
            [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    **assistant,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_weather",
                    "tool_call_id": "call_0",
                    "content": "Sunny.",
                },
                {
                    "role": "assistant",
                    "content": "It is sunny.",
                    "reasoning_content": "Summarize.",
                },
                {"role": "user", "content": "Thanks. What next?"},
                {
                    "role": "assistant",
                    "content": "Go outside.",
                    "reasoning_content": "Suggest an activity.",
                },
            ],
            TOOLS,
        )


@pytest.mark.parametrize("model, name, kwargs, stable", _cases())
def test_prefix_stability_matches_rendering(model, name, kwargs, stable):
    config_cls = _config_class_for(name)
    template_kwargs = {
        k: v for k, v in kwargs.items() if k in config_cls.template_field_names()
    }
    internal_kwargs = {k: v for k, v in kwargs.items() if k not in template_kwargs}
    renderer = create_renderer(
        _tokenizer(model),
        config_cls(**internal_kwargs),
        chat_template_kwargs=template_kwargs,
    )
    assert renderer.is_prefix_stable is stable
    if name == "default":
        return  # Opaque means unknown, even if these particular inputs are stable.
    changed = False
    for messages, tools in _conversations():
        for end, message in enumerate(messages[:-1], 1):
            if message["role"] != "assistant":
                continue
            before = renderer.render_ids(messages[:end], tools=tools)
            for extended_end in range(end + 1, len(messages) + 1):
                after = renderer.render_ids(messages[:extended_end], tools=tools)
                extends = after[: len(before)] == before
                if stable:
                    assert extends, (model, kwargs, end, extended_end)
                changed |= not extends
    if not stable:
        assert changed, (
            "Exercise a counterexample for every known-unstable configuration"
        )


def test_stability_uses_auto_resolved_template_kwargs():
    tokenizer = _tokenizer("Qwen/Qwen3.8-27B")
    assert create_renderer(tokenizer).is_prefix_stable is True
    assert (
        create_renderer(
            tokenizer, chat_template_kwargs={"preserve_thinking": False}
        ).is_prefix_stable
        is False
    )
