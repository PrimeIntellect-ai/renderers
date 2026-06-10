"""Exhaustive token-for-token parity for the Nemotron-3 renderer.

The shared barrage in ``test_render_ids.py`` covers the common message
shapes against every model. This file pins the Nemotron-3-specific template
branches that the shared matrix can't reach — they'd fail on other models or
exercise behaviour unique to the Nemotron-3 chat template:

* reasoning + empty / ``None`` content with and without tool calls (the
  template trims the assembled ``<think>…</think>{content}`` block and appends
  exactly one separator — a stray ``\\n`` here is the most common agentic
  regression);
* the historical-thinking truncation boundary, which is ``loop.index0 <
  last_user_idx`` in **all three** variants (Nano / Super / Ultra) — so an
  in-flight tool cycle (assistant turns after the last user message) keeps its
  reasoning by default;
* inline ``<think>…</think>`` tags carried in ``content`` rendering verbatim
  (the template only reformats reasoning supplied via ``reasoning_content``);
* verbatim (unstripped) user / system / tool content and ``reasoning_content``;
* the ``enable_thinking`` / ``truncate_history_thinking`` template kwargs;
* the per-variant reasoning-effort kwargs: ``low_effort`` (Super) and
  ``medium_effort`` (Ultra), each a no-op on the variants that don't define it.

Every assertion compares ``renderer.render_ids(...)`` to
``tokenizer.apply_chat_template(..., tokenize=True)`` — a pass means the
renderer is byte-for-byte faithful for that case. Tokenizers are loaded from
the local HF cache (offline); no network.

The variants split across two configs: ``nemotron-3`` (Nano / Super, with
``low_effort``) and ``nemotron-3-ultra`` (Ultra, with ``medium_effort``). The
helper resolves the right config class per model from ``MODEL_RENDERER_MAP``.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from renderers import create_renderer
from renderers.base import MODEL_RENDERER_MAP, load_tokenizer
from renderers.configs import _config_class_for

# BF16 / FP8 share a tokenizer; only the BF16 checkpoints are cached for tests.
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
ULTRA = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODELS = [NANO, SUPER, ULTRA]


@lru_cache
def _tok(model: str):
    return load_tokenizer(model)


def _config_cls(model: str):
    """The typed-config class the model resolves to (``nemotron-3`` for
    Nano/Super, ``nemotron-3-ultra`` for Ultra)."""
    return _config_class_for(MODEL_RENDERER_MAP[model])


def _renderer(model: str, **flags):
    # Build with the model's own variant config so the renderer picks the right
    # ``</think>`` glue (and only valid kwargs are accepted).
    return create_renderer(_tok(model), _config_cls(model)(**flags))


def _expected(
    model: str, messages, *, tools=None, add_generation_prompt=False, **kwargs
):
    out = _tok(model).apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if isinstance(out, str):  # some tokenizers return str even with tokenize=True
        return list(_tok(model).encode(out, add_special_tokens=False))
    return list(out)


def _assert_parity(
    model, messages, *, tools=None, add_generation_prompt=False, **template_kwargs
):
    """Renderer ids == apply_chat_template ids for ``model``.

    ``template_kwargs`` (e.g. ``enable_thinking``, ``truncate_history_thinking``)
    are forwarded to both the renderer config and ``apply_chat_template`` so the
    two sides stay aligned.
    """
    renderer = _renderer(model, **template_kwargs)
    got = renderer.render_ids(
        messages, tools=tools, add_generation_prompt=add_generation_prompt
    )
    exp = _expected(
        model,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    assert got == exp, (
        f"{model}: render_ids diverged from apply_chat_template\n"
        f"  exp …{_tok(model).decode(exp[-40:])!r}\n"
        f"  got …{_tok(model).decode(got[-40:])!r}"
    )


pytestmark = pytest.mark.parametrize("model", MODELS, ids=["nano", "super", "ultra"])


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"],
            },
        },
    }
]


# ── Reasoning + tool calls: the trim / separator boundary ─────────────


def test_reasoning_empty_content_tool_call(model):
    """reason → tool call, no prose. Must be ``</think>\\n<tool_call>`` (one
    newline), not ``</think>\\n\\n<tool_call>``."""
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "reasoning_content": "I should call the weather tool.",
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
        ],
        tools=TOOLS,
    )


def test_reasoning_none_content_tool_call(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "reasoning_content": "Call the tool.",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        }
                    }
                ],
            },
        ],
        tools=TOOLS,
    )


def test_reasoning_content_tool_call(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "reasoning_content": "Think first.",
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
        ],
        tools=TOOLS,
    )


def test_reasoning_empty_content_no_tool_call(model):
    """reason → empty answer, no tool call: ``</think>`` glued to ``<|im_end|>``."""
    _assert_parity(
        model,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "reasoning_content": "thinking", "content": ""},
        ],
    )


def test_multiple_tool_calls_with_reasoning(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Weather in Paris and London?"},
            {
                "role": "assistant",
                "reasoning_content": "Two cities — two calls.",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        }
                    },
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        }
                    },
                ],
            },
        ],
        tools=TOOLS,
    )


def test_tool_call_with_nested_object_args(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "Paris",
                                "opts": {"unit": "c", "days": [1, 2]},
                            },
                        }
                    }
                ],
            },
        ],
        tools=TOOLS,
    )


# ── Historical-thinking truncation boundary (last_user_idx) ───────────


def test_inflight_tool_cycle_keeps_reasoning(model):
    """Assistant turns after the last user message (the in-flight tool cycle)
    keep their reasoning by default — boundary is ``loop.index0 <
    last_user_idx`` in every variant."""
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "reasoning_content": "Call the tool first.",
                "content": "calling",
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
            {
                "role": "assistant",
                "reasoning_content": "Now I can answer.",
                "content": "It is 20 degrees.",
            },
        ],
        tools=TOOLS,
    )


def test_history_truncation_drops_older_reasoning(model):
    """A reasoning turn before the last user message is collapsed to an empty
    think block (tool-call branch trims the remainder on Nano/Super)."""
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "reasoning_content": "reasoning before tool",
                "content": "calling",
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
            {"role": "assistant", "reasoning_content": "after", "content": "Done."},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "reasoning_content": "final", "content": "A2"},
        ],
        tools=TOOLS,
    )


def test_two_block_tool_conversation(model):
    _assert_parity(
        model,
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "reasoning_content": "R2",
                "content": "calling.",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "a"}}}
                ],
            },
            {"role": "tool", "content": "result-a"},
            {"role": "assistant", "reasoning_content": "R4", "content": "answer-1"},
            {"role": "user", "content": "second"},
            {
                "role": "assistant",
                "reasoning_content": "R6",
                "content": "calling.",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "b"}}}
                ],
            },
            {"role": "tool", "content": "result-b"},
            {"role": "assistant", "reasoning_content": "R8", "content": "answer-2"},
        ],
        tools=TOOLS,
    )


def test_plain_multi_turn_reasoning_truncation(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "reasoning_content": "long reasoning one",
                "content": "A1",
            },
            {"role": "user", "content": "Q2"},
            {
                "role": "assistant",
                "reasoning_content": "long reasoning two",
                "content": "A2",
            },
        ],
    )


@pytest.mark.parametrize("truncate", [True, False])
def test_truncate_history_thinking_kwarg(model, truncate):
    """``truncate_history_thinking=False`` keeps reasoning on every past turn."""
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "reasoning_content": "first reasoning",
                "content": "A1",
            },
            {"role": "user", "content": "Q2"},
            {
                "role": "assistant",
                "reasoning_content": "second reasoning",
                "content": "A2",
            },
        ],
        truncate_history_thinking=truncate,
    )


# ── Inline <think> tags carried in content (no reasoning_content field) ─


def test_inline_think_tags_final_turn_verbatim(model):
    """Inline ``<think>…</think>`` in the final assistant ``content`` renders
    verbatim — the renderer must not parse + reformat it."""
    _assert_parity(
        model,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>secret</think>visible"},
        ],
    )


def test_inline_think_tags_history_turn(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "<think>secret reasoning</think>visible answer",
            },
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "second"},
        ],
    )


# ── Verbatim (unstripped) content ─────────────────────────────────────


def test_system_content_whitespace_verbatim(model):
    _assert_parity(
        model,
        [
            {"role": "system", "content": "  padded system  "},
            {"role": "user", "content": "hi"},
        ],
    )


def test_user_content_whitespace_verbatim(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "  padded user  "},
            {"role": "assistant", "content": "ok"},
        ],
    )


def test_assistant_content_whitespace(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "  spaced answer  "},
        ],
    )


def test_reasoning_content_whitespace_verbatim(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "  padded reason  ",
                "content": "answer",
            },
        ],
    )


def test_tool_content_whitespace_verbatim(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
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
            {"role": "tool", "content": "  spaced tool result  "},
            {"role": "assistant", "content": "done"},
        ],
        tools=TOOLS,
    )


def test_system_whitespace_with_tools(model):
    _assert_parity(
        model,
        [
            {"role": "system", "content": "  weather bot  "},
            {"role": "user", "content": "Weather?"},
        ],
        tools=TOOLS,
    )


# ── Generation prompt / thinking toggle ───────────────────────────────


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_generation_prompt_thinking_toggle(model, enable_thinking):
    _assert_parity(
        model,
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def test_generation_prompt_after_tool_response(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
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
            {"role": "tool", "content": '{"temp": 20}'},
        ],
        tools=TOOLS,
        add_generation_prompt=True,
    )


# ── Whole-conversation cycles, no reasoning ───────────────────────────


def test_full_tool_cycle_no_reasoning(model):
    _assert_parity(
        model,
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
            {"role": "tool", "content": '{"temp": 20, "condition": "sunny"}'},
            {"role": "assistant", "content": "It is 20 degrees and sunny."},
        ],
        tools=TOOLS,
    )


def test_consecutive_tool_responses(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Weather in Paris and London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        }
                    },
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        }
                    },
                ],
            },
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "tool", "content": '{"temp": 15}'},
            {"role": "assistant", "content": "Paris: 20, London: 15."},
        ],
        tools=TOOLS,
    )


def test_no_system_no_tools_injects_empty_system(model):
    _assert_parity(
        model,
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    )


# ── Reasoning-effort kwargs (variant-specific) ────────────────────────

_EFFORT_SHAPES = [
    # gen-prompt shape: hint rides on the (only) user message.
    ([{"role": "user", "content": "solve it"}], {"add_generation_prompt": True}),
    # multi-turn: hint must land on the LAST user message, not the first.
    (
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ],
        {"add_generation_prompt": True},
    ),
]


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize(
    "shape,extra", _EFFORT_SHAPES, ids=["gen_prompt", "multi_turn"]
)
def test_low_effort_kwarg(model, flag, shape, extra):
    """``low_effort`` appends ``\\n\\n{reasoning effort: low}`` to the last user
    message on **Super**; it's a no-op on **Nano** (its template never defines
    it). Ultra's config has no such field, so it's skipped."""
    if model == ULTRA:
        pytest.skip("low_effort is a nemotron-3 (Nano/Super) kwarg")
    _assert_parity(model, shape, low_effort=flag, **extra)


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize(
    "shape,extra", _EFFORT_SHAPES, ids=["gen_prompt", "multi_turn"]
)
def test_medium_effort_kwarg(model, flag, shape, extra):
    """``medium_effort`` appends ``\\n\\n{reasoning effort: efficient}`` on
    **Ultra**. Nano/Super configs have no such field, so they're skipped."""
    if model != ULTRA:
        pytest.skip("medium_effort is a nemotron-3-ultra kwarg")
    _assert_parity(model, shape, medium_effort=flag, **extra)


def test_effort_kwarg_lives_on_the_right_variant(model):
    """Each effort kwarg is declared only on the variant whose template defines
    it — the discriminated union rejects the wrong combination at config load."""
    fields = _config_cls(model).template_field_names()
    if model == ULTRA:
        assert "medium_effort" in fields and "low_effort" not in fields
    else:
        assert "low_effort" in fields and "medium_effort" not in fields
