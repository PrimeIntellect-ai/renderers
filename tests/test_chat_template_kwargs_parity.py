"""Parity for ``chat_template_kwargs`` against the upstream chat template.

The renderer accepts a per-renderer allowlist of template-control kwargs
declared as ``CHAT_TEMPLATE_KWARGS`` on each renderer class and forwards
them to the constructor. ``test_chat_template_kwargs.py`` covers the
wiring; this file covers the only thing that matters downstream: that
flipping a kwarg produces token streams byte-identical to
``tokenizer.apply_chat_template(messages, chat_template_kwargs={...})``.

Without this, the kwarg surface is a promise the renderer doesn't keep.

Discovery is automatic — the parity matrix is built from each renderer
class's ``CHAT_TEMPLATE_KWARGS`` frozenset crossed with the per-kwarg
value list in ``_KWARG_VALUES``. To extend coverage to a new kwarg:
declare it in the renderer's ``CHAT_TEMPLATE_KWARGS`` and add the
values to exercise to ``_KWARG_VALUES`` below.

``gpt-oss`` parity is against ``openai-harmony`` (its renderer diverges
from HF Jinja by design — see ``test_gpt_oss_harmony_parity.py``); it
lives in its own test below, with the same auto-derived discovery.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import pytest

from renderers import create_renderer
from renderers.base import (
    MODEL_RENDERER_MAP,
    RENDERER_REGISTRY,
    _populate_registry,
    load_tokenizer,
)


# Models exercised by the parity tests. Mirrors ``conftest.RENDERER_MODELS``
# in spirit — one representative model per renderer family — plus the
# ``gpt-oss`` entry that conftest skips for HF parity (gpt-oss parity is
# against harmony, handled separately below).
_RENDERER_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),
    ("Qwen/Qwen3.5-9B", "auto"),
    ("Qwen/Qwen3.6-35B-A3B", "auto"),
    ("zai-org/GLM-5", "auto"),
    ("zai-org/GLM-5.1", "auto"),
    ("zai-org/GLM-4.7-Flash", "auto"),
    ("THUDM/GLM-4.5-Air", "auto"),
    ("moonshotai/Kimi-K2.5", "auto"),
    ("moonshotai/Kimi-K2.6", "auto"),
    ("deepseek-ai/DeepSeek-V3", "auto"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "auto"),
    ("poolside/Laguna-XS.2", "auto"),
    ("openai/gpt-oss-20b", "gpt-oss"),
]


# Per-kwarg value list. Each kwarg that appears in any renderer's
# ``CHAT_TEMPLATE_KWARGS`` must have an entry here, or the parity matrix
# silently skips it. The test below asserts coverage so a future kwarg
# can't slip through without an explicit value list.
_KWARG_VALUES: dict[str, list[Any]] = {
    "enable_thinking": [True, False],
    # Kimi K2.5 / K2.6 — same semantics as ``enable_thinking`` but the
    # upstream template uses ``thinking`` as the variable name. The
    # renderer mirrors that name so passing ``thinking`` via
    # ``chat_template_kwargs`` flows straight through to the template's
    # gate.
    "thinking": [True, False],
    "reasoning_effort": ["low", "medium", "high"],
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    }
]


# (id, messages, render_kwargs). Each shape exercises a distinct branch
# at least one renderer is known to flip on a kwarg:
#
# - ``system_user_gen``: forces the generation-prompt branch (e.g.
#   Qwen3 / GLM / Nemotron emit a synthesized ``<think></think>`` here
#   under ``enable_thinking=False``).
# - ``single_turn``: terminal assistant with plain content — the
#   "render historical thinking?" branch.
# - ``with_reasoning``: assistant carries ``reasoning_content`` — flips
#   whether thinking is emitted into the rendered history.
# - ``multi_turn``: two assistant turns separated by a user; the
#   in-flight-vs-historical distinction matters for several renderers.
# - ``tool_cycle``: assistant tool call + tool response + final
#   assistant, with ``add_generation_prompt=True`` so the second
#   gen-prompt branch is hit too.
_MESSAGE_SHAPES = [
    (
        "system_user_gen",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        {"add_generation_prompt": True},
    ),
    (
        "single_turn",
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        {},
    ),
    (
        "with_reasoning",
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Simple arithmetic.",
                "content": "4",
            },
        ],
        {},
    ),
    (
        "multi_turn",
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ],
        {},
    ),
    (
        "tool_cycle",
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
            {"role": "assistant", "content": "It is 20 degrees."},
        ],
        {"tools": TOOLS, "add_generation_prompt": True},
    ),
]


# ── Matrix discovery ───────────────────────────────────────────────────


_populate_registry()


def _resolve_renderer_cls(model: str, renderer_name: str) -> type:
    """Resolve ``(model, renderer_name)`` to the concrete renderer class."""
    if renderer_name == "auto":
        renderer_name = MODEL_RENDERER_MAP.get(model, "default")
    return RENDERER_REGISTRY[renderer_name]


def _hf_parity_matrix() -> list[Any]:
    """Auto-derived ``(model, renderer_name, kwarg, value)`` matrix for
    every renderer that declares a kwarg in ``CHAT_TEMPLATE_KWARGS``,
    minus gpt-oss (handled separately against harmony).
    """
    out = []
    for model, name in _RENDERER_MODELS:
        cls = _resolve_renderer_cls(model, name)
        if name == "gpt-oss":
            continue
        for kwarg in sorted(getattr(cls, "CHAT_TEMPLATE_KWARGS", ())):
            for value in _KWARG_VALUES.get(kwarg, []):
                out.append(
                    pytest.param(
                        model, name, kwarg, value, id=f"{model}-{kwarg}={value}"
                    )
                )
    return out


def _harmony_parity_matrix() -> list[Any]:
    """Auto-derived ``(model, renderer_name, kwarg, value)`` matrix for
    gpt-oss (parity against ``openai-harmony``).
    """
    out = []
    for model, name in _RENDERER_MODELS:
        if name != "gpt-oss":
            continue
        cls = _resolve_renderer_cls(model, name)
        for kwarg in sorted(getattr(cls, "CHAT_TEMPLATE_KWARGS", ())):
            for value in _KWARG_VALUES.get(kwarg, []):
                out.append(
                    pytest.param(
                        model, name, kwarg, value, id=f"{model}-{kwarg}={value}"
                    )
                )
    return out


def test_kwarg_values_covers_every_declared_kwarg():
    """Every kwarg any renderer declares in ``CHAT_TEMPLATE_KWARGS`` must
    have an entry in ``_KWARG_VALUES`` — otherwise it silently drops out
    of parity coverage.
    """
    declared: set[str] = set()
    for model, name in _RENDERER_MODELS:
        cls = _resolve_renderer_cls(model, name)
        declared.update(getattr(cls, "CHAT_TEMPLATE_KWARGS", ()))
    missing = sorted(declared - _KWARG_VALUES.keys())
    assert not missing, (
        f"chat_template_kwargs declared but not covered: {missing}. "
        f"Add a value list to _KWARG_VALUES in this file."
    )


# ── Test caches ────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _tokenizer(model_name: str):
    return load_tokenizer(model_name)


@lru_cache(maxsize=None)
def _renderer_with_kwarg(model_name: str, renderer_name: str, kwarg: str, value: Any):
    tok = _tokenizer(model_name)
    return create_renderer(
        tok,
        renderer=renderer_name,
        chat_template_kwargs={kwarg: value},
    )


def _expected_hf(tokenizer, messages, *, kwarg: str, value: Any, **render_kwargs):
    """Render via ``apply_chat_template`` with the kwarg spread as a
    top-level argument.

    transformers v5.x silently drops ``chat_template_kwargs={...}`` —
    only direct kwargs propagate into the Jinja environment. The two
    invocation styles are semantically the same for the Jinja template,
    so we pick the one that actually fires. (Our ``create_renderer``
    API accepts the dict form because it is the standard wire format
    in OpenAI-compatible servers; we translate it to constructor kwargs
    on our side.)
    """
    render_kwargs.setdefault("add_generation_prompt", False)
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        **{kwarg: value},
        **render_kwargs,
    )
    if isinstance(result, dict):
        return list(result["input_ids"])
    if isinstance(result, str):
        return list(tokenizer.encode(result, add_special_tokens=False))
    return list(result)


# ── HF-Jinja parity (every renderer except gpt-oss) ────────────────────


@pytest.mark.parametrize("model,renderer_name,kwarg,value", _hf_parity_matrix())
@pytest.mark.parametrize(
    "shape_id,messages,render_kwargs",
    _MESSAGE_SHAPES,
    ids=[s[0] for s in _MESSAGE_SHAPES],
)
def test_chat_template_kwarg_parity_hf(
    model,
    renderer_name,
    kwarg,
    value,
    shape_id,
    messages,
    render_kwargs,
):
    tokenizer = _tokenizer(model)
    renderer = _renderer_with_kwarg(model, renderer_name, kwarg, value)
    # Guard: the renderer must actually declare the kwarg. ``create_renderer``
    # already enforces this at construction; asserting here gives a louder
    # failure on a future renderer subclass that drops the declaration.
    assert kwarg in getattr(type(renderer), "CHAT_TEMPLATE_KWARGS", frozenset())

    try:
        expected = _expected_hf(
            tokenizer, messages, kwarg=kwarg, value=value, **render_kwargs
        )
    except Exception as exc:
        pytest.xfail(
            f"{model}: apply_chat_template raised {type(exc).__name__}: "
            f"{str(exc)[:160]}"
        )

    got = renderer.render_ids(messages, **render_kwargs)
    assert got == expected, (
        f"{model} / shape={shape_id} / {kwarg}={value}: renderer diverged "
        f"from apply_chat_template (len got={len(got)}, expected={len(expected)})"
    )


# ── Harmony parity (gpt-oss only) ──────────────────────────────────────


_DATE_FOR_PARITY = datetime.now().strftime("%Y-%m-%d")


@lru_cache(maxsize=None)
def _gpt_oss_renderer(kwarg: str, value: Any):
    from renderers.gpt_oss import GptOssRenderer

    tok = _tokenizer("openai/gpt-oss-20b")
    # Pin the conversation start date so the rendered preamble matches
    # the harmony oracle built with the same fixed date.
    return GptOssRenderer(
        tok,
        conversation_start_date=_DATE_FOR_PARITY,
        **{kwarg: value},
    )


def _harmony_expected(kwarg: str, value: Any, messages: list[dict[str, Any]]) -> list[int]:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message as HarmonyMessage,
        ReasoningEffort,
        Role,
        SystemContent,
        load_harmony_encoding,
    )

    sys_content = SystemContent.new().with_conversation_start_date(_DATE_FOR_PARITY)
    if kwarg == "reasoning_effort":
        effort_enum = {
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }[value]
        sys_content = sys_content.with_reasoning_effort(effort_enum)
    else:
        raise AssertionError(
            f"Harmony oracle: unhandled gpt-oss chat_template_kwarg {kwarg!r}. "
            "Add a branch here when extending CHAT_TEMPLATE_KWARGS on GptOssRenderer."
        )

    harmony_msgs: list[HarmonyMessage] = [
        HarmonyMessage.from_role_and_content(Role.SYSTEM, sys_content)
    ]
    for m in messages:
        role = m["role"]
        content = m.get("content", "") or ""
        if role == "user":
            harmony_msgs.append(HarmonyMessage.from_role_and_content(Role.USER, content))
        elif role == "assistant":
            harmony_msgs.append(
                HarmonyMessage.from_role_and_content(Role.ASSISTANT, content).with_channel("final")
            )
        else:
            raise AssertionError(
                f"Harmony oracle helper does not handle role={role!r}; add a "
                "branch or constrain the shapes used for gpt-oss parity."
            )
    encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoder.render_conversation_for_training(Conversation.from_messages(harmony_msgs))


# Harmony oracle is only wired for the simplest shapes (user-only and
# user+assistant content). Tool-call and reasoning_content shapes have
# a richer mapping that the dedicated ``test_gpt_oss_harmony_parity.py``
# already covers — duplicating that here would only test the harness.
_HARMONY_SHAPES = [
    (
        "user_only_gen",
        [{"role": "user", "content": "Hello!"}],
        {},
    ),
    (
        "user_and_assistant",
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        {},
    ),
]


@pytest.mark.parametrize("model,renderer_name,kwarg,value", _harmony_parity_matrix())
@pytest.mark.parametrize(
    "shape_id,messages,render_kwargs",
    _HARMONY_SHAPES,
    ids=[s[0] for s in _HARMONY_SHAPES],
)
def test_chat_template_kwarg_parity_harmony(
    model,
    renderer_name,
    kwarg,
    value,
    shape_id,
    messages,
    render_kwargs,
):
    renderer = _gpt_oss_renderer(kwarg, value)
    assert kwarg in getattr(type(renderer), "CHAT_TEMPLATE_KWARGS", frozenset())

    got = renderer.render_ids(messages, **render_kwargs)
    expected = _harmony_expected(kwarg, value, messages)
    assert got == expected, (
        f"{model} / shape={shape_id} / {kwarg}={value}: renderer diverged "
        f"from harmony oracle (len got={len(got)}, expected={len(expected)})"
    )
