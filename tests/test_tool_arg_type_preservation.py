"""Regression: XML-style tool parsers lose string type for JSON-looking values.

When the model emits a tool call whose argument is a string that happens
to look like JSON (e.g. ``"true"``, ``"42"``, ``"[1,2,3]"``), every
XML-style parser (Qwen3.5, GLM, MiniMax, Laguna) re-parses the value via
``json.loads`` and returns a bool / int / list instead of the original
string. The chat template's ``<arg_value>X</arg_value>`` form has no
quoting to distinguish the two on the wire, so the tool schema is the
only disambiguating signal — and the renderer parsers don't see it.

vLLM / SGLang's reference parsers (e.g. ``vllm/glm45_tool_parser.py``)
overlay the tool schema to apply ``json.loads`` ONLY to parameters whose
declared type is not ``string``. The renderers library currently does
not; the JSON-/section-style parsers (Qwen3 hermes, Kimi K2, DeepSeek)
sidestep the bug because their wire format quotes strings.

These tests fail loudly against current main — they're the
specification for the fix, not documentation of accepted behavior. CI
should be red until ``parse_response`` is schema-aware.

Originally raised by Robin (Poolside) on PR #21:

    > Tool call parsing for all XML-like parsers may corrupt string
    > parameter values if they happen to be valid JSON strings, e.g.
    > "true" -> True (str -> bool), same for lists, objs etc. vLLM
    > parsers "solve" this by passing the tools definition into the
    > parsers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pytest


# (HuggingFace model name, renderer name). Two controls (JSON-shaped
# parsers that already preserve string types) + four XML-style parsers
# that currently corrupt them.
_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),  # hermes JSON  — control, expected to pass
    ("moonshotai/Kimi-K2-Instruct", "auto"),  # section JSON — control, expected to pass
    ("Qwen/Qwen3.5-9B", "auto"),  # XML — currently fails
    ("zai-org/GLM-5", "auto"),  # XML — currently fails
    ("MiniMaxAI/MiniMax-M2.5", "auto"),  # XML — currently fails
    ("poolside/Laguna-XS.2", "auto"),  # XML — currently fails
]


@lru_cache(maxsize=None)
def _load(model: str, renderer_name: str):
    from renderers import create_renderer
    from renderers.base import load_tokenizer

    tok = load_tokenizer(model)
    return tok, create_renderer(tok, renderer=renderer_name)


def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        metafunc.parametrize(
            "model,renderer_name",
            _MODELS,
            ids=[m for m, _ in _MODELS],
        )


@pytest.fixture
def renderer(model, renderer_name):
    return _load(model, renderer_name)[1]


PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Call f."},
]


# Each entry: a single-key arguments dict whose value is a STRING that
# happens to be valid JSON of another type. The bug surfaces when the
# parser silently coerces the string into the encoded type.
JSON_LOOKING_STRING_ARGS = [
    pytest.param({"flag": "true"}, id="string-bool"),
    pytest.param({"n": "42"}, id="string-int"),
    pytest.param({"x": "null"}, id="string-null"),
    pytest.param({"x": "[1,2,3]"}, id="string-array"),
    pytest.param({"x": '{"k": 1}'}, id="string-object"),
]


def _normalize_args(args: Any) -> Any:
    """Mirror ``test_roundtrip._normalize_args`` — some parsers return a
    JSON string, others a dict; compare by value."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def _extract_assistant_tokens(renderer, prompt, assistant_msg):
    prompt_ids = renderer.render_ids(prompt, add_generation_prompt=False)
    full_ids = renderer.render_ids(prompt + [assistant_msg])
    return full_ids[len(prompt_ids) :]


@pytest.mark.parametrize("args", JSON_LOOKING_STRING_ARGS)
def test_string_arg_preserves_type(model, renderer_name, renderer, args):
    """Tool-call args of declared type ``str`` must round-trip as ``str``,
    not get re-parsed as bool/int/null/list/dict by the parser."""
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "functions.f:0",
                "function": {"name": "f", "arguments": args},
            }
        ],
    }
    completion_ids = _extract_assistant_tokens(renderer, PROMPT, msg)
    parsed = renderer.parse_response(completion_ids)

    assert parsed.tool_calls, f"{model}: parser returned no tool_calls"
    got = _normalize_args(parsed.tool_calls[0].arguments)
    assert got == args, (
        f"{model}: tool-arg type drift — sent {args!r}, parser returned {got!r}"
    )
