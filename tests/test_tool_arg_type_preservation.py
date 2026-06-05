"""Tool-arg string-type preservation across renderers.

XML-style chat templates (Qwen3.5, GLM, MiniMax, Laguna) render tool-call
argument values verbatim inside ``<arg_value>X</arg_value>`` tags with
no quoting, so a value of ``true`` could be either a bool or the string
``"true"``. Without the tool schema, the parser has no signal to choose
between them and defaults to ``json.loads`` — which silently corrupts
string args that look like JSON.

This test passes the tool schema to ``parse_response`` so the parser can
preserve declared-string params verbatim, matching vLLM / SGLang's
reference parsers (e.g. ``vllm/glm45_tool_parser.py``). The hermes-JSON
(Qwen3) and section-JSON (Kimi K2) parsers sidestep the bug because
their wire format quotes strings; both serve as controls.

Originally raised by Robin (Poolside) on PR #21.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pytest


# (HuggingFace model name, renderer name). Two JSON-shaped controls
# (string types already preserved by the wire format) + five XML-style
# parsers that rely on the schema to preserve them.
_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),  # hermes JSON  — control
    ("moonshotai/Kimi-K2-Instruct", "auto"),  # section JSON — control
    ("Qwen/Qwen3.5-9B", "auto"),  # XML
    ("zai-org/GLM-5", "auto"),  # XML
    ("MiniMaxAI/MiniMax-M2.5", "auto"),  # XML
    ("poolside/Laguna-XS.2", "auto"),  # XML
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "auto"),  # XML
]


@lru_cache(maxsize=None)
def _load(model: str, renderer_name: str):
    from renderers import config_from_name, create_renderer
    from renderers.base import load_tokenizer

    tok = load_tokenizer(model)
    return tok, create_renderer(tok, config_from_name(renderer_name))


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


# Each case: a single ``x`` argument whose value is a STRING that
# happens to be valid JSON of another type. With a schema declaring
# ``x: string``, the parser must return the string verbatim.
JSON_LOOKING_STRING_ARGS = [
    pytest.param({"x": "true"}, id="string-bool"),
    pytest.param({"x": "42"}, id="string-int"),
    pytest.param({"x": "null"}, id="string-null"),
    pytest.param({"x": "[1,2,3]"}, id="string-array"),
    pytest.param({"x": '{"k": 1}'}, id="string-object"),
]


# Tool schema with the single string-typed parameter exercised by every
# case above. Passed to both ``render`` (so the system prompt declares
# the tool) and ``parse_response`` (so the parser preserves the type).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "description": "Test tool with one string parameter.",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
    }
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


def _extract_assistant_tokens(renderer, prompt, assistant_msg, *, tools=None):
    prompt_ids = renderer.render_ids(prompt, tools=tools, add_generation_prompt=False)
    full_ids = renderer.render_ids(prompt + [assistant_msg], tools=tools)
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
    completion_ids = _extract_assistant_tokens(renderer, PROMPT, msg, tools=TOOLS)
    parsed = renderer.parse_response(completion_ids, tools=TOOLS)

    assert parsed.tool_calls, f"{model}: parser returned no tool_calls"
    got = _normalize_args(parsed.tool_calls[0].arguments)
    assert got == args, (
        f"{model}: tool-arg type drift — sent {args!r}, parser returned {got!r}"
    )


# Schemas where ``string`` is one branch of a union (``anyOf`` / ``oneOf``).
# These are common in practice — e.g. ``form_input.value: str | bool`` in
# Pydantic serialises to ``{"anyOf": [{"type": "string"}, {"type": "boolean"}]}``.
# Without the union-aware check, the XML parser's ``json.loads`` falls back
# to raw text for bare strings, but flags the call ``INVALID_JSON`` because
# no top-level ``type`` key declared a string — silently dropping otherwise
# valid tool calls in the renderer client.
UNION_WITH_STRING_SCHEMAS = [
    pytest.param(
        {"anyOf": [{"type": "string"}, {"type": "boolean"}]},
        id="anyOf-string-boolean",
    ),
    pytest.param(
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        id="anyOf-string-null",
    ),
    pytest.param(
        {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        id="oneOf-string-integer",
    ),
]


@pytest.mark.parametrize("param_schema", UNION_WITH_STRING_SCHEMAS)
def test_union_with_string_emits_ok_status(
    model, renderer_name, renderer, param_schema
):
    """Union schemas containing ``string`` must yield ``status=OK`` when
    the model emits a bare string. Pre-fix, ``_coerce_arg_value`` flagged
    this as ``INVALID_JSON`` because the top-level ``type`` key was
    absent (the string branch was under ``anyOf`` / ``oneOf``)."""
    from renderers.base import ToolCallParseStatus

    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "Test tool with one union-typed parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": param_schema},
                    "required": ["x"],
                },
            },
        }
    ]
    args = {"x": "abc"}
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
    completion_ids = _extract_assistant_tokens(renderer, PROMPT, msg, tools=tools)
    parsed = renderer.parse_response(completion_ids, tools=tools)

    assert parsed.tool_calls, f"{model}: parser returned no tool_calls"
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK, (
        f"{model}: union-with-string schema flagged {tc.status} on bare string"
    )
    got = _normalize_args(tc.arguments)
    assert got == args, (
        f"{model}: tool-arg drift — sent {args!r}, parser returned {got!r}"
    )
