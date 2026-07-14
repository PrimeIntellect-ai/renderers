"""Schema-aware argument coercion parity with vLLM / SGLang.

The XML-style tool-call wire format (Qwen3.5, GLM, MiniMax, Laguna)
renders parameter values verbatim with no quoting, so the parser must
consult the tool schema to know whether ``True`` is a bool or a string.
Renderers' previous coercion was strict ``json.loads``, which flagged
``True`` / ``False`` (capitalised Python literals) as ``INVALID_JSON``
for boolean params — but both vLLM's ``Qwen3CoderToolParser`` and
SGLang's ``Qwen3CoderDetector`` accept them via case-folded comparison.

These tests pin the new ladder to the vLLM / SGLang reference shape and
guard the bug from issue #47.
"""

from __future__ import annotations

import pytest

from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.parsing import _coerce_arg_value, parse_qwen35


# ── Direct coercion ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
    ],
)
def test_boolean_accepts_case_insensitive(raw, expected):
    value, used_fallback = _coerce_arg_value(raw, {"type": "boolean"})
    assert value is expected
    assert used_fallback is False


@pytest.mark.parametrize("declared", ["boolean", "bool"])
def test_boolean_type_aliases_all_accepted(declared):
    value, used_fallback = _coerce_arg_value("True", {"type": declared})
    assert value is True
    assert used_fallback is False


def test_boolean_garbage_keeps_raw_text_with_invalid_flag():
    """vLLM ``coerce_to_schema_type``: when the boolean branch declines
    (value not in ``{true, false, 1, 0}``) and no other type in the
    schema accepts the value, the terminal ``json.loads`` is attempted
    and on failure the raw text is returned. We surface that path
    via ``used_fallback=True`` for the verifier / RL-loss signal."""
    value, used_fallback = _coerce_arg_value("yes", {"type": "boolean"})
    assert value == "yes"
    assert used_fallback is True


@pytest.mark.parametrize(
    "declared",
    [["boolean", "null"], ["integer", "null"], ["number", "null"], ["object", "null"]],
)
def test_null_coerces_when_null_is_in_type_set(declared):
    """vLLM null-coerces only when ``"null"`` is one of the declared
    schema types — typically via JSON-Schema array-form ``["X", "null"]``
    (the ``Optional[X]`` shape) or via ``anyOf: [..., {"type": "null"}]``.
    Case-insensitive: matches ``null`` / ``Null`` / ``NULL``."""
    for raw in ("null", "Null", "NULL"):
        value, used_fallback = _coerce_arg_value(raw, {"type": declared})
        assert value is None
        assert used_fallback is False


def test_string_typed_null_returns_verbatim():
    """Pure string-typed param: ``"null"`` on the wire is just the
    string ``"null"`` — no null coercion because ``"null"`` is not in
    the schema's type set. Byte parity with vLLM
    ``coerce_to_schema_type`` (string branch always succeeds)."""
    for raw in ("null", "Null", "NULL"):
        value, used_fallback = _coerce_arg_value(raw, {"type": "string"})
        assert value == raw
        assert used_fallback is False


@pytest.mark.parametrize(
    "declared,raw,expected",
    [
        ("integer", "42", 42),
        ("int", "-7", -7),
        ("uint", "0", 0),
        ("long", "9999999999", 9999999999),
        ("short", "12", 12),
        ("unsigned", "5", 5),
    ],
)
def test_int_family_coerces(declared, raw, expected):
    value, used_fallback = _coerce_arg_value(raw, {"type": declared})
    assert value == expected
    assert isinstance(value, int)
    assert used_fallback is False


def test_int_failure_keeps_raw_and_flags_fallback():
    value, used_fallback = _coerce_arg_value("abc", {"type": "integer"})
    assert value == "abc"
    assert used_fallback is True


@pytest.mark.parametrize(
    "raw,expected,exact_type",
    [
        ("3.14", 3.14, float),
        ("-2.5", -2.5, float),
        # vLLM ``coerce_to_schema_type`` int-demotes whenever the parsed
        # float is a whole number, regardless of source-string shape —
        # so ``1e3``, ``1.0``, and ``7`` all become int.
        ("1e3", 1000, int),
        ("1.0", 1, int),
        ("7", 7, int),
    ],
)
def test_float_family_coerces(raw, expected, exact_type):
    value, used_fallback = _coerce_arg_value(raw, {"type": "number"})
    assert value == expected
    assert type(value) is exact_type
    assert used_fallback is False


def test_float_failure_keeps_raw_and_flags_fallback():
    value, used_fallback = _coerce_arg_value("not-a-number", {"type": "float"})
    assert value == "not-a-number"
    assert used_fallback is True


def test_object_via_json_loads():
    value, used_fallback = _coerce_arg_value('{"k": 1}', {"type": "object"})
    assert value == {"k": 1}
    assert used_fallback is False


def test_object_python_literal_does_not_parse():
    """vLLM ``coerce_to_schema_type`` only uses ``json.loads`` for
    object/array types — no ``ast.literal_eval`` fallback. Python-style
    single-quoted dicts therefore land on the terminal raw-text path
    and get flagged. (Renderers' previous ``ast.literal_eval`` fallback
    was a vLLM deviation; we dropped it for parity.)"""
    value, used_fallback = _coerce_arg_value("{'k': 1}", {"type": "object"})
    assert value == "{'k': 1}"
    assert used_fallback is True


def test_array_via_json_loads():
    value, used_fallback = _coerce_arg_value("[1, 2, 3]", {"type": "array"})
    assert value == [1, 2, 3]
    assert used_fallback is False


def test_object_total_failure_flags_invalid():
    value, used_fallback = _coerce_arg_value(
        "this is not a json object", {"type": "object"}
    )
    assert value == "this is not a json object"
    assert used_fallback is True


def test_anyof_recursively_flattens_types():
    """vLLM ``extract_types_from_schema`` recurses through ``anyOf`` /
    ``oneOf`` / ``allOf`` to build the type set. ``Union[int, str]`` ⇒
    ``{integer, string}``: integer parses first, falling back to string
    for non-numeric values — and a bare string never gets flagged
    because the string branch is the always-succeeding terminal."""
    schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}

    # Integer-shaped wire value rides the integer branch.
    value, used_fallback = _coerce_arg_value("42", schema)
    assert value == 42
    assert used_fallback is False

    # Non-numeric wire value declines integer, lands on string. The
    # exact regression from PR #63: pre-fix, this was flagged
    # ``INVALID_JSON`` because top-level ``type`` was absent and the
    # parser tried ``json.loads("LIS")``. vLLM's recursive flatten +
    # priority-ordered ladder fixes that as a side effect.
    value, used_fallback = _coerce_arg_value("LIS", schema)
    assert value == "LIS"
    assert used_fallback is False


def test_oneof_with_string_branch_recovers_bare_strings():
    """Same recursive-flatten path as ``anyOf``: ``oneOf`` with a
    string branch must accept bare-string wire values without flagging."""
    schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
    value, used_fallback = _coerce_arg_value("hello", schema)
    assert value == "hello"
    assert used_fallback is False


@pytest.mark.parametrize(
    "declared", ["string", "str", "text", "varchar", "char", "enum"]
)
def test_string_family_returns_verbatim(declared):
    value, used_fallback = _coerce_arg_value("True", {"type": declared})
    assert value == "True"
    assert used_fallback is False


def test_string_with_list_form_type():
    value, used_fallback = _coerce_arg_value("True", {"type": ["string"]})
    assert value == "True"
    assert used_fallback is False


def test_no_schema_returns_verbatim_string():
    """vLLM ``extract_types_from_schema(None)`` returns ``["string"]``,
    which routes every value through the always-succeeding string
    branch of the priority ladder. Untyped ``"null"`` therefore stays
    a string — there is no top-level null short-circuit ahead of the
    type check."""
    value, used_fallback = _coerce_arg_value("42", None)
    assert value == "42"
    assert used_fallback is False

    value, used_fallback = _coerce_arg_value("True", None)
    assert value == "True"
    assert used_fallback is False

    value, used_fallback = _coerce_arg_value("hello", None)
    assert value == "hello"
    assert used_fallback is False

    value, used_fallback = _coerce_arg_value("null", None)
    assert value == "null"
    assert used_fallback is False


def test_unknown_type_falls_through_to_terminal_json_loads():
    """A declared type the priority ladder doesn't recognise (e.g.
    ``"date"``) skips every typed branch and lands on the terminal
    ``json.loads``. JSON-shaped values still parse; everything else
    keeps the raw text and gets flagged."""
    value, used_fallback = _coerce_arg_value("2024-01-01", {"type": "date"})
    assert value == "2024-01-01"
    assert used_fallback is True

    # JSON-shaped value rides the terminal ``json.loads``.
    value, used_fallback = _coerce_arg_value("[1, 2]", {"type": "date"})
    assert value == [1, 2]
    assert used_fallback is False


# ── Integration: parse_qwen35 end-to-end (the bug from issue #47) ───


_TOOLS_FROM_ISSUE = [
    {
        "type": "function",
        "function": {
            "name": "filesystem_server_get_directory_tree",
            "description": "List a directory tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer"},
                    "include_files": {"type": "boolean"},
                    "show_size": {"type": "boolean"},
                },
            },
        },
    }
]


def _parse(tok, completion: str, *, tools=None):
    ids = tok.encode(completion, add_special_tokens=False)
    return parse_qwen35(
        tok,
        list(ids),
        stop_ids={tok.eos_token_id} if tok.eos_token_id is not None else set(),
        think_id=tok.convert_tokens_to_ids("<think>"),
        think_end_id=tok.convert_tokens_to_ids("</think>"),
        tool_call_id=tok.convert_tokens_to_ids("<tool_call>"),
        tool_call_end_id=tok.convert_tokens_to_ids("</tool_call>"),
        tools=tools,
    )


def test_qwen35_capital_true_for_boolean_is_ok_status():
    """Regression for issue #47: ``<parameter=include_files>True</parameter>``
    used to be flagged ``INVALID_JSON`` because ``json.loads("True")``
    fails. SGLang and vLLM both accept it via case-folded comparison;
    renderers now does too."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(
        tok,
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n"
        "<parameter=max_depth>2</parameter>\n"
        "<parameter=include_files>True</parameter>\n"
        "<parameter=show_size>True</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "filesystem_server_get_directory_tree"
    assert tc.arguments == {
        "path": "/",
        "max_depth": 2,
        "include_files": True,
        "show_size": True,
    }


# ── Structural tolerance: parity with vLLM's three regex patterns ────


def test_qwen35_tolerates_missing_closing_parameter():
    """vLLM ``tool_call_parameter_regex`` uses positive lookaheads so a
    missing ``</parameter>`` is recovered from the next ``<parameter=``
    or ``</function>`` boundary instead of dropping the value."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(
        tok,
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/\n"  # missing </parameter>
        "<parameter=max_depth>2</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "filesystem_server_get_directory_tree"
    assert tc.arguments == {"path": "/", "max_depth": 2}


def test_qwen35_tolerates_unclosed_function():
    """vLLM ``tool_call_function_regex`` second alternation matches an
    unclosed ``<function=…`` body to end-of-text; we still flag the
    surrounding diagnostic via ``UNCLOSED_BLOCK`` but recover the call."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(
        tok,
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n",  # no </function>, no </tool_call>
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.UNCLOSED_BLOCK
    assert tc.name == "filesystem_server_get_directory_tree"
    assert tc.arguments == {"path": "/"}


def test_qwen35_backoff_no_tool_call_markers():
    """vLLM ``qwen3coder_tool_parser.py:269-271``: when the output
    contains no ``<tool_call>`` markers, treat the whole completion as
    a single tool-call region and scan for ``<function=…>`` directly.
    Content text is the raw slice up to the first ``<function=``,
    matching ``qwen3coder_tool_parser.py:316-319`` (no strip)."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(
        tok,
        "sure, calling now.\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/etc</parameter>\n"
        "<parameter=max_depth>1</parameter>\n"
        "</function>",
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "filesystem_server_get_directory_tree"
    assert tc.arguments == {"path": "/etc", "max_depth": 1}
    # vLLM-parity: content = text up to first ``<function=``, no strip.
    assert parsed.content == "sure, calling now.\n"


def test_qwen35_backoff_returns_no_calls_when_no_function_either():
    """No ``<tool_call>`` and no ``<function=`` → no recovered calls,
    full text returned as content verbatim (no strip, vLLM parity)."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(tok, "just a chat reply", tools=_TOOLS_FROM_ISSUE)
    assert parsed.tool_calls == []
    assert parsed.content == "just a chat reply"


def test_qwen35_content_text_not_stripped():
    """vLLM ``qwen3coder_tool_parser.py:316-319`` slices raw text before
    the first ``<tool_call>``; the commented-out ``.rstrip()`` confirms
    intent. We mirror that — surrounding whitespace round-trips."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(
        tok,
        "let me check\n\n"  # trailing whitespace before <tool_call>
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tools=_TOOLS_FROM_ISSUE,
    )
    assert parsed.content == "let me check\n\n"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].status == ToolCallParseStatus.OK


def test_qwen35_tool_call_id_matches_vllm_format():
    """vLLM's ``ToolCall`` default factory mints ids of the form
    ``chatcmpl-tool-<16 hex>``. Each surfaced call gets a fresh id."""
    import re as _re

    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(
        tok,
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/etc</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 2
    pattern = _re.compile(r"^chatcmpl-tool-[0-9a-f]{16}$")
    ids = [tc.id for tc in parsed.tool_calls]
    for tcid in ids:
        assert tcid is not None and pattern.match(tcid), tcid
    # Two surfaced calls must get distinct ids.
    assert ids[0] != ids[1]


def test_qwen35_exception_fallback_returns_full_text():
    """vLLM ``qwen3coder_tool_parser.py:327-331`` catch-all returns
    ``tools_called=False, tool_calls=[], content=model_output`` on any
    unhandled extraction error. We mirror that by wrapping the
    tool-call extraction; reasoning content extracted earlier is kept.
    """
    from unittest.mock import patch

    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    completion = (
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    ids = tok.encode(completion, add_special_tokens=False)

    with patch(
        "renderers.parsing._parse_xml_tool_calls",
        side_effect=RuntimeError("boom"),
    ):
        parsed = parse_qwen35(
            tok,
            list(ids),
            stop_ids={tok.eos_token_id} if tok.eos_token_id is not None else set(),
            think_id=tok.convert_tokens_to_ids("<think>"),
            think_end_id=tok.convert_tokens_to_ids("</think>"),
            tool_call_id=tok.convert_tokens_to_ids("<tool_call>"),
            tool_call_end_id=tok.convert_tokens_to_ids("</tool_call>"),
            tools=_TOOLS_FROM_ISSUE,
        )
    assert parsed.tool_calls == []
    # content is the full decoded post-think text, surfaced verbatim.
    assert "<tool_call>" in parsed.content
    assert "<function=filesystem_server_get_directory_tree>" in parsed.content


def test_qwen35_parameter_value_preserves_internal_whitespace():
    """vLLM strips exactly one leading and one trailing ``\\n`` from the
    parameter body (``qwen3coder_tool_parser.py:246-250``); inner
    indentation must round-trip verbatim."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")
    parsed = _parse(
        tok,
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>\n  /etc  \n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    # One leading + one trailing \n stripped — interior spaces preserved.
    assert tc.arguments == {"path": "  /etc  "}
