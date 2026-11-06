"""Schema-aware XML argument coercion and Qwen extraction."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.parsing import _coerce_arg_value, parse_qwen35


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer("Qwen/Qwen3.5-9B")


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
    value, used_fallback = _coerce_arg_value("yes", {"type": "boolean"})
    assert value == "yes"
    assert used_fallback is True


@pytest.mark.parametrize(
    "declared",
    [["boolean", "null"], ["integer", "null"], ["number", "null"], ["object", "null"]],
)
def test_null_coerces_when_null_is_in_type_set(declared):
    for raw in ("null", "Null", "NULL"):
        value, used_fallback = _coerce_arg_value(raw, {"type": declared})
        assert value is None
        assert used_fallback is False


def test_string_typed_null_returns_verbatim():
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
    schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}

    value, used_fallback = _coerce_arg_value("42", schema)
    assert value == 42
    assert used_fallback is False

    value, used_fallback = _coerce_arg_value("LIS", schema)
    assert value == "LIS"
    assert used_fallback is False


def test_oneof_with_string_branch_recovers_bare_strings():
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
    value, used_fallback = _coerce_arg_value("2024-01-01", {"type": "date"})
    assert value == "2024-01-01"
    assert used_fallback is True

    value, used_fallback = _coerce_arg_value("[1, 2]", {"type": "date"})
    assert value == [1, 2]
    assert used_fallback is False


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


def test_qwen35_capital_true_for_boolean_is_ok_status(tok):
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


def test_qwen35_tolerates_missing_closing_parameter(tok):
    parsed = _parse(
        tok,
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/\n"
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


def test_qwen35_tolerates_unclosed_function(tok):
    parsed = _parse(
        tok,
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n",
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.UNCLOSED_BLOCK
    assert tc.name == "filesystem_server_get_directory_tree"
    assert tc.arguments == {"path": "/"}


def test_qwen35_backoff_no_tool_call_markers(tok):
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

    assert parsed.content == "sure, calling now.\n"


def test_qwen35_backoff_returns_no_calls_when_no_function_either(tok):
    parsed = _parse(tok, "just a chat reply", tools=_TOOLS_FROM_ISSUE)
    assert parsed.tool_calls == []
    assert parsed.content == "just a chat reply"


def test_qwen35_content_text_not_stripped(tok):
    parsed = _parse(
        tok,
        "let me check\n\n"
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


def test_qwen35_tool_call_id_matches_vllm_format(tok):
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
    pattern = re.compile(r"^chatcmpl-tool-[0-9a-f]{16}$")
    ids = [tc.id for tc in parsed.tool_calls]
    for tcid in ids:
        assert tcid is not None and pattern.match(tcid), tcid

    assert ids[0] != ids[1]


def test_qwen35_exception_fallback_returns_full_text(tok):
    completion = (
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    with patch(
        "renderers.parsing._parse_xml_tool_calls",
        side_effect=RuntimeError("boom"),
    ):
        parsed = _parse(tok, completion, tools=_TOOLS_FROM_ISSUE)
    assert parsed.tool_calls == []

    assert "<tool_call>" in parsed.content
    assert "<function=filesystem_server_get_directory_tree>" in parsed.content


def test_qwen35_parameter_value_preserves_internal_whitespace(tok):
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

    assert tc.arguments == {"path": "  /etc  "}


@pytest.mark.parametrize(
    "schema,raw",
    [
        ({"type": "number"}, "NaN"),
        ({"type": "number"}, "inf"),
        ({"type": "number"}, "-Infinity"),
        ({"type": "number"}, "1e999"),
        ({"type": "array"}, "[1e999]"),
        ({"type": "object"}, '{"nested": [NaN]}'),
        ({"type": "integer"}, "Infinity"),
        ({"type": "date"}, '{"nested": [1e999]}'),
    ],
)
def test_qwen35_nonfinite_arguments_preserve_text_and_flag_invalid(tok, schema, raw):
    tools = [{"name": "f", "parameters": {"properties": {"x": schema}}}]
    parsed = _parse(
        tok,
        f"<tool_call><function=f><parameter=x>{raw}</parameter></function></tool_call>",
        tools=tools,
    )
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.status == ToolCallParseStatus.INVALID_JSON
    assert call.arguments == {"x": raw}


@pytest.mark.parametrize(
    "schema", [None, {"type": "string"}, {"type": ["number", "string"]}]
)
def test_qwen35_nonfinite_text_with_string_schema_is_ok(tok, schema):
    tools = [{"name": "f", "parameters": {"properties": {"x": schema}}}]
    parsed = _parse(
        tok,
        "<tool_call><function=f><parameter=x>NaN</parameter></function></tool_call>",
        tools=tools,
    )
    assert parsed.tool_calls[0].status == ToolCallParseStatus.OK
    assert parsed.tool_calls[0].arguments == {"x": "NaN"}


@pytest.mark.parametrize(
    "body,status",
    [
        ("<function=></function>", ToolCallParseStatus.MISSING_NAME),
        (
            "<function=f\n<parameter=x>1</parameter></function>",
            ToolCallParseStatus.MALFORMED_STRUCTURE,
        ),
    ],
)
def test_qwen35_malformed_function_is_not_ok(tok, body, status):
    parsed = _parse(tok, f"<tool_call>{body}</tool_call>")
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].status == status


def test_qwen35_pluggable_parser_preserves_untyped_values(tok):
    from renderers.parsers import get_tool_parser

    parser = get_tool_parser("qwen3.5", tok)
    ids = tok.encode(
        "prefix<tool_call><function=f>"
        "<parameter=count>42</parameter><parameter=enabled>True</parameter>"
        "<parameter=empty>null</parameter></function></tool_call>",
        add_special_tokens=False,
    )
    content_ids, calls = parser.extract(list(ids))
    assert tok.decode(content_ids, skip_special_tokens=False) == "prefix"
    assert len(calls) == 1
    assert calls[0].status == ToolCallParseStatus.OK
    assert calls[0].arguments == {"count": "42", "enabled": "True", "empty": "null"}
    assert calls[0].token_span == (len(content_ids), len(ids))
