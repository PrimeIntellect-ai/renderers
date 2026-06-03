"""Tests for the pluggable ToolParser / ReasoningParser registry."""

from __future__ import annotations

import pytest

from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.parsers import (
    REASONING_PARSERS,
    TOOL_PARSERS,
    Qwen3ToolParser,
    ThinkTextReasoningParser,
    get_reasoning_parser,
    get_tool_parser,
)


def test_registries_nonempty():
    assert "qwen3" in TOOL_PARSERS
    assert "qwen3.5" in TOOL_PARSERS
    assert "glm" in TOOL_PARSERS
    assert "deepseek-v3" in TOOL_PARSERS
    assert "think" in REASONING_PARSERS


def test_unknown_parser_errors():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    with pytest.raises(ValueError, match="Unknown tool_parser"):
        get_tool_parser("does-not-exist", tok)
    with pytest.raises(ValueError, match="Unknown reasoning_parser"):
        get_reasoning_parser("does-not-exist", tok)


def test_qwen3_tool_parser_roundtrip():
    """Tokenize a Hermes-style tool call, parse it back out."""
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    assert isinstance(parser, Qwen3ToolParser)

    completion_text = 'hello\n<tool_call>\n{"name": "search", "arguments": {"q": "rain"}}\n</tool_call>'
    token_ids = tok.encode(completion_text, add_special_tokens=False)
    content_ids, tool_calls = parser.extract(list(token_ids))
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "search"
    assert tc.arguments == {"q": "rain"}
    assert tc.token_span is not None and tc.token_span[0] < tc.token_span[1]
    # content_ids should cover everything up to (but not including) <tool_call>
    content_text = tok.decode(content_ids, skip_special_tokens=False)
    assert "hello" in content_text
    assert "<tool_call>" not in content_text


def test_qwen3_tool_parser_no_tool_call():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    ids = tok.encode("just plain text response", add_special_tokens=False)
    content_ids, tool_calls = parser.extract(list(ids))
    assert tool_calls == []
    assert content_ids == list(ids)


def test_qwen3_tool_parser_records_invalid_json():
    """Malformed JSON in a <tool_call> block surfaces as INVALID_JSON, not silently dropped."""
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    completion = (
        'hi\n<tool_call>\n{"name": "f", "arguments": {broken json\n</tool_call>'
    )
    ids = tok.encode(completion, add_special_tokens=False)
    _, tool_calls = parser.extract(list(ids))
    assert len(tool_calls) == 1
    assert tool_calls[0].status == ToolCallParseStatus.INVALID_JSON
    assert tool_calls[0].raw  # raw block text preserved


def test_qwen3_tool_parser_parallel_partial_success():
    """Parallel calls: parser keeps the good ones AND records the broken one."""
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    completion = (
        "pre\n"
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        "<tool_call>\n{broken\n</tool_call>\n"
        '<tool_call>\n{"name": "c", "arguments": {"x": 1}}\n</tool_call>'
    )
    ids = tok.encode(completion, add_special_tokens=False)
    _, tool_calls = parser.extract(list(ids))
    assert [tc.status for tc in tool_calls] == [
        ToolCallParseStatus.OK,
        ToolCallParseStatus.INVALID_JSON,
        ToolCallParseStatus.OK,
    ]
    assert [tc.name for tc in tool_calls] == ["a", None, "c"]


def test_think_reasoning_parser_extracts_block():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_reasoning_parser("think", tok)
    assert isinstance(parser, ThinkTextReasoningParser)
    reasoning, content = parser.extract("<think>let me think</think>the answer")
    assert reasoning == "let me think"
    assert content == "the answer"


def test_think_reasoning_parser_no_block():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_reasoning_parser("think", tok)
    reasoning, content = parser.extract("no reasoning here")
    assert reasoning is None
    assert content == "no reasoning here"


def test_default_renderer_uses_parsers():
    """DefaultRenderer + parsers should extract tool calls and reasoning."""
    from renderers import DefaultRendererConfig, create_renderer

    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    renderer = create_renderer(
        tok,
        DefaultRendererConfig(tool_parser="qwen3", reasoning_parser="think"),
    )
    assert renderer.supports_tools is True

    completion = '<think>think</think>ok\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    ids = tok.encode(completion, add_special_tokens=False)
    parsed = renderer.parse_response(list(ids))
    assert parsed.reasoning_content == "think"
    assert parsed.content.startswith("ok")
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "f"
    assert parsed.tool_calls[0].status == ToolCallParseStatus.OK


def test_default_renderer_without_parsers_is_backward_compatible():
    """Without parsers, DefaultRenderer still does basic <think> extraction."""
    from renderers import DefaultRendererConfig, create_renderer

    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    renderer = create_renderer(tok, DefaultRendererConfig())
    assert renderer.supports_tools is False

    ids = tok.encode("<think>r</think>a", add_special_tokens=False)
    parsed = renderer.parse_response(list(ids))
    assert parsed.reasoning_content == "r"
    assert parsed.content == "a"
    assert parsed.tool_calls == []
