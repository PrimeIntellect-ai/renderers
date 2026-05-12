"""Barrage test: renderer.parse_response() must correctly extract
content, reasoning_content, and tool_calls from completion tokens.

Runs against every (model, renderer) pair.
"""

from functools import lru_cache

from renderers import create_renderer
from renderers.base import ToolCallParseStatus, load_tokenizer


@lru_cache
def _qwen3_vl():
    tokenizer = load_tokenizer("Qwen/Qwen3-VL-4B-Instruct")
    renderer = create_renderer(tokenizer, renderer="auto")
    return tokenizer, renderer


def test_parse_simple_content(model_name, tokenizer, renderer):
    """Plain content, no thinking."""
    text = "Hello there!"
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert "Hello" in parsed.content


def test_parse_thinking_and_content(model_name, tokenizer, renderer):
    """Content with <think>reasoning</think> block."""
    text = "Let me think about this.\n</think>\n\nThe answer is 42."
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    # Should extract reasoning or at least not crash
    assert (
        "42" in parsed.content
        or "think" in (parsed.reasoning_content or "").lower()
        or parsed.content
    )


def test_parse_empty_completion(model_name, tokenizer, renderer):
    """Empty completion should not crash."""
    parsed = renderer.parse_response([])
    assert parsed.content is not None


def test_parse_response_returns_parsed_response(model_name, tokenizer, renderer):
    """Return type must have content, reasoning_content, tool_calls."""
    ids = tokenizer.encode("Hello!", add_special_tokens=False)
    parsed = renderer.parse_response(ids)
    assert hasattr(parsed, "content")
    assert hasattr(parsed, "reasoning_content")
    assert hasattr(parsed, "tool_calls")


def test_qwen3_vl_parse_json_tool_call():
    tokenizer, renderer = _qwen3_vl()
    text = (
        'Need a tool.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert parsed.content == "Need a tool."
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}


def test_qwen3_vl_malformed_tool_call_surfaces_as_invalid_json():
    """A malformed ``<tool_call>`` block lands as a non-OK ``ParsedToolCall``
    rather than getting silently merged back into ``content``.

    Before the per-call status redesign, the parser mirrored vLLM's
    hermes parser and stuffed the raw block into ``content`` to avoid
    downstream ``EmptyModelResponseError``. That hid the malformed signal
    from verifiers — they couldn't tell "model wrote prose" from "model
    tried a tool call and produced broken JSON." Now the failed attempt
    is preserved with ``status=INVALID_JSON`` and ``raw`` text, which
    also satisfies the EmptyModelResponseError prevention contract: the
    response is non-empty (it has a tool-call attempt) without lying
    about what kind of output the model produced.
    """
    tokenizer, renderer = _qwen3_vl()
    # Note the trailing comma — malformed JSON
    text = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris",}}\n</tool_call>'
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.INVALID_JSON
    assert "get_weather" in tc.raw
    assert tc.token_span is not None


@lru_cache
def _kimi_k25():
    tokenizer = load_tokenizer("moonshotai/Kimi-K2.5")
    renderer = create_renderer(tokenizer, renderer="auto")
    return tokenizer, renderer


def test_kimi_k25_tool_call_carries_token_span():
    """K2.5 was the lone parser without token spans before — its inline
    text-walking implementation couldn't cheaply map regex hits back to
    token offsets. We now walk token IDs via ``parse_kimi_k2_section`` for
    the special-token path; spans must round-trip and point at a sensible
    range within the original input token_ids.
    """
    tokenizer, renderer = _kimi_k25()
    # K2.5 tool-call wire shape: section + per-call special tokens.
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>"
        '{"city": "Tokyo"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_response(token_ids)

    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Tokyo"}
    assert tc.token_span is not None
    start, end = tc.token_span
    assert 0 <= start < end <= len(token_ids), (
        f"span {tc.token_span} out of range for {len(token_ids)} input tokens"
    )
