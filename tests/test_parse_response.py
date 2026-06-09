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
    renderer = create_renderer(tokenizer)
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
def _qwen3():
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")
    renderer = create_renderer(tokenizer)
    return tokenizer, renderer


def test_qwen3_in_think_tool_call_is_not_a_real_call():
    """A ``<tool_call>`` the model drafts *inside* its ``<think>`` trace must
    stay reasoning — only the call emitted after ``</think>`` counts.

    Regression for #78: Thinking models (e.g. Qwen3-*-Thinking-2507) draft
    tool-call syntax while planning. Because ``<tool_call>`` is a real vocab
    token, the parser used to scan the whole stream and emit the in-think
    draft *and* the genuine post-``</think>`` call as two tool calls — a
    phantom duplicate that made callers execute the same code twice. The scan
    is now anchored after ``</think>``, mirroring vLLM's reasoning-then-tools
    ordering.
    """
    tokenizer, renderer = _qwen3()
    text = (
        "<think>\nLet me draft the call:\n"
        '<tool_call>\n{"name": "execute_code", "arguments": {"code": "print(1)"}}\n'
        "</tool_call>\nYes, that looks right.\n</think>\n"
        '<tool_call>\n{"name": "execute_code", "arguments": {"code": "print(1)"}}\n'
        "</tool_call>"
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "execute_code"
    assert tc.arguments == {"code": "print(1)"}
    # The drafted call stays in the reasoning trace, not content.
    assert parsed.reasoning_content is not None
    assert "<tool_call>" in parsed.reasoning_content
    assert parsed.content == ""


def test_qwen3_distinct_parallel_calls_after_think_are_preserved():
    """The fix must not over-correct: two *genuine* parallel calls emitted
    after ``</think>`` are still both returned (no dedup), preserving the
    faithful-transcription contract for real invocations.
    """
    tokenizer, renderer = _qwen3()
    text = (
        "<think>\nplan\n</think>\n"
        '<tool_call>\n{"name": "execute_code", "arguments": {"code": "print(1)"}}\n'
        "</tool_call>\n"
        '<tool_call>\n{"name": "execute_code", "arguments": {"code": "print(2)"}}\n'
        "</tool_call>"
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert len(parsed.tool_calls) == 2
    assert [tc.arguments for tc in parsed.tool_calls] == [
        {"code": "print(1)"},
        {"code": "print(2)"},
    ]
    assert parsed.reasoning_content == "plan"


@lru_cache
def _kimi_k25():
    tokenizer = load_tokenizer("moonshotai/Kimi-K2.5")
    renderer = create_renderer(tokenizer)
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


def test_kimi_k25_in_think_section_is_not_a_real_call():
    """A tool-call section the model drafts inside its ``<think>`` trace must
    not be parsed — only the section after ``</think>`` counts.

    Regression for #78. K2.5's failure mode differed from Qwen3's: the
    in-think section tripped the "truncated reasoning" guard and the parser
    *dropped every tool call* (returned zero), losing the genuine call. The
    scan is now anchored past ``</think>``.
    """
    tokenizer, renderer = _kimi_k25()
    section = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.execute_code:0"
        '<|tool_call_argument_begin|>{"code": "print(1)"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    text = f"<think>\nLet me draft:\n{section}\nlooks right.\n</think>\nGo.\n{section}"
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "execute_code"
    assert tc.arguments == {"code": "print(1)"}
    # The drafted section stays in the reasoning trace.
    assert parsed.reasoning_content is not None
    assert "<|tool_calls_section_begin|>" in parsed.reasoning_content
    assert parsed.content == "Go."


@lru_cache
def _deepseek_v3():
    tokenizer = load_tokenizer("deepseek-ai/DeepSeek-V3")
    renderer = create_renderer(tokenizer)
    return tokenizer, renderer


def test_deepseek_v3_in_think_section_is_not_a_real_call():
    """A tool-call section drafted inside ``<think>`` must not be parsed —
    only the section after ``</think>`` counts.

    Regression for #78. DeepSeek-V3's failure mode: it returned the *wrong*
    call (the in-think draft) and lost reasoning, because ``</think>`` is
    multi-token text there and the scan wasn't anchored past it.
    """
    tokenizer, renderer = _deepseek_v3()

    def section(name: str) -> str:
        return (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
            f'{name}\n```json\n{{"code": "print(1)"}}\n```'
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )

    text = (
        f"<think>\nLet me draft:\n{section('draft_tool')}\nlooks right.\n</think>\n"
        f"Go.\n{section('real_tool')}"
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    # Assert the *post-``</think>``* section was chosen, not the in-think draft.
    # (Use ``startswith`` rather than ``== "real_tool"``: under transformers
    # 5.x the DeepSeek tokenizer's decode drops the ``\n`` between name and the
    # ```json fence, so ``_parse_deepseek_tool_calls`` folds the fence into the
    # name — a pre-existing, #78-unrelated quirk. What matters here is *which*
    # section won.)
    assert tc.name is not None and tc.name.startswith("real_tool")
    assert "draft_tool" not in tc.name
    # The drafted section stays in the reasoning trace, not content.
    assert parsed.reasoning_content is not None
    assert "draft_tool" in parsed.reasoning_content
    assert parsed.content == "Go."
