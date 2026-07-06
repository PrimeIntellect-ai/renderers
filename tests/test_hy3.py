"""Hy3-specific coverage beyond the shared barrage.

The parity / roundtrip / bridge matrices already assert byte-exact
``apply_chat_template`` agreement and the emit/parse round trip. This file
pins the behaviours unique to Hy3: the ``reasoning_effort`` generation-prompt
polarity, parsing of a live inference stream (where the ``<think>`` opener
lives in the prompt, not the completion), the reasoning-mode marker
placement, and tool-call parse statuses.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from renderers import Hy3RendererConfig, create_renderer
from renderers.base import ToolCallParseStatus, load_tokenizer

_MODEL = "tencent/Hy3"

_ASSISTANT = "<｜hy_Assistant:opensource｜>"
_THINK = "<think:opensource>"
_THINK_END = "</think:opensource>"
_EOS = "<｜hy_eos:opensource｜>"
_REASONING_MODE = "<｜reasoning_mode:opensource｜>"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@lru_cache(maxsize=None)
def _tok():
    return load_tokenizer(_MODEL)


def _renderer(**flags):
    return create_renderer(_tok(), Hy3RendererConfig(**flags))


def _decode(ids):
    return _tok().decode(ids, skip_special_tokens=False)


# ── generation-prompt polarity ─────────────────────────────────────────


@pytest.mark.parametrize(
    "effort,expected_tail",
    [
        ("no_think", _ASSISTANT + _THINK + _THINK_END),
        ("low", _ASSISTANT + _THINK),
        ("high", _ASSISTANT + _THINK),
    ],
)
def test_generation_prompt_polarity(effort, expected_tail):
    r = _renderer(reasoning_effort=effort)
    ids = r.render_ids([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert _decode(ids).endswith(expected_tail), (
        f"effort={effort}: gen prompt tail was {_decode(ids)[-60:]!r}"
    )


def test_reasoning_mode_marker_in_system_without_tools():
    """Without tools, ``<｜reasoning_mode｜>reasoning_effort:{effort}`` is
    appended to the system blob."""
    r = _renderer(reasoning_effort="high")
    text = _decode(r.render_ids([{"role": "user", "content": "Hi"}]))
    assert _REASONING_MODE + "reasoning_effort:high" in text


def test_reasoning_mode_marker_rides_tools_footer():
    """With tools, the reasoning-mode marker moves to the end of the tool
    instructions (after ``</tool_calls>``), not the system blob."""
    r = _renderer(reasoning_effort="low")
    text = _decode(r.render_ids([{"role": "user", "content": "Hi"}], tools=TOOLS))
    assert "</tool_calls:opensource>" + _REASONING_MODE + "reasoning_effort:low" in text


# ── stop token ──────────────────────────────────────────────────────────


def test_stop_token_is_eos_only():
    r = _renderer()
    eos_id = _tok().convert_tokens_to_ids(_EOS)
    assert r.get_stop_token_ids() == [eos_id]


# ── parsing a live inference stream ──────────────────────────────────────


def test_parse_low_mode_inference_stream():
    """In low/high mode the completion starts with reasoning text and closes
    it with ``</think>`` the model emits itself (the ``<think>`` opener was in
    the prompt)."""
    r = _renderer(reasoning_effort="low")
    comp = _tok().encode(
        "Let me work it out." + _THINK_END + "It is 4." + _EOS,
        add_special_tokens=False,
    )
    parsed = r.parse_response(comp)
    assert parsed.reasoning_content == "Let me work it out."
    assert parsed.content == "It is 4."
    assert not parsed.tool_calls


def test_parse_no_think_inference_stream():
    """In no_think mode the completion is the bare answer (both think tokens
    were prefilled into the prompt)."""
    r = _renderer()
    comp = _tok().encode("It is 4." + _EOS, add_special_tokens=False)
    parsed = r.parse_response(comp)
    assert parsed.content == "It is 4."
    assert parsed.reasoning_content is None


def test_parse_tool_call_stream_with_schema():
    """A tool-call completion parses name + typed args; the schema keeps a
    string arg verbatim (status OK, no JSON fallback)."""
    r = _renderer()
    comp = _tok().encode(
        "<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n"
        "<arg_key:opensource>city</arg_key:opensource>\n"
        "<arg_value:opensource>Paris</arg_value:opensource>\n"
        "</tool_call:opensource>\n</tool_calls:opensource>" + _EOS,
        add_special_tokens=False,
    )
    parsed = r.parse_response(comp, tools=TOOLS)
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}
    assert tc.status is ToolCallParseStatus.OK
    assert parsed.content == ""


def test_parse_unclosed_tool_call_is_flagged():
    r = _renderer()
    comp = _tok().encode(
        "<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n"
        "<arg_key:opensource>city</arg_key:opensource>\n"
        "<arg_value:opensource>Paris</arg_value:opensource>\n",
        add_special_tokens=False,
    )
    parsed = r.parse_response(comp)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK


def test_parse_content_before_tool_call_preserved():
    r = _renderer()
    comp = _tok().encode(
        "Let me check."
        "<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n"
        "<arg_key:opensource>city</arg_key:opensource>\n"
        "<arg_value:opensource>Paris</arg_value:opensource>\n"
        "</tool_call:opensource>\n</tool_calls:opensource>" + _EOS,
        add_special_tokens=False,
    )
    parsed = r.parse_response(comp, tools=TOOLS)
    assert parsed.content == "Let me check."
    assert parsed.tool_calls[0].name == "get_weather"


# ── preserved_thinking history retention ─────────────────────────────────


def test_preserved_thinking_history_retention():
    """A historical assistant (before the last user turn) keeps its reasoning
    only when ``preserved_thinking`` resolves True."""
    convo = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "R1", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "reasoning_content": "R2", "content": "A2"},
    ]
    stripped = _decode(_renderer(preserved_thinking=False).render_ids(convo))
    # Historical reasoning "R1" dropped; in-flight "R2" kept.
    assert "R1" not in stripped
    assert "R2" in stripped

    kept = _decode(_renderer(preserved_thinking=True).render_ids(convo))
    assert "R1" in kept and "R2" in kept


def test_preserved_thinking_defaults_to_tools_presence():
    """With the default (None) config, the template keeps historical reasoning
    iff tools are supplied at render time."""
    convo = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "R1", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "reasoning_content": "R2", "content": "A2"},
    ]
    r = _renderer()
    assert "R1" not in _decode(r.render_ids(convo))
    assert "R1" in _decode(r.render_ids(convo, tools=TOOLS))


# ── bridge policy resolution ─────────────────────────────────────────────


def test_effective_retention_defaults_conservative():
    assert _renderer().effective_thinking_retention == "tool_cycle"
    assert _renderer(preserved_thinking=True).effective_thinking_retention == "all"
    assert _renderer(thinking_retention="all").effective_thinking_retention == "all"


def test_preserved_thinking_conflict_raises():
    with pytest.raises(ValueError):
        Hy3RendererConfig(preserved_thinking=True, thinking_retention="tool_cycle")
    with pytest.raises(ValueError):
        Hy3RendererConfig(preserved_thinking=False, thinking_retention="all")


# ── is_training / raw_last_assistant / fallback_strategy ─────────────────


def test_is_training_keeps_all_thinking_and_closes_final_assistant():
    """``is_training=True`` keeps historical reasoning regardless of position
    and terminates the final assistant with ``<｜hy_eos｜>``."""
    convo = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "R1", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "reasoning_content": "R2", "content": "A2"},
    ]
    default = _decode(_renderer().render_ids(convo))
    training = _decode(_renderer(is_training=True).render_ids(convo))
    assert "R1" not in default and not default.endswith(_EOS)
    assert "R1" in training and "R2" in training and training.endswith(_EOS)


def test_raw_last_assistant_drops_wrap_and_eos():
    """A trailing non-tool assistant renders as bare visible content."""
    convo = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "reasoning_content": "R", "content": "the answer"},
    ]
    raw = _decode(_renderer(raw_last_assistant=True).render_ids(convo))
    assert raw.endswith(_ASSISTANT + "the answer")  # no think wrap, no eos
    assert _THINK not in raw.split(_ASSISTANT)[-1]


def test_fallback_strategy_forces_high_and_no_gen_prompt():
    """``reasoning_toolcall_retry`` forces high effort and suppresses the gen
    prompt even when the caller asks for it."""
    r = _renderer(fallback_strategy="reasoning_toolcall_retry")
    ids = r.render_ids([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    text = _decode(ids)
    assert _REASONING_MODE + "reasoning_effort:high" in text
    assert not text.endswith(_ASSISTANT + _THINK)  # gen prompt suppressed
    assert not text.endswith(_ASSISTANT)
