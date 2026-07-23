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


# ── tool-group re-opening (is_tool_first state machine) ─────────────────


def test_tool_group_not_reopened_after_plain_assistant():
    """The template resets ``is_tool_first`` only on an assistant that made
    tool calls; tool results arriving after a plain assistant continue the
    earlier <tool_responses> group without re-opening it. Byte-parity check
    on a shape the shared matrices never generate."""
    convo = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "Checking again."},
        {"role": "tool", "content": "rainy"},
        {"role": "user", "content": "So?"},
    ]
    ours = _renderer().render_ids(convo, tools=TOOLS, add_generation_prompt=True)
    theirs = _tok().apply_chat_template(
        convo,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    assert ours == list(theirs)


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


_BRIDGE_ASST = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ],
}
_BRIDGE_EXT = [
    {"role": "tool", "content": "sunny"},
    {"role": "user", "content": "And tomorrow?"},
]


def test_bridge_default_config_with_tools_extends_across_user_turn():
    """With the default (None) config the bridge policy mirrors the template's
    tools-dependent ``preserved_thinking``: a full render with tools keeps
    historical reasoning across user turns, so declining the bridge there
    would force a re-render that drops the verbatim sampled tokens for no
    semantic gain. The bridge must extend, byte-identical to a fresh render."""
    r = _renderer()
    prev = [{"role": "user", "content": "Weather?"}]
    pp = r.render_ids(prev, tools=TOOLS, add_generation_prompt=True)
    pc = list(r.render_ids([*prev, _BRIDGE_ASST], tools=TOOLS)[len(pp) :])

    bridged = r.bridge_to_next_turn(pp, pc, _BRIDGE_EXT, tools=TOOLS)
    fresh = r.render_ids(
        [*prev, _BRIDGE_ASST, *_BRIDGE_EXT], tools=TOOLS, add_generation_prompt=True
    )
    assert bridged is not None and bridged.token_ids == fresh


def test_bridge_default_config_without_tools_declines_at_user_turn():
    """Without tools the template default drops past-cycle reasoning once a
    new user query arrives, so the faithful bridge declines there."""
    r = _renderer()
    prev = [{"role": "user", "content": "Q1"}]
    pp = r.render_ids(prev, add_generation_prompt=True)
    pc = list(r.render_ids([*prev, {"role": "assistant", "content": "A1"}])[len(pp) :])
    assert r.bridge_to_next_turn(pp, pc, [{"role": "user", "content": "Q2"}]) is None


def test_bridge_explicit_tool_cycle_declines_at_user_turn_despite_tools():
    """An explicit ``thinking_retention="tool_cycle"`` wins over the
    tools-dependent template default: the bridge still declines at a new
    user turn even though tools are present."""
    r = _renderer(thinking_retention="tool_cycle")
    prev = [{"role": "user", "content": "Weather?"}]
    pp = r.render_ids(prev, tools=TOOLS, add_generation_prompt=True)
    pc = list(r.render_ids([*prev, _BRIDGE_ASST], tools=TOOLS)[len(pp) :])
    assert r.bridge_to_next_turn(pp, pc, _BRIDGE_EXT, tools=TOOLS) is None


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


def test_fallback_strategy_bridge_matches_full_render():
    """The bridge must also suppress the generation prompt under the fallback
    retry strategy, so it stays consistent with a fresh full render."""
    r = _renderer(
        fallback_strategy="reasoning_toolcall_retry", thinking_retention="all"
    )
    eos = _tok().convert_tokens_to_ids(_EOS)
    prior = [{"role": "user", "content": "Hi."}]
    pp = r.render_ids(prior, add_generation_prompt=True)
    full = r.render_ids([*prior, {"role": "assistant", "content": "Hello!"}])
    pc = list(full[len(pp) :])
    if not pc or pc[-1] != eos:
        pc = pc + [eos]

    bridged = r.bridge_to_next_turn(pp, pc, [{"role": "tool", "content": "result"}])
    assert bridged is not None
    # No generation prompt appended (fallback forces add_generation_prompt=False).
    assert bridged.token_ids[-1] != _tok().convert_tokens_to_ids(_ASSISTANT)
    assert _ASSISTANT not in _decode(bridged.token_ids[len(pp) + len(pc) :])


# ── tool-call token_span reporting ───────────────────────────────────────


def test_tool_call_token_span_indexes_stripped_stream():
    """``token_span`` must slice the stop-stripped completion back to the
    <tool_call>…</tool_call> block, accounting for the reasoning + leading
    content stripped before parsing."""
    from renderers.parsing import _strip_stop_tokens

    r = _renderer(reasoning_effort="low")
    tok = _tok()
    comp = tok.encode(
        "Let me think." + _THINK_END + "Checking now."
        "<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n"
        "<arg_key:opensource>city</arg_key:opensource>\n"
        "<arg_value:opensource>Paris</arg_value:opensource>\n"
        "</tool_call:opensource>\n</tool_calls:opensource>" + _EOS,
        add_special_tokens=False,
    )
    parsed = r.parse_response(comp, tools=TOOLS)
    tc = parsed.tool_calls[0]
    assert tc.status is ToolCallParseStatus.OK
    stripped = _strip_stop_tokens(comp, {tok.convert_tokens_to_ids(_EOS)})
    s, e = tc.token_span
    assert stripped[s] == tok.convert_tokens_to_ids("<tool_call:opensource>")
    assert stripped[e - 1] == tok.convert_tokens_to_ids("</tool_call:opensource>")
    # The reported block decodes to exactly this call, no reasoning/content leak.
    block = tok.decode(stripped[s:e], skip_special_tokens=False)
    assert block.startswith("<tool_call:opensource>get_weather")
    assert "Let me think" not in block and "Checking now" not in block


# ── bridge close-token synthesis ─────────────────────────────────────────


def test_bridge_tool_cycle_matches_full_render():
    """A bridge over a clean assistant tool-call turn (completion ends in
    <｜hy_eos｜>) must be byte-identical to a fresh full render — no spurious
    close token wedged in."""
    r = _renderer(thinking_retention="all")
    eos = _tok().convert_tokens_to_ids(_EOS)
    asst = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    }
    prev = [{"role": "user", "content": "Weather?"}]
    pp = r.render_ids(prev, add_generation_prompt=True)
    pc = list(r.render_ids([*prev, asst])[len(pp) :])
    assert pc[-1] == eos  # tool-call turn closes with eos

    bridged = r.bridge_to_next_turn(pp, pc, [{"role": "tool", "content": '{"t": 20}'}])
    fresh = r.render_ids(
        [*prev, asst, {"role": "tool", "content": '{"t": 20}'}],
        add_generation_prompt=True,
    )
    assert bridged is not None and bridged.token_ids == fresh


def test_bridge_tool_groups_across_user_turn_match_full_render():
    """Within one extension, ``is_tool_first`` is consumed by the first tool
    group and never resets (no assistant turns are allowed in extensions), so
    a second group after a user turn must not re-open <tool_responses> —
    exactly as a fresh full render tokenizes it."""
    r = _renderer(preserved_thinking=True)
    asst = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    }
    prev = [{"role": "user", "content": "Weather?"}]
    pp = r.render_ids(prev, tools=TOOLS, add_generation_prompt=True)
    pc = list(r.render_ids([*prev, asst], tools=TOOLS)[len(pp) :])

    ext = [
        {"role": "tool", "content": "sunny"},
        {"role": "user", "content": "And tomorrow?"},
        {"role": "tool", "content": "rainy"},
    ]
    bridged = r.bridge_to_next_turn(pp, pc, ext, tools=TOOLS)
    fresh = r.render_ids([*prev, asst, *ext], tools=TOOLS, add_generation_prompt=True)
    assert bridged is not None and bridged.token_ids == fresh


def test_bridge_declines_on_empty_completion():
    """With no sampled completion there is no assistant turn to extend — the
    bridge declines (so a stream ending at a </tool_responses> tool-section
    boundary never gets a spurious <｜hy_eos｜> wedged before the extension)."""
    rf = _renderer(
        fallback_strategy="reasoning_toolcall_retry", thinking_retention="all"
    )
    asst = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    }
    base = [
        {"role": "user", "content": "Weather?"},
        asst,
        {"role": "tool", "content": '{"t": 20}'},
    ]
    prompt = rf.render_ids(base, add_generation_prompt=True)  # fallback → no gen prompt
    assert prompt[-1] == _tok().convert_tokens_to_ids("</tool_responses:opensource>")
    assert (
        rf.bridge_to_next_turn(prompt, [], [{"role": "tool", "content": "more"}])
        is None
    )
