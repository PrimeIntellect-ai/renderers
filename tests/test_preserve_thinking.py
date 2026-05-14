"""Smoke coverage for the ``preserve_*_thinking`` override flags.

Flags are constructor-only (``create_renderer(..., preserve_all_thinking=True)``)
and stored as instance attributes — there is no call-site ``render`` /
``render_ids`` override. Each test that wants a non-default flag builds
a fresh renderer for that configuration via ``_make`` below.

Two invariants per renderer:

1. Default render (no flags) is byte-identical to the existing
   ``apply_chat_template`` parity baseline — covered exhaustively elsewhere.
2. Setting either flag never *removes* tokens compared to the default and,
   for renderers whose template would drop past-asst thinking, actually
   adds tokens for a conversation containing past-asst ``reasoning_content``.

Renderers whose template either always preserves thinking (DeepSeek-V3) or
never references ``reasoning_content`` for past-asst (Kimi-K2, Qwen3-VL)
are no-ops by design — they're listed below and the test asserts the
default==override equality instead of strict growth.
"""

from __future__ import annotations

import pytest

from renderers import create_renderer
from renderers.base import should_preserve_past_thinking


def _make(tokenizer, renderer_name, **flags):
    """Build a fresh renderer with the given preserve_*_thinking flags
    bound at construction. Reuses the cached tokenizer fixture."""
    return create_renderer(tokenizer, renderer=renderer_name, **flags)


# Renderers whose template doesn't drop past-asst thinking or has no
# place to re-emit it. For these, override flags MUST be no-ops.
NO_OP_MODELS = {
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-Base",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "poolside/Laguna-XS.2",
}


CONVERSATION = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "I should call the weather tool for Paris.",
        "content": "Let me check.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "name": "get_weather", "content": "Sunny, 22C"},
    {
        "role": "assistant",
        "reasoning_content": "The tool returned the weather.",
        "content": "Sunny, 22C in Paris.",
    },
    {"role": "user", "content": "And Berlin?"},
]


def test_should_preserve_past_thinking_classification():
    # CURRENT-block-only behaviour. between_tool_calls preserves thinking
    # ONLY for asst messages that sit AFTER the last user turn AND are in
    # a segment that contains a tool. Anything before the last user turn
    # falls back to template default (typically dropped).

    # Live tool cycle: U-A_tc-T-A_final, no trailing user. The whole
    # post-user segment contains a tool, so both A's are preserved.
    live_cycle = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "reasoning_content": "r1",
            "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
        },
        {"role": "tool", "name": "f", "content": "data"},
        {"role": "assistant", "reasoning_content": "r2", "content": "answer"},
    ]
    assert should_preserve_past_thinking(
        live_cycle,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    assert should_preserve_past_thinking(
        live_cycle,
        3,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )

    # Same shape with a NEW user appended → now the prior tool block is
    # "older" and between_tool_calls must drop its thinking (template
    # default). Only preserve_all_thinking would keep them.
    closed_cycle = live_cycle + [{"role": "user", "content": "next"}]
    assert not should_preserve_past_thinking(
        closed_cycle,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    assert not should_preserve_past_thinking(
        closed_cycle,
        3,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    # preserve_all_thinking still keeps them.
    assert should_preserve_past_thinking(
        closed_cycle,
        1,
        preserve_all_thinking=True,
        preserve_thinking_between_tool_calls=False,
    )
    assert should_preserve_past_thinking(
        closed_cycle,
        3,
        preserve_all_thinking=True,
        preserve_thinking_between_tool_calls=False,
    )

    # Current segment without a tool → not a tool cycle → not preserved.
    no_tool_yet = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "r", "content": "a"},
    ]
    assert not should_preserve_past_thinking(
        no_tool_yet,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )

    # Both flags False → always False.
    assert not should_preserve_past_thinking(
        live_cycle,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
    )


def test_preserve_flags_default_unchanged(
    model_name, tokenizer, renderer_name, renderer
):
    # A renderer constructed with both flags explicitly ``False`` must
    # produce byte-identical output to one constructed with the defaults.
    bare = renderer.render_ids(CONVERSATION)
    explicit_off = _make(
        tokenizer,
        renderer_name,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
    ).render_ids(CONVERSATION)
    assert bare == explicit_off, (
        f"{model_name}: explicit flags=False must equal bare default render"
    )


def test_preserve_all_thinking_grows_or_no_op(
    model_name, tokenizer, renderer_name, renderer
):
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    preserved = _make(tokenizer, renderer_name, preserve_all_thinking=True).render_ids(
        CONVERSATION
    )

    if model_name in NO_OP_MODELS:
        assert preserved == default, (
            f"{model_name} is a no-op renderer; preserve_all_thinking must "
            f"not change output (got {len(default)} → {len(preserved)})"
        )
    else:
        assert len(preserved) > len(default), (
            f"{model_name}: preserve_all_thinking should add tokens for a "
            f"conversation with past-asst reasoning_content "
            f"(default={len(default)}, preserved={len(preserved)})"
        )


def test_preserve_between_tool_calls_strict_subset(
    model_name, tokenizer, renderer_name, renderer
):
    """``preserve_thinking_between_tool_calls`` is strictly weaker than
    ``preserve_all_thinking``: token count satisfies default <= between <= all."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    between = _make(
        tokenizer, renderer_name, preserve_thinking_between_tool_calls=True
    ).render_ids(CONVERSATION)
    all_ = _make(tokenizer, renderer_name, preserve_all_thinking=True).render_ids(
        CONVERSATION
    )
    assert len(default) <= len(between) <= len(all_), (
        f"{model_name}: expected default <= between <= all, "
        f"got {len(default)} <= {len(between)} <= {len(all_)}"
    )


LIVE_TOOL_CYCLE = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "Let me call the tool.",
        "content": "Calling.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "name": "get_weather", "content": "Sunny, 22C"},
    {
        "role": "assistant",
        "reasoning_content": "Tool returned weather.",
        "content": "Sunny.",
    },
]


def test_preserve_btc_on_live_cycle_matches_all(
    model_name, tokenizer, renderer_name, renderer
):
    """In a live tool cycle (no trailing user), every past-asst sits in
    the current tool-bearing segment. ``preserve_thinking_between_tool_calls``
    should preserve all of their thinking — same set of asst messages as
    ``preserve_all_thinking``, so the resulting token sequences must be
    identical (independent of which template-default condition each
    renderer uses internally)."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    btc = _make(
        tokenizer, renderer_name, preserve_thinking_between_tool_calls=True
    ).render_ids(LIVE_TOOL_CYCLE)
    all_ = _make(tokenizer, renderer_name, preserve_all_thinking=True).render_ids(
        LIVE_TOOL_CYCLE
    )
    assert btc == all_, (
        f"{model_name}: in a live tool cycle btc must match preserve_all "
        f"(got len(btc)={len(btc)}, len(all)={len(all_)})"
    )


# ---------------------------------------------------------------------------
# End-to-end visibility matrix
# ---------------------------------------------------------------------------

# Conversation shape: S-U-A-T-A-U-A-T-A. Each assistant carries a unique
# sentinel string in ``reasoning_content`` so we can grep the decoded
# output to see whose thinking was kept.
TWO_BLOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look up a value",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]

TWO_BLOCK_CONV = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "first"},
    {
        "role": "assistant",
        "reasoning_content": "REASON-A2",
        "content": "calling.",
        "tool_calls": [{"function": {"name": "lookup", "arguments": {"key": "a"}}}],
    },
    {"role": "tool", "name": "lookup", "content": "result-a"},
    {"role": "assistant", "reasoning_content": "REASON-A4", "content": "answer-1"},
    {"role": "user", "content": "second"},
    {
        "role": "assistant",
        "reasoning_content": "REASON-A6",
        "content": "calling.",
        "tool_calls": [{"function": {"name": "lookup", "arguments": {"key": "b"}}}],
    },
    {"role": "tool", "name": "lookup", "content": "result-b"},
    {"role": "assistant", "reasoning_content": "REASON-A8", "content": "answer-2"},
]

ALL_SENTINELS = ("REASON-A2", "REASON-A4", "REASON-A6", "REASON-A8")
CURRENT_BLOCK_SENTINELS = ("REASON-A6", "REASON-A8")
OLDER_BLOCK_SENTINELS = ("REASON-A2", "REASON-A4")

# Renderers whose template renders ``reasoning_content`` for past-asst
# under no condition. Flags accepted as no-ops; sentinels never appear.
NEVER_PRESERVES_MODELS = {
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
}


def test_preserve_all_thinking_emits_every_asst_reasoning(
    model_name, tokenizer, renderer_name, renderer
):
    """``preserve_all_thinking=True`` must surface every past-asst's
    ``reasoning_content`` in the decoded output — for renderers that
    have any pathway to render reasoning at all."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")

    ids = _make(tokenizer, renderer_name, preserve_all_thinking=True).render_ids(
        TWO_BLOCK_CONV, tools=TWO_BLOCK_TOOLS
    )
    text = tokenizer.decode(ids)

    if model_name in NEVER_PRESERVES_MODELS:
        for sentinel in ALL_SENTINELS:
            assert sentinel not in text, (
                f"{model_name}: never-preserves renderer leaked {sentinel} "
                f"under preserve_all_thinking"
            )
    else:
        for sentinel in ALL_SENTINELS:
            assert sentinel in text, (
                f"{model_name}: preserve_all_thinking did not emit {sentinel} "
                f"in decoded output"
            )


def test_preserve_btc_emits_current_block_reasoning(
    model_name, tokenizer, renderer_name, renderer
):
    """``preserve_thinking_between_tool_calls=True`` must surface the
    current (post-last-user) tool block's reasoning. Older blocks fall
    back to template default, which varies per renderer — no universal
    assertion there."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")

    ids = _make(
        tokenizer, renderer_name, preserve_thinking_between_tool_calls=True
    ).render_ids(TWO_BLOCK_CONV, tools=TWO_BLOCK_TOOLS)
    text = tokenizer.decode(ids)

    if model_name in NEVER_PRESERVES_MODELS:
        for sentinel in ALL_SENTINELS:
            assert sentinel not in text, (
                f"{model_name}: never-preserves renderer leaked {sentinel} "
                f"under preserve_thinking_between_tool_calls"
            )
    else:
        for sentinel in CURRENT_BLOCK_SENTINELS:
            assert sentinel in text, (
                f"{model_name}: btc did not emit current-block {sentinel} "
                f"in decoded output"
            )


def test_default_renderer_raises_on_flags():
    """``DefaultRenderer`` falls back to apply_chat_template with no
    selective re-emit pathway, so constructing one with either flag set
    must raise — fail fast, before any render is attempted."""
    from renderers.base import load_tokenizer

    tok = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    # No flags → constructs cleanly.
    create_renderer(tok, renderer="default")
    # Either flag set → raises at construction.
    with pytest.raises(NotImplementedError):
        create_renderer(tok, renderer="default", preserve_all_thinking=True)
    with pytest.raises(NotImplementedError):
        create_renderer(
            tok, renderer="default", preserve_thinking_between_tool_calls=True
        )


# ---------------------------------------------------------------------------
# Construction-time configuration is discoverable via instance attributes
# ---------------------------------------------------------------------------


def test_create_renderer_records_flag_state(model_name, renderer_name, tokenizer):
    """Each renderer exposes the bound flag state via ``_preserve_*``
    attributes — useful for downstream code (pool cache keys, logging,
    test assertions) that needs to confirm what was constructed."""
    from renderers.default import DefaultRenderer

    bare = create_renderer(tokenizer, renderer=renderer_name)
    assert bare._preserve_all_thinking is False
    assert bare._preserve_thinking_between_tool_calls is False

    if not isinstance(bare, DefaultRenderer):
        # DefaultRenderer raises at construction with either flag set —
        # covered by ``test_default_renderer_raises_on_flags``.
        all_on = create_renderer(
            tokenizer, renderer=renderer_name, preserve_all_thinking=True
        )
        assert all_on._preserve_all_thinking is True
        assert all_on._preserve_thinking_between_tool_calls is False

        btc_on = create_renderer(
            tokenizer,
            renderer=renderer_name,
            preserve_thinking_between_tool_calls=True,
        )
        assert btc_on._preserve_all_thinking is False
        assert btc_on._preserve_thinking_between_tool_calls is True


# ---------------------------------------------------------------------------
# Regression: legacy chat-template-kwarg pass-throughs are gone
# ---------------------------------------------------------------------------


def test_glm5_constructor_rejects_clear_thinking():
    """``clear_thinking`` was a chat-template-kwarg pass-through. It is
    superseded by the renderer-agnostic ``preserve_all_thinking`` override
    and must no longer be accepted by the constructor — its default-True
    semantics are now baked into the render gate."""
    from renderers.base import load_tokenizer
    from renderers.glm5 import GLM5Renderer

    tok = load_tokenizer("zai-org/GLM-5")
    with pytest.raises(TypeError):
        GLM5Renderer(tok, clear_thinking=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        GLM5Renderer(tok, clear_thinking=False)  # type: ignore[call-arg]


def test_qwen36_constructor_rejects_preserve_thinking():
    """``preserve_thinking`` on Qwen3.6 was a chat-template-kwarg
    pass-through. It is superseded by the renderer-agnostic
    ``preserve_all_thinking`` override and must no longer be accepted by
    the constructor — its default-False semantics are now inherited from
    Qwen3.5's render gate."""
    from renderers.base import load_tokenizer
    from renderers.qwen36 import Qwen36Renderer

    tok = load_tokenizer("Qwen/Qwen3.6-35B-A3B")
    with pytest.raises(TypeError):
        Qwen36Renderer(tok, preserve_thinking=True)  # type: ignore[call-arg]
