"""Smoke coverage for the ``thinking_retention`` override flag.

The flag lives on the typed renderer config (e.g.
``Qwen3RendererConfig(thinking_retention="all")``) and is stored on the
renderer as ``self.config.thinking_retention``. Each test that wants a
non-default level builds a fresh renderer for that configuration via
``_make`` below.

Two invariants per renderer:

1. Default render (``thinking_retention=None``) is byte-identical to
   the existing ``apply_chat_template`` parity baseline — covered
   exhaustively elsewhere.
2. Raising the level never *removes* tokens compared to the default and,
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
from renderers.base import MODEL_RENDERER_MAP, should_preserve_past_thinking
from renderers.configs import _config_class_for


def _make(tokenizer, renderer_name, **flags):
    """Build a fresh renderer with the given thinking_retention level
    bound at construction. Reuses the cached tokenizer fixture."""
    if renderer_name == "auto":
        renderer_name = MODEL_RENDERER_MAP.get(
            getattr(tokenizer, "name_or_path", ""), "default"
        )
    config = _config_class_for(renderer_name)(**flags)
    return create_renderer(tokenizer, config)


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
    # Llama-3 has no reasoning channel at all — thinking_retention can't
    # add or drop anything, so it's a pure no-op.
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
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
    # CURRENT-block-only behaviour. "tool_cycle" preserves thinking
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
        thinking_retention="tool_cycle",
    )
    assert should_preserve_past_thinking(
        live_cycle,
        3,
        thinking_retention="tool_cycle",
    )

    # Same shape with a NEW user appended → now the prior tool block is
    # "older" and "tool_cycle" must drop its thinking (template default).
    # Only thinking_retention="all" would keep them.
    closed_cycle = live_cycle + [{"role": "user", "content": "next"}]
    assert not should_preserve_past_thinking(
        closed_cycle,
        1,
        thinking_retention="tool_cycle",
    )
    assert not should_preserve_past_thinking(
        closed_cycle,
        3,
        thinking_retention="tool_cycle",
    )
    # thinking_retention="all" still keeps them.
    assert should_preserve_past_thinking(
        closed_cycle,
        1,
        thinking_retention="all",
    )
    assert should_preserve_past_thinking(
        closed_cycle,
        3,
        thinking_retention="all",
    )

    # Current segment without a tool → not a tool cycle → not preserved.
    no_tool_yet = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "r", "content": "a"},
    ]
    assert not should_preserve_past_thinking(
        no_tool_yet,
        1,
        thinking_retention="tool_cycle",
    )

def test_thinking_retention_default_unchanged(
    model_name, tokenizer, renderer_name, renderer
):
    # A renderer constructed without an explicit retention override should
    # produce byte-identical output to the fixture's default renderer.
    bare = renderer.render_ids(CONVERSATION)
    derived = _make(tokenizer, renderer_name).render_ids(CONVERSATION)
    assert bare == derived, f"{model_name}: default construction changed output"


def test_thinking_retention_all_grows_or_no_op(
    model_name, tokenizer, renderer_name, renderer
):
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    preserved = _make(tokenizer, renderer_name, thinking_retention="all").render_ids(
        CONVERSATION
    )

    if model_name in NO_OP_MODELS:
        assert preserved == default, (
            f"{model_name} is a no-op renderer; thinking_retention='all' must "
            f"not change output (got {len(default)} → {len(preserved)})"
        )
    else:
        assert len(preserved) > len(default), (
            f"{model_name}: thinking_retention='all' should add tokens for a "
            f"conversation with past-asst reasoning_content "
            f"(default={len(default)}, preserved={len(preserved)})"
        )


def test_thinking_retention_tool_cycle_strict_subset(
    model_name, tokenizer, renderer_name, renderer
):
    """``thinking_retention="tool_cycle"`` is strictly weaker than
    ``"all"``: token count satisfies default <= tool_cycle <= all."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    between = _make(
        tokenizer, renderer_name, thinking_retention="tool_cycle"
    ).render_ids(CONVERSATION)
    all_ = _make(tokenizer, renderer_name, thinking_retention="all").render_ids(
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


def test_thinking_retention_tool_cycle_matches_all_on_live_cycle(
    model_name, tokenizer, renderer_name, renderer
):
    """In a live tool cycle (no trailing user), every past-asst sits in
    the current tool-bearing segment. ``thinking_retention="tool_cycle"``
    should preserve all of their thinking — same set of asst messages as
    ``"all"``, so the resulting token sequences must be
    identical (independent of which template-default condition each
    renderer uses internally)."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    btc = _make(tokenizer, renderer_name, thinking_retention="tool_cycle").render_ids(
        LIVE_TOOL_CYCLE
    )
    all_ = _make(tokenizer, renderer_name, thinking_retention="all").render_ids(
        LIVE_TOOL_CYCLE
    )
    assert btc == all_, (
        f"{model_name}: in a live tool cycle tool_cycle must match all "
        f"(got len(tool_cycle)={len(btc)}, len(all)={len(all_)})"
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
    # Llama-3 ships no <think> rendering path, so reasoning_content never
    # surfaces in the output regardless of thinking_retention.
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
}


def test_thinking_retention_all_emits_every_asst_reasoning(
    model_name, tokenizer, renderer_name, renderer
):
    """``thinking_retention="all"`` must surface every past-asst's
    ``reasoning_content`` in the decoded output — for renderers that
    have any pathway to render reasoning at all."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")

    ids = _make(tokenizer, renderer_name, thinking_retention="all").render_ids(
        TWO_BLOCK_CONV, tools=TWO_BLOCK_TOOLS
    )
    text = tokenizer.decode(ids)

    if model_name in NEVER_PRESERVES_MODELS:
        for sentinel in ALL_SENTINELS:
            assert sentinel not in text, (
                f"{model_name}: never-preserves renderer leaked {sentinel} "
                f"under thinking_retention='all'"
            )
    else:
        for sentinel in ALL_SENTINELS:
            assert sentinel in text, (
                f"{model_name}: thinking_retention='all' did not emit {sentinel} "
                f"in decoded output"
            )


def test_thinking_retention_tool_cycle_emits_current_block_reasoning(
    model_name, tokenizer, renderer_name, renderer
):
    """``thinking_retention="tool_cycle"`` must surface the current
    (post-last-user) tool block's reasoning. Older blocks fall back to
    template default, which varies per renderer — no universal assertion
    there."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")

    ids = _make(tokenizer, renderer_name, thinking_retention="tool_cycle").render_ids(
        TWO_BLOCK_CONV, tools=TWO_BLOCK_TOOLS
    )
    text = tokenizer.decode(ids)

    if model_name in NEVER_PRESERVES_MODELS:
        for sentinel in ALL_SENTINELS:
            assert sentinel not in text, (
                f"{model_name}: never-preserves renderer leaked {sentinel} "
                f"under thinking_retention='tool_cycle'"
            )
    else:
        for sentinel in CURRENT_BLOCK_SENTINELS:
            assert sentinel in text, (
                f"{model_name}: btc did not emit current-block {sentinel} "
                f"in decoded output"
            )


def test_default_renderer_raises_on_explicit_retention():
    """``DefaultRenderer`` falls back to apply_chat_template with no
    selective re-emit pathway, so constructing one with explicit
    thinking_retention must raise — fail fast, before any render."""
    from renderers import DefaultRendererConfig
    from renderers.base import load_tokenizer

    tok = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    # Default unset policy → constructs cleanly.
    create_renderer(tok, DefaultRendererConfig())
    # Any explicit level → raises at construction.
    with pytest.raises(ValueError):
        create_renderer(tok, DefaultRendererConfig(thinking_retention="all"))
    with pytest.raises(ValueError):
        create_renderer(
            tok,
            DefaultRendererConfig(thinking_retention="tool_cycle"),
        )


# ---------------------------------------------------------------------------
# Construction-time configuration is discoverable via instance attributes
# ---------------------------------------------------------------------------


def test_create_renderer_records_flag_state(model_name, renderer_name, tokenizer):
    """Each renderer exposes the bound flag state via ``self.config`` —
    useful for downstream code (pool cache keys, logging, test
    assertions) that needs to confirm what was constructed."""
    from renderers.default import DefaultRenderer

    bare = _make(tokenizer, renderer_name)
    assert bare.config.thinking_retention is None
    assert bare.effective_thinking_retention in {"template", "tool_cycle", "all"}

    if not isinstance(bare, DefaultRenderer):
        # DefaultRenderer raises at construction with explicit retention —
        # covered by ``test_default_renderer_raises_on_explicit_retention``.
        all_on = _make(tokenizer, renderer_name, thinking_retention="all")
        assert all_on.config.thinking_retention == "all"

        btc_on = _make(tokenizer, renderer_name, thinking_retention="tool_cycle")
        assert btc_on.config.thinking_retention == "tool_cycle"


# ---------------------------------------------------------------------------
# Regression: legacy chat-template-kwarg pass-throughs are gone
# ---------------------------------------------------------------------------


def test_glm5_config_accepts_clear_thinking():
    """``clear_thinking`` is a chat-template field on GLM-5's typed
    config. The GLM-5 / GLM-5.1 Jinja templates gate historical
    reasoning on ``clear_thinking is defined and not clear_thinking``,
    so passing ``clear_thinking=False`` here must reach the renderer's
    historical-reasoning gate. Parity vs ``apply_chat_template`` is
    asserted in ``test_renderer_config_parity``."""
    from renderers import GLM5RendererConfig
    from renderers.base import load_tokenizer
    from renderers.glm5 import GLM5Renderer

    tok = load_tokenizer("zai-org/GLM-5")
    # Both values must be accepted without raising.
    GLM5Renderer(tok, GLM5RendererConfig(clear_thinking=True))
    GLM5Renderer(tok, GLM5RendererConfig(clear_thinking=False))


def test_qwen36_config_accepts_preserve_thinking():
    """``preserve_thinking`` is a Qwen3.6 chat-template kwarg and should be
    exposed directly on the typed config."""
    from renderers import Qwen36RendererConfig

    Qwen36RendererConfig(preserve_thinking=True)
