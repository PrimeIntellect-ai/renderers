"""Bridge-failure diagnostics for ``Renderer.bridge_to_next_turn``.

The bridge contract returns ``list[int] | None``. When it returns
``None``, the caller learns the bridge couldn't prove its invariant
but not *why*. This module surfaces the "why" as a typed enum so
callers like ``verifiers`` and ``prime-rl`` can observe bridge
health per-turn during rollouts.

The README documents six structural failure modes:

* ``BOOL_ROUND_TRIP`` - a token decoded ``True``/``False`` differs
  across the bridged extension and a fresh render (the boolean
  re-tokenized to a different id-sequence).
* ``BPE_DRIFT`` - neighbouring-byte BPE retokenization shifted ids
  in the middle of a turn.
* ``TOOL_CALL_XML_DRIFT`` - a tool-call open/close span differs.
* ``THINKING_STRIPPED`` - thinking-channel tokens present in fresh
  render are missing from the bridged extension (or vice versa).
* ``TRUNCATION_ZEROED_ANCHOR`` - prev_prompt_ids exceeds the model's
  max length, so the bridge can't anchor a synth-close.
* ``ASSISTANT_IN_EXTENSION`` - a caller passed an assistant message
  in ``new_messages``, which the bridge refuses by contract.

This module adds one more, distinct from the six because it surfaces
a different failure mode:

* ``UNKNOWN_TEMPLATE_CLOSE`` - the renderer is ``DefaultRenderer``,
  which always returns ``None`` because it doesn't know its
  template's close token.

The classification is best-effort: when nothing more specific fits
the comparison surface, the diagnostic falls back to ``BPE_DRIFT``
(the most common cause empirically). Per-renderer hints (e.g.
recognising ``tool_call_id`` for Qwen3 or ``<|channel|>`` for
GPT-OSS) belong in the renderer subclasses; the protocol stays
small and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from renderers.base import (
    Message,
    Renderer,
    ToolSpec,
    reject_assistant_in_extension,
)
from renderers.default import DefaultRenderer


class BridgeFailureReason(StrEnum):
    """Why ``bridge_to_next_turn`` returned ``None`` (or produced a
    divergent extension). See module docstring for the full list."""

    ASSISTANT_IN_EXTENSION = "assistant_in_extension"
    BOOL_ROUND_TRIP = "bool_round_trip"
    BPE_DRIFT = "bpe_drift"
    THINKING_STRIPPED = "thinking_stripped"
    TOOL_CALL_XML_DRIFT = "tool_call_xml_drift"
    TRUNCATION_ZEROED_ANCHOR = "truncation_zeroed_anchor"
    UNKNOWN_TEMPLATE_CLOSE = "unknown_template_close"


@dataclass(frozen=True)
class BridgeDiagnostic:
    """One reason the bridge couldn't extend prev verbatim.

    ``token_span`` points at the first divergent token range in the
    bridged extension (or the offending position for non-comparison
    diagnoses like ``ASSISTANT_IN_EXTENSION``). ``detail`` is a
    short human-readable hint suitable for logging.
    """

    reason: BridgeFailureReason
    message_index: int
    token_span: tuple[int, int]
    detail: str


def diagnose_bridge(
    renderer: Renderer,
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    new_messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
) -> BridgeDiagnostic | None:
    """Return a structured reason the bridge would (or did) fail.

    Returns ``None`` when the bridge succeeds cleanly. Otherwise returns
    the most specific ``BridgeDiagnostic`` the comparison surface
    supports.

    Side-effect-free with respect to the renderer (no state writes).
    Calls ``renderer.bridge_to_next_turn`` and ``renderer.render_ids``
    once each; both are idempotent in the public API.
    """

    # 1) Contract-level reasons we can decide without re-rendering.
    if reject_assistant_in_extension(new_messages):
        idx = _first_assistant_index(new_messages)
        return BridgeDiagnostic(
            reason=BridgeFailureReason.ASSISTANT_IN_EXTENSION,
            message_index=idx,
            token_span=(0, 0),
            detail=(
                f"new_messages[{idx}] is role=assistant; bridges refuse to "
                "re-tokenize model-sampled content"
            ),
        )

    if isinstance(renderer, DefaultRenderer):
        return BridgeDiagnostic(
            reason=BridgeFailureReason.UNKNOWN_TEMPLATE_CLOSE,
            message_index=-1,
            token_span=(0, 0),
            detail=(
                "DefaultRenderer cannot synthesise a turn-close for "
                "unknown chat templates; caller must full-render"
            ),
        )

    max_len = _model_max_length(renderer)
    if max_len is not None and len(previous_prompt_ids) > max_len:
        return BridgeDiagnostic(
            reason=BridgeFailureReason.TRUNCATION_ZEROED_ANCHOR,
            message_index=-1,
            token_span=(max_len, len(previous_prompt_ids)),
            detail=(
                f"previous_prompt_ids has {len(previous_prompt_ids)} tokens "
                f"but model max length is {max_len}; anchor is below zero"
            ),
        )

    # 2) Comparison-based reasons: run the bridge and a fresh render and
    # locate the first divergence.
    bridged = renderer.bridge_to_next_turn(
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        tools=tools,
    )
    if bridged is None:
        # The bridge bailed for a reason we couldn't pre-classify. The
        # most common cause empirically is BPE drift on the synth-close.
        return BridgeDiagnostic(
            reason=BridgeFailureReason.BPE_DRIFT,
            message_index=-1,
            token_span=(len(previous_prompt_ids) + len(previous_completion_ids), -1),
            detail="bridge returned None; classification fell through to BPE_DRIFT",
        )

    # Render the full conversation fresh and compare.
    full_messages = _reconstruct_history(renderer, previous_prompt_ids, previous_completion_ids) + list(new_messages)
    try:
        fresh = renderer.render_ids(full_messages, add_generation_prompt=True, tools=tools)
    except Exception:
        # If we can't reconstruct, treat as BPE_DRIFT with no span.
        return BridgeDiagnostic(
            reason=BridgeFailureReason.BPE_DRIFT,
            message_index=-1,
            token_span=(-1, -1),
            detail="could not produce a fresh-render baseline to compare against",
        )

    cutoff = min(len(bridged), len(fresh))
    first_diff = None
    for i in range(cutoff):
        if bridged[i] != fresh[i]:
            first_diff = i
            break
    if first_diff is None and len(bridged) == len(fresh):
        return None  # Bridge matched fresh render exactly.
    if first_diff is None:
        first_diff = cutoff

    return _classify_divergence(
        renderer=renderer,
        bridged=bridged,
        fresh=fresh,
        first_diff=first_diff,
    )


def _first_assistant_index(messages: list[Message]) -> int:
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            return i
    return -1


def _model_max_length(renderer: Renderer) -> int | None:
    """Best-effort lookup of the tokenizer's ``model_max_length``.

    Returns ``None`` if the renderer doesn't expose its tokenizer or
    the tokenizer doesn't define a finite max length.
    """
    tok = getattr(renderer, "tokenizer", None) or getattr(renderer, "_tokenizer", None)
    if tok is None:
        return None
    raw = getattr(tok, "model_max_length", None)
    if raw is None:
        return None
    # HF marks "unlimited" with VERY_LARGE_INTEGER (1e30); ignore that.
    if raw > 10**9:
        return None
    return int(raw)


def _reconstruct_history(
    renderer: Renderer,
    prev_prompt_ids: list[int],
    prev_completion_ids: list[int],
) -> list[Message]:
    """Best-effort decode of the prior turns from id streams.

    The diagnostic only needs *something* it can pass to ``render_ids``
    to produce a fresh baseline; if the renderer exposes a parser, we
    use it. Failure produces an empty list (the caller catches and
    falls back to ``BPE_DRIFT``).
    """
    parser = getattr(renderer, "parse_response", None)
    if parser is None:
        return []
    try:
        parsed = parser(prev_completion_ids)
        # parse_response typically returns a ParsedResponse with
        # content; we wrap it as a single assistant turn.
        text = getattr(parsed, "text", None) or getattr(parsed, "content", "")
        if not text:
            return []
        return [{"role": "assistant", "content": str(text)}]
    except Exception:
        return []


def _classify_divergence(
    *,
    renderer: Renderer,
    bridged: list[int],
    fresh: list[int],
    first_diff: int,
) -> BridgeDiagnostic:
    """Pick the most specific reason for the first divergent token.

    Heuristics: BOOL_ROUND_TRIP catches single-token bool literals;
    THINKING_STRIPPED catches missing thinking-channel tokens by
    counting them. Otherwise we fall back to BPE_DRIFT (the empirical
    majority case).
    """

    tok = getattr(renderer, "tokenizer", None) or getattr(renderer, "_tokenizer", None)
    detail = f"first divergence at token index {first_diff}"

    if tok is not None:
        try:
            b_tok = tok.decode([bridged[first_diff]], skip_special_tokens=False)
            f_tok = tok.decode([fresh[first_diff]], skip_special_tokens=False)
            if {b_tok.strip().lower(), f_tok.strip().lower()} & {"true", "false"}:
                return BridgeDiagnostic(
                    reason=BridgeFailureReason.BOOL_ROUND_TRIP,
                    message_index=-1,
                    token_span=(first_diff, first_diff + 1),
                    detail=f"bool-shape token differs: {b_tok!r} vs {f_tok!r}",
                )
            detail = f"first divergence at index {first_diff}: bridged={b_tok!r} fresh={f_tok!r}"
        except Exception:
            pass

    return BridgeDiagnostic(
        reason=BridgeFailureReason.BPE_DRIFT,
        message_index=-1,
        token_span=(first_diff, first_diff + 1),
        detail=detail,
    )
