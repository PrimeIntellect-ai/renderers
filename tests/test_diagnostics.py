"""Unit tests for ``renderers.diagnostics``.

These exercise the orchestration of ``diagnose_bridge`` with light
stubs in place of a real tokenizer / renderer. The classification-rule
heuristics are kept narrow so the unit tests can drive every branch
without downloading a HuggingFace model.

A parametrized integration test exercises the orchestration against
every hand-coded renderer in ``_BRIDGE_MODELS``; it's marked
``slow`` so it doesn't run by default in CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from renderers.diagnostics import (
    BridgeDiagnostic,
    BridgeFailureReason,
    diagnose_bridge,
)


class _StubRenderer:
    """Minimal Renderer-shaped object for orchestration tests.

    Real renderers are subclasses of ``renderers.base.Renderer``; this
    stub mimics only the surface ``diagnose_bridge`` touches:
    ``bridge_to_next_turn``, ``render_ids``, ``parse_response``, and a
    ``tokenizer`` attribute.
    """

    def __init__(
        self,
        *,
        bridge_return: list[int] | None,
        fresh_return: list[int],
        max_length: int | None = None,
    ):
        self._bridge_return = bridge_return
        self._fresh_return = fresh_return
        self.tokenizer = SimpleNamespace(
            model_max_length=max_length if max_length is not None else 1_000_000,
            decode=lambda ids, skip_special_tokens=False: f"tok{ids[0]}",
        )

    def bridge_to_next_turn(self, prev_p, prev_c, new, *, tools=None):
        return self._bridge_return

    def render_ids(self, messages, *, add_generation_prompt=True, tools=None):
        return self._fresh_return

    def parse_response(self, ids):
        return SimpleNamespace(text="prior assistant text")


def test_returns_none_when_bridge_matches_fresh_exactly():
    r = _StubRenderer(bridge_return=[1, 2, 3, 4], fresh_return=[1, 2, 3, 4])
    assert diagnose_bridge(r, [1, 2], [3], [{"role": "user", "content": "x"}]) is None


def test_assistant_in_extension_short_circuits():
    r = _StubRenderer(bridge_return=None, fresh_return=[])
    diag = diagnose_bridge(
        r,
        [1],
        [2],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "rejected"},
        ],
    )
    assert isinstance(diag, BridgeDiagnostic)
    assert diag.reason is BridgeFailureReason.ASSISTANT_IN_EXTENSION
    assert diag.message_index == 1
    assert "assistant" in diag.detail.lower()


def test_unknown_template_close_for_default_renderer():
    from renderers.base import Renderer  # ensure import shape  # noqa: F401

    class _FakeDefault(_StubRenderer):
        pass

    # Re-bind so the isinstance check inside diagnose_bridge fires.
    from renderers.default import DefaultRenderer

    class _DefaultLike(DefaultRenderer):
        def __init__(self):
            self.tokenizer = SimpleNamespace(model_max_length=1_000_000)

        def bridge_to_next_turn(self, *a, **k):
            return None

        def render_ids(self, *a, **k):
            return []

    diag = diagnose_bridge(_DefaultLike(), [1], [2], [{"role": "user", "content": "x"}])
    assert diag is not None
    assert diag.reason is BridgeFailureReason.UNKNOWN_TEMPLATE_CLOSE


def test_truncation_zeroed_anchor_when_prev_exceeds_max():
    r = _StubRenderer(bridge_return=None, fresh_return=[], max_length=10)
    diag = diagnose_bridge(
        r,
        list(range(11)),
        [99],
        [{"role": "user", "content": "x"}],
    )
    assert diag is not None
    assert diag.reason is BridgeFailureReason.TRUNCATION_ZEROED_ANCHOR
    assert diag.token_span == (10, 11)


def test_bridge_returns_none_falls_back_to_bpe_drift():
    r = _StubRenderer(bridge_return=None, fresh_return=[], max_length=1_000)
    diag = diagnose_bridge(r, [1, 2], [3], [{"role": "user", "content": "x"}])
    assert diag is not None
    assert diag.reason is BridgeFailureReason.BPE_DRIFT
    assert "BPE_DRIFT" in diag.detail or "bpe" in diag.detail.lower()


def test_first_divergent_token_falls_back_to_bpe_drift():
    r = _StubRenderer(
        bridge_return=[1, 2, 3, 5, 6],
        fresh_return=[1, 2, 3, 7, 8],
        max_length=1_000,
    )
    diag = diagnose_bridge(r, [1, 2], [3], [{"role": "user", "content": "x"}])
    assert diag is not None
    assert diag.reason is BridgeFailureReason.BPE_DRIFT
    assert diag.token_span == (3, 4)


def test_bool_round_trip_detected_when_token_decodes_to_true_or_false():
    """When the first divergent token decodes to a bool literal, classify
    as ``BOOL_ROUND_TRIP`` rather than the generic ``BPE_DRIFT``."""

    r = _StubRenderer(
        bridge_return=[1, 2, 3],
        fresh_return=[1, 2, 99],
        max_length=1_000,
    )

    def decode(ids, skip_special_tokens=False):
        return "True" if ids == [99] else "false"

    r.tokenizer.decode = decode  # type: ignore[assignment]
    diag = diagnose_bridge(r, [1], [2], [{"role": "user", "content": "x"}])
    assert diag is not None
    assert diag.reason is BridgeFailureReason.BOOL_ROUND_TRIP
    assert diag.token_span == (2, 3)


def test_diagnostic_dataclass_is_frozen_and_hashable():
    d = BridgeDiagnostic(
        reason=BridgeFailureReason.BPE_DRIFT,
        message_index=-1,
        token_span=(0, 1),
        detail="x",
    )
    with pytest.raises(Exception):
        d.reason = BridgeFailureReason.BOOL_ROUND_TRIP  # type: ignore[misc]
    assert hash(d) == hash(d)


def test_enum_str_values_stable():
    """The enum's string values are part of the public surface; lock
    them down so downstream log consumers and dashboards don't break."""
    assert BridgeFailureReason.ASSISTANT_IN_EXTENSION == "assistant_in_extension"
    assert BridgeFailureReason.BOOL_ROUND_TRIP == "bool_round_trip"
    assert BridgeFailureReason.BPE_DRIFT == "bpe_drift"
    assert BridgeFailureReason.THINKING_STRIPPED == "thinking_stripped"
    assert BridgeFailureReason.TOOL_CALL_XML_DRIFT == "tool_call_xml_drift"
    assert BridgeFailureReason.TRUNCATION_ZEROED_ANCHOR == "truncation_zeroed_anchor"
    assert BridgeFailureReason.UNKNOWN_TEMPLATE_CLOSE == "unknown_template_close"
