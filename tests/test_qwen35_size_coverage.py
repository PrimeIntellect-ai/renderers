"""Qwen3.5 size coverage in ``MODEL_RENDERER_MAP``.

Seven Qwen3.5 sizes route to ``Qwen35Renderer``. The 4B / 9B / 35B-A3B /
122B-A10B / 397B-A17B sizes ship one chat template (default
``enable_thinking=true``); the smaller 0.8B / 2B sizes ship the polarity-
flipped variant (default ``enable_thinking=false`` → empty
``<think>\\n\\n</think>\\n\\n`` at the gen-prompt boundary). The renderer
hard-codes this polarity per model (``_ENABLE_THINKING_DEFAULTS``), so
both variants render byte-identical to their own ``apply_chat_template``.

These tests lock in the map and polarity metadata. Byte parity for every size
lives in the unified matrix in ``test_parity.py``.
"""

from __future__ import annotations

import pytest
from parity import MODEL_CATALOG

from renderers import Qwen35Renderer, Qwen35RendererConfig, create_renderer
from renderers.base import MODEL_RENDERER_MAP, load_tokenizer


_QWEN35_IN_MAP = {
    case.model for case in MODEL_CATALOG if case.resolved_renderer == "qwen3.5"
}


def test_map_includes_expected_qwen35_sizes():
    """Every parity-verified Qwen3.5 size routes to the ``qwen3.5`` renderer."""
    for model in _QWEN35_IN_MAP:
        assert MODEL_RENDERER_MAP.get(model) == "qwen3.5", (
            f"{model}: expected to route to 'qwen3.5'"
        )


def test_no_other_qwen35_sizes_silently_added():
    """Catches silent additions: any Qwen3.5 size in the map MUST be in
    ``_QWEN35_IN_MAP`` so the parity barrage below covers it."""
    listed_qwen35 = {
        m
        for m, r in MODEL_RENDERER_MAP.items()
        if r == "qwen3.5" and m.startswith("Qwen/Qwen3.5-")
    }
    assert listed_qwen35 == _QWEN35_IN_MAP, (
        f"Qwen3.5 entries in MODEL_RENDERER_MAP drifted from the parity "
        f"matrix; map={sorted(listed_qwen35)} test={sorted(_QWEN35_IN_MAP)}"
    )


# ---------------------------------------------------------------------------
# Polarity defaults: 0.8B / 2B flip ``enable_thinking`` default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qwen35_model,expected_default",
    [
        ("Qwen/Qwen3.5-0.8B", False),
        ("Qwen/Qwen3.5-2B", False),
        ("Qwen/Qwen3.5-4B", True),
        ("Qwen/Qwen3.5-9B", True),
        ("Qwen/Qwen3.5-35B-A3B", True),
        ("Qwen/Qwen3.5-122B-A10B", True),
        ("Qwen/Qwen3.5-397B-A17B", True),
    ],
)
def test_qwen35_enable_thinking_polarity_default(qwen35_model, expected_default):
    """With no explicit flag, the renderer resolves ``enable_thinking`` from
    the hard-coded per-model default — so big / small sizes each match their
    own template at the gen-prompt boundary."""
    tok = load_tokenizer(qwen35_model)
    renderer = create_renderer(tok, Qwen35RendererConfig())
    assert isinstance(renderer, Qwen35Renderer)
    assert renderer.config.enable_thinking is expected_default, (
        f"{qwen35_model}: expected enable_thinking default {expected_default}, "
        f"got {renderer.config.enable_thinking}"
    )


def test_construction_does_not_call_apply_chat_template():
    """The ``enable_thinking`` default is hard-coded per model, so building a
    ``Qwen35Renderer`` must not probe ``apply_chat_template`` — a
    bring-your-own tokenizer with no chat-template support still works."""

    class _Stub:
        name_or_path = "Qwen/Qwen3.5-0.8B"
        unk_token_id = -1

        def convert_tokens_to_ids(self, token):
            # Any stable non-unk id per token; the renderer only needs the
            # special tokens to resolve to distinct, in-vocab ids.
            return abs(hash(token)) % 1_000_000 + 1

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError(
                "apply_chat_template must not be called at construction"
            )

    renderer = Qwen35Renderer(_Stub())
    # 0.8B is a small size → thinking defaults off, from the hard-coded table.
    assert renderer.config.enable_thinking is False
