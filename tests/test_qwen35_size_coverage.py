"""Qwen3.5 size coverage in ``MODEL_RENDERER_MAP``.

Seven Qwen3.5 sizes route to ``Qwen35Renderer``. The 4B / 9B / 35B-A3B /
122B-A10B / 397B-A17B sizes ship one chat template (default
``enable_thinking=true``); the smaller 0.8B / 2B sizes ship the polarity-
flipped variant (default ``enable_thinking=false`` → empty
``<think>\\n\\n</think>\\n\\n`` at the gen-prompt boundary). The renderer
detects polarity from the tokenizer's chat_template at construction, so
both variants render byte-identical to their own
``apply_chat_template``.

These tests lock in (a) the exact set of Qwen3.5 sizes in the map and
(b) byte parity for every one of them across representative
conversations including the ``add_generation_prompt=True`` boundary
where the polarity divergence shows up.
"""

from __future__ import annotations

import pytest

from renderers import Qwen35Renderer, create_renderer
from renderers.base import MODEL_RENDERER_MAP, load_tokenizer


_QWEN35_IN_MAP = {
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
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
# Polarity auto-detection: 0.8B / 2B flip ``enable_thinking`` default.
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
def test_qwen35_enable_thinking_polarity_autodetected(qwen35_model, expected_default):
    """The renderer's ``_enable_thinking`` resolves to the chat template's
    own default when no explicit flag is passed — so big / small sizes
    each match their own template at the gen-prompt boundary."""
    tok = load_tokenizer(qwen35_model)
    renderer = create_renderer(tok, renderer="qwen3.5")
    assert isinstance(renderer, Qwen35Renderer)
    assert renderer._enable_thinking is expected_default, (
        f"{qwen35_model}: expected enable_thinking default {expected_default}, "
        f"got {renderer._enable_thinking}"
    )


# ---------------------------------------------------------------------------
# Byte parity for each in-map Qwen3.5 size.
# ---------------------------------------------------------------------------


_PARITY_CASES = [
    pytest.param(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi."},
        ],
        False,
        id="system_user",
    ),
    pytest.param(
        [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        False,
        id="single_turn",
    ),
    pytest.param(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi."},
        ],
        True,
        id="with_gen_prompt",
    ),
    pytest.param(
        [
            {"role": "user", "content": "Solve."},
            {
                "role": "assistant",
                "content": "42",
                "reasoning_content": "Think hard.",
            },
            {"role": "user", "content": "Why?"},
            {
                "role": "assistant",
                "content": "Because.",
                "reasoning_content": "More thinking.",
            },
        ],
        False,
        id="with_reasoning",
    ),
]


@pytest.mark.parametrize("qwen35_model", sorted(_QWEN35_IN_MAP))
@pytest.mark.parametrize("messages,add_gen_prompt", _PARITY_CASES)
def test_qwen35_size_parity_with_apply_chat_template(
    qwen35_model, messages, add_gen_prompt
):
    """Each in-map Qwen3.5 size renders byte-identical to its own
    ``apply_chat_template`` output. Locks in the property that lets us
    share ``Qwen35Renderer`` across all seven sizes — the polarity
    flip on 0.8B / 2B is absorbed by the constructor's auto-detect."""
    tok = load_tokenizer(qwen35_model)
    renderer = create_renderer(tok, renderer="qwen3.5")
    assert isinstance(renderer, Qwen35Renderer)

    ours = renderer.render_ids(messages, add_generation_prompt=add_gen_prompt)
    theirs = list(
        tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=add_gen_prompt,
        )
    )
    assert ours == theirs, (
        f"{qwen35_model}: Qwen35Renderer diverges from apply_chat_template "
        f"(add_generation_prompt={add_gen_prompt})"
    )
