"""Per-token attribution invariants for ``RenderedTokens.message_indices``.

Why this file exists
--------------------
The matrix in ``test_parity`` checks token-id parity against
``apply_chat_template`` — useful, but only for token *bytes*. The
per-token attribution (``RenderedTokens.message_indices``) was never
covered, even though it directly drives the loss mask in
``build_training_sample``.

That gap surfaced through two real bugs:

- ``KimiK2Renderer.render``'s unknown-role fallback emitted the closing
  ``<|im_end|>`` with the post-normalisation index ``i`` instead of the
  caller-relative ``oi``.
- ``KimiK25Renderer.render`` used the raw post-normalisation index
  everywhere, shifting *every* index by one whenever auto-system
  injection happened.

In both cases token IDs were unchanged (so render-parity still passed),
but ``message_indices`` pointed at wrong (or out-of-range) messages.

Tests in this file:

1. ``test_message_indices_in_range`` — parametrised via the conftest
   matrix. Renders a no-system multi-role conversation and asserts the
   contract: every entry in ``message_indices`` is either ``-1``
   (structural scaffolding) or a valid caller-relative index, and every
   caller message contributes at least one token. The no-system input
   forces renderers that auto-inject default system messages
   (Kimi K2 / K2.5 / K2.6) into the injection path — the path where the
   post-normalisation index can shift away from the caller-relative one.

2. ``test_kimi_k2_unknown_role_message_indices`` — Kimi-K2-specific
   regression: triggers auto-system injection plus an unknown role,
   which is exactly the path the original bug was on. Without the fix
   this test catches a stray index that points past the caller's last
   message.
"""

from __future__ import annotations

import pytest

from renderers import build_training_sample
from renderers.glm45 import GLM45Renderer
from renderers.glm5 import GLM5Renderer
from tests.reference_rendering import render_reference


def test_glm_tool_stop_ownership(model_name, tokenizer, renderer):
    """The first emitted observation closes the assistant, even after reordering."""
    if not isinstance(renderer, (GLM5Renderer, GLM45Renderer)):
        pytest.skip("GLM turn-closing role markers")
    messages = [
        {"role": "user", "content": "Look up both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": f"f{i}", "arguments": {}},
                }
                for i in (1, 2)
            ],
        },
        {
            "role": "tool",
            "content": "Second result",
            "tool_call_id": "c2",
            "name": "f2",
        },
        {"role": "tool", "content": "First result", "tool_call_id": "c1", "name": "f1"},
        {"role": "assistant", "content": "Done"},
    ]
    rendered = renderer.render(messages)
    assert rendered.token_ids == render_reference(tokenizer, messages)
    observation = tokenizer.encode("<|observation|>", add_special_tokens=False)[0]
    positions = [
        k for k, token in enumerate(rendered.token_ids) if token == observation
    ]
    first, *rest = positions
    assert rendered.message_indices[first] == 1
    assert rendered.sampled_mask[first] and rendered.is_content[first]
    assert not any(rendered.sampled_mask[k] for k in rest)
    assert not rendered.content_mask_for_roles({"tool"})[first]
    span = rendered.message_token_spans()[1]
    assert span is not None
    assert all(index == 1 for index in rendered.message_indices[span[0] : span[1]])
    for selected in (False, True):
        sample = build_training_sample(
            renderer,
            messages,
            message_loss_mask=[0, selected, not selected, not selected, 0],
            content_sft_roles={"tool"},
            ensure_final_stop=True,
        )
        assert sample.loss_mask[first] == selected
        for k, is_tool_body in enumerate(rendered.content_mask_for_roles({"tool"})):
            if is_tool_body:
                assert sample.loss_mask[k] == (not selected)
        assert not sample.loss_mask[-1]


def test_message_indices_in_range(model_name, renderer):
    """Every emitted token's ``message_indices`` must be in
    ``[-1, len(messages))``. ``-1`` is the documented sentinel for
    structural scaffolding (e.g. trailing generation prompt).

    Uses a no-system input so renderers that auto-inject default system
    messages (Kimi K2 / K2.5 / K2.6) actually take the injection path —
    that's the path where the post-normalisation index can shift away
    from the caller-relative one and the bug surfaces."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    n = len(msgs)

    assert len(rendered.token_ids) == len(rendered.message_indices), (
        f"{model_name}: token_ids and message_indices length mismatch"
    )
    bad = [
        (k, idx)
        for k, idx in enumerate(rendered.message_indices)
        if not (idx == -1 or 0 <= idx < n)
    ]
    assert not bad, (
        f"{model_name}: out-of-range message_indices entries (k, idx): {bad[:8]}"
    )

    # Every caller message must contribute at least one token.
    seen = set(rendered.message_indices)
    missing = [k for k in range(n) if k not in seen]
    assert not missing, (
        f"{model_name}: messages not represented in message_indices: {missing}"
    )


def test_kimi_k2_unknown_role_message_indices():
    """Regression for the ``i`` vs ``oi`` mistake in
    ``KimiK2Renderer.render``'s unknown-role fallback. To surface, we
    need (a) auto-system injection (no system in the input) and (b) an
    unknown role hitting the fallback, and (c) the unknown role at the
    *last* caller position so the post-normalisation index points past
    the caller's input length.

    Layout under test (caller indices on the right):

        normalized[0] = auto-injected system     (oi = -1)
        normalized[1] = user                     (oi = 0)
        normalized[2] = developer  (unknown)     (oi = 1)   ← bug emits i=2

    With the bug, ``<|im_end|>`` on the developer message is emitted
    with the post-normalisation index ``i=2`` — out of range for a
    caller list of length 2.
    """
    from renderers import create_renderer
    from renderers.base import load_tokenizer

    tok = load_tokenizer("moonshotai/Kimi-K2-Instruct")
    renderer = create_renderer(tok)

    msgs = [
        {"role": "user", "content": "hi"},
        # Unknown role hits the system-style fallback in KimiK2Renderer.render.
        {"role": "developer", "content": "internal note"},
    ]
    rendered = renderer.render(msgs)

    n = len(msgs)
    bad = [
        (k, idx)
        for k, idx in enumerate(rendered.message_indices)
        if not (idx == -1 or 0 <= idx < n)
    ]
    assert not bad, (
        f"out-of-range message_indices on Kimi K2 unknown-role fallback "
        f"(k, idx): {bad[:8]}"
    )


def test_default_renderer_indices_match_tokens_when_template_shrinks():
    """``DefaultRenderer`` must not out-run ``token_ids`` on a non-monotone
    template.

    Jinja chat templates are not prefix-monotone: a cumulative render can be
    shorter than the previous one, because Qwen3 drops a historical
    assistant's reasoning block once a later user query arrives. Building
    ``message_indices`` by assuming strict extension produced a list longer
    than ``token_ids``, which silently propagated into a mismatched
    ``loss_mask`` from ``build_training_sample``.

    ``DefaultRenderer`` is not part of the ``conftest`` parametrisation
    matrix, so the shared ``test_message_indices_in_range`` never exercised
    it — this test covers the renderer directly.
    """
    from renderers.base import build_training_sample, load_tokenizer
    from renderers.default import DefaultRenderer

    tok = load_tokenizer("Qwen/Qwen3-8B")
    renderer = DefaultRenderer(tok)

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "Two plus two is four.",
            "content": "4",
        },
        {"role": "user", "content": ""},
    ]

    # Guard the premise: the third cumulative render really is shorter.
    lengths = [
        len(renderer.render_ids(messages[: i + 1])) for i in range(len(messages))
    ]
    assert lengths[-1] < lengths[-2], (
        f"premise broken: cumulative renders {lengths} are prefix-monotone, "
        "so this test no longer exercises the shrinking path"
    )

    for add_generation_prompt in (False, True):
        rendered = renderer.render(
            messages, add_generation_prompt=add_generation_prompt
        )
        assert len(rendered.message_indices) == len(rendered.token_ids), (
            "message_indices length "
            f"{len(rendered.message_indices)} != token_ids length "
            f"{len(rendered.token_ids)} (add_generation_prompt="
            f"{add_generation_prompt})"
        )

    sample = build_training_sample(
        renderer, messages, role_to_mask=lambda m: m.get("role") == "assistant"
    )
    assert len(sample.loss_mask) == len(sample.token_ids), (
        f"loss_mask length {len(sample.loss_mask)} != token_ids length "
        f"{len(sample.token_ids)}"
    )
