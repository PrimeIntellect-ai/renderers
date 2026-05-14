"""Per-token ``RenderedTokens.sampled_mask`` invariants.

``sampled_mask[k]`` answers "would the model have produced
``token_ids[k]`` at inference?" — distinct from ``message_indices``,
which only attributes a token to a source message. SFT loss masks
should AND both signals so the trainer only sees tokens the model
would actually sample.

These tests parametrise across every renderer in ``conftest.RENDERER_MODELS``
and verify the contract for hand-coded renderers. ``DefaultRenderer``
leaves ``sampled_mask`` empty by design (the Jinja template is opaque,
so the renderer cannot know the prompt / completion split) and is
exempt from the populated-length check.
"""

from __future__ import annotations


def test_sampled_mask_length_or_empty(model_name, renderer):
    """``sampled_mask`` is either empty (opt-out) or matches token_ids
    length exactly. No partial fills."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    n_tokens = len(rendered.token_ids)
    n_mask = len(rendered.sampled_mask)
    assert n_mask == 0 or n_mask == n_tokens, (
        f"{model_name}: sampled_mask length {n_mask} must be 0 or match "
        f"token_ids length {n_tokens}"
    )


def test_sampled_mask_excludes_user_and_system(model_name, renderer):
    """Every token attributed to a user / system / tool message must be
    is_sampled=False — the model never samples conversation history.

    Skips renderers that don't populate sampled_mask (DefaultRenderer)."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    if not rendered.sampled_mask:
        return  # renderer opted out

    bad = []
    for k, (msg_idx, is_sampled) in enumerate(
        zip(rendered.message_indices, rendered.sampled_mask)
    ):
        if msg_idx < 0:
            continue
        role = msgs[msg_idx].get("role")
        if role in ("user", "system", "tool") and is_sampled:
            bad.append((k, role, msg_idx))
    assert not bad, (
        f"{model_name}: non-assistant tokens marked is_sampled=True "
        f"(k, role, msg_idx): {bad[:8]}"
    )


def test_sampled_mask_excludes_generation_prompt(model_name, renderer):
    """All generation-prompt tokens (msg_idx=-1 when add_generation_prompt=True)
    must be is_sampled=False — they are template-injected scaffolding the
    model continues from, not samples."""
    msgs = [{"role": "user", "content": "Hi"}]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    if not rendered.sampled_mask:
        return

    bad = [
        k
        for k, (msg_idx, is_sampled) in enumerate(
            zip(rendered.message_indices, rendered.sampled_mask)
        )
        if msg_idx == -1 and is_sampled
    ]
    assert not bad, (
        f"{model_name}: generation-prompt tokens marked is_sampled=True "
        f"at positions {bad[:8]}"
    )


def test_sampled_mask_assistant_role_tag_excluded(model_name, renderer):
    """The leading ``<|im_start|>{role}\\n`` (or equivalent role-tag run)
    on an assistant message must be is_sampled=False — the chat template
    emits it as part of the generation prompt at inference, so the model
    never samples it. Asserts that the *first* token attributed to an
    assistant message is is_sampled=False; the last token (turn-close
    signal) should be is_sampled=True."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello world!"},
    ]
    rendered = renderer.render(msgs)
    if not rendered.sampled_mask:
        return

    assistant_positions = [
        k for k, idx in enumerate(rendered.message_indices) if idx == 1
    ]
    assert assistant_positions, (
        f"{model_name}: no tokens attributed to assistant message"
    )
    first_k = assistant_positions[0]
    assert not rendered.sampled_mask[first_k], (
        f"{model_name}: first assistant-attributed token at k={first_k} "
        f"should be is_sampled=False (role-tag scaffolding), but was True"
    )

    # At least one assistant-attributed token must be is_sampled=True
    # (the content + turn close — otherwise the loss mask is empty).
    any_sampled = any(rendered.sampled_mask[k] for k in assistant_positions)
    assert any_sampled, (
        f"{model_name}: no assistant tokens marked is_sampled=True — the "
        f"resulting SFT loss mask would be empty"
    )
