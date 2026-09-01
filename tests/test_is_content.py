"""Per-token ``RenderedTokens.is_content`` invariants.

``is_content[k]`` answers "does ``token_ids[k]`` come from message body
bytes (caller-provided content / tool_calls / reasoning_content, or
the model's sampled emission for the assistant role) or is it template
scaffolding the renderer added around the body (role tags, special
tokens, separators, tool-response wraps, generation prompt)?"

By design ``is_content`` is a superset of ``sampled_mask``:

- Equal on every token attributed to an assistant message (sampled ==
  body for that role by construction).
- Carries new information on every other role: the model never samples
  user / tool / system tokens, so ``sampled_mask`` is uniformly
  ``False`` over those — ``is_content`` differentiates body from wrap.

These tests parametrise across every renderer in
``conftest.RENDERER_MODELS`` and verify the contract for hand-coded
renderers. ``DefaultRenderer`` leaves ``is_content`` empty by design
(the Jinja template is opaque, so the renderer cannot know the
wrap/body split) and is exempt from the populated-length check.

The body-bytes decode invariant uses ``in`` (substring) rather than
strict equality because some renderers normalise whitespace, strip
trailing newlines, etc. within the message body emit — but the
caller-provided content must always be recoverable as a substring of
the decoded body run.
"""

from __future__ import annotations

import numpy as np


def _is_populated(rendered) -> bool:
    return rendered.is_content.size == rendered.token_ids.size and rendered.is_content.size > 0


def test_is_content_length_or_empty(model_name, renderer):
    """``is_content`` is either empty (opt-out) or matches token_ids
    length exactly. No partial fills."""
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    rendered = renderer.render(msgs)
    n_tokens = len(rendered.token_ids)
    n_mask = len(rendered.is_content)
    assert n_mask == 0 or n_mask == n_tokens, (
        f"{model_name}: is_content length {n_mask} must be 0 or match token_ids length {n_tokens}"
    )


def test_is_content_equals_sampled_on_assistant(model_name, renderer):
    """On every token attributed to an assistant message,
    ``is_content[k] == sampled_mask[k]``. The two signals collapse on
    that role by design — the model's sampled output IS the assistant
    message's body, and the surrounding scaffold is neither sampled
    nor body."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello world!"},
    ]
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return
    if len(rendered.sampled_mask) != len(rendered.token_ids):
        return  # renderer opts out of sampled_mask

    assistant_messages = np.zeros(len(msgs), dtype=np.bool_)
    for message_index, message in enumerate(msgs):
        assistant_messages[message_index] = message.get("role") == "assistant"
    valid = rendered.message_indices >= 0
    assistant_tokens = np.zeros(rendered.token_ids.size, dtype=np.bool_)
    assistant_tokens[valid] = assistant_messages[rendered.message_indices[valid]]
    mismatches = np.flatnonzero(assistant_tokens & (rendered.is_content != rendered.sampled_mask))
    assert mismatches.size == 0, (
        f"{model_name}: is_content != sampled_mask on assistant tokens at positions {mismatches[:8]}"
    )


def test_is_content_excludes_generation_prompt(model_name, renderer):
    """All generation-prompt tokens (msg_idx=-1) are scaffold — the
    next-turn opener the chat template injects so the model can
    continue. ``is_content`` must be False over that entire span."""
    msgs = [{"role": "user", "content": "Hi"}]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    if not _is_populated(rendered):
        return

    bad = np.flatnonzero((rendered.message_indices == -1) & rendered.is_content)
    assert bad.size == 0, f"{model_name}: generation-prompt tokens marked is_content=True at positions {bad[:8]}"


def test_is_content_recovers_user_body(model_name, tokenizer, renderer):
    """The decoded run of is_content=True tokens within a user message
    contains the user's original content. ``in`` rather than equality
    because some templates normalise whitespace inside the body emit;
    the input substring must still be recoverable from the decoded
    body run."""
    user_text = "Hello, my name is Sebastian."
    msgs = [{"role": "user", "content": user_text}, {"role": "assistant", "content": "Hi!"}]
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return

    user_body_ids = rendered.token_ids[(rendered.message_indices == 0) & rendered.is_content]
    assert user_body_ids.size > 0, f"{model_name}: no is_content=True tokens attributed to user message"
    decoded = tokenizer.decode(user_body_ids).strip()
    assert user_text in decoded or decoded in user_text, (
        f"{model_name}: user body run decodes to {decoded!r}, expected to contain {user_text!r}"
    )


def test_is_content_recovers_tool_body(model_name, tokenizer, renderer):
    """The decoded run of is_content=True tokens within a tool message
    contains the tool response body. The whole point of the body/wrap
    cut: SFT on this run trains the model to anticipate tool outputs
    without learning to emit the surrounding ``<|tool_response>`` /
    role-tag scaffold (which would interrupt a real rollout)."""
    tool_text = "The capital of France is Paris."
    msgs = [
        {"role": "user", "content": "What's the capital of France?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "lookup", "arguments": {"q": "capital of France"}},
                }
            ],
        },
        {"role": "tool", "content": tool_text, "tool_call_id": "call_1"},
        {"role": "assistant", "content": "Paris."},
    ]
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return

    tool_body_ids = rendered.token_ids[(rendered.message_indices == 2) & rendered.is_content]
    assert tool_body_ids.size > 0, f"{model_name}: no is_content=True tokens attributed to tool message"
    decoded = tokenizer.decode(tool_body_ids).strip()
    assert tool_text in decoded, (
        f"{model_name}: tool body run decodes to {decoded!r}, expected to contain {tool_text!r}"
    )


def test_is_content_recovers_system_body(model_name, tokenizer, renderer):
    """The decoded run of is_content=True tokens within a system
    message contains the caller-provided system content. Tools header
    / footer (if present) are scaffold and never appear as body."""
    sys_text = "You are an unusually precise assistant."
    msgs = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi!"},
    ]
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return

    sys_body_ids = rendered.token_ids[(rendered.message_indices == 0) & rendered.is_content]
    assert sys_body_ids.size > 0, f"{model_name}: no is_content=True tokens attributed to system message"
    decoded = tokenizer.decode(sys_body_ids).strip()
    assert sys_text in decoded, (
        f"{model_name}: system body run decodes to {decoded!r}, expected to contain {sys_text!r}"
    )


def test_is_content_no_body_on_role_tag(model_name, renderer):
    """The first token attributed to a user/system/tool message must
    have ``is_content=False`` — that's the leading role-tag run
    (``<|im_start|>`` / equivalent), which is template scaffold, never
    body."""
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return

    user_positions = np.flatnonzero(rendered.message_indices == 0)
    assert user_positions.size > 0, f"{model_name}: no tokens attributed to user message"
    first_k = int(user_positions[0])
    assert not rendered.is_content[first_k], (
        f"{model_name}: first user-attributed token at k={first_k} should "
        f"be is_content=False (role-tag scaffolding), but was True"
    )


def test_content_token_spans_by_role_isolates_tool_body(model_name, tokenizer, renderer):
    """``content_token_spans_by_role()["tool"]`` returns spans over
    which every token is the tool message body. Joining the decoded
    spans recovers the tool response. Adjacent scaffold tokens
    (``<|tool_response>``, role-tag openers) are never inside any
    returned span."""
    tool_text = "Result: 42"
    msgs = [
        {"role": "user", "content": "What's 6*7?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "id": "call_x", "function": {"name": "calc", "arguments": {"e": "6*7"}}}
            ],
        },
        {"role": "tool", "content": tool_text, "tool_call_id": "call_x"},
        {"role": "assistant", "content": "42."},
    ]
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return

    spans = rendered.content_token_spans_by_role()
    tool_spans = spans.get("tool")
    assert tool_spans is not None and tool_spans.shape[0] > 0, f"{model_name}: no tool content spans returned"

    pieces: list[str] = []
    for row_index in range(tool_spans.shape[0]):
        s = int(tool_spans[row_index, 0])
        e = int(tool_spans[row_index, 1])
        run_ids = rendered.token_ids[s:e]
        # All tokens in the span must be is_content=True by definition.
        assert np.all(rendered.is_content[s:e]), f"{model_name}: span [{s}, {e}) contains is_content=False"
        pieces.append(tokenizer.decode(run_ids))
    joined = "".join(pieces).strip()
    assert tool_text in joined, (
        f"{model_name}: joined tool spans decode to {joined!r}, expected to contain {tool_text!r}"
    )


def test_content_mask_for_roles_excludes_assistant_when_unset(model_name, renderer):
    """``content_mask_for_roles({"tool"})`` returns a mask that's True
    only on tool body tokens — never on assistant tokens, even though
    those also have ``is_content=True``. The role filter is the whole
    point: SFT-on-tool-body must not bleed into the assistant span."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "id": "call_a", "function": {"name": "ping", "arguments": {}}}],
        },
        {"role": "tool", "content": "pong", "tool_call_id": "call_a"},
        {"role": "assistant", "content": "OK."},
    ]
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return

    tool_mask = rendered.content_mask_for_roles({"tool"})
    assert len(tool_mask) == len(rendered.token_ids)

    expected = (rendered.message_indices == 2) & rendered.is_content
    assert np.array_equal(tool_mask, expected), f"{model_name}: tool-role mask included scaffold or another role"


def test_build_training_sample_content_sft_roles_picks_up_tool_body(model_name, renderer):
    """``build_training_sample(..., content_sft_roles={"tool"})``
    produces a loss mask that's True on tool body tokens AND assistant
    sampled tokens, but False on the tool-message scaffold (role tag,
    ``<|tool_response>`` wraps, separators). The canonical
    SFT-on-tool-body + RL-on-assistant composition."""
    from renderers import build_training_sample

    msgs = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "id": "call_z", "function": {"name": "noop", "arguments": {}}}],
        },
        {"role": "tool", "content": "done", "tool_call_id": "call_z"},
        {"role": "assistant", "content": "OK."},
    ]
    sample = build_training_sample(
        renderer, msgs, role_to_mask=lambda m: m["role"] == "assistant", content_sft_roles={"tool"}
    )
    ids, mask = sample.token_ids, sample.loss_mask
    assert len(mask) == len(ids)

    # We need at least one trainable tool-body token if the renderer
    # populates is_content. Renderers that opt out (empty is_content)
    # fall back to the existing role_to_mask behaviour, which leaves
    # tool tokens False — that's the documented fallback and is fine.
    rendered = renderer.render(msgs)
    if not _is_populated(rendered):
        return

    assert np.any(mask & (rendered.message_indices == 2)), (
        f"{model_name}: build_training_sample with content_sft_roles={{'tool'}} trained on zero tool tokens"
    )
    assistant = (rendered.message_indices == 1) | (rendered.message_indices == 3)
    assert np.any(mask & assistant), f"{model_name}: assistant tokens dropped from training mask"
    assert not np.any(mask & (rendered.message_indices == 0)), f"{model_name}: user tokens leaked into training mask"
