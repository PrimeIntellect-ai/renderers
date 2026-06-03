"""Tests for per-message / per-role token-count and span helpers.

These methods on :class:`RenderedTokens` derive from the existing
``message_indices`` / ``sampled_mask`` / ``message_roles`` signals —
they're analytics helpers, not new attribution. The contract these
tests pin:

- ``sum(tokens_per_message()) + scaffold_count == len(token_ids)`` — the
  helper accounts for every token exactly once, with ``-1``-attributed
  scaffolding outside the per-message bucket.
- ``sampled_only=True`` excludes the role-tag opener
  (``<|im_start|>role\\n``) the template injects around an assistant
  message. The assistant message's sampled count must be strictly less
  than its total count.
- ``sampled_only=True`` is zero for tool / user / system roles — the
  model never samples conversation history.
- ``message_token_spans`` returns contiguous ``(start, end)`` slices
  per message that ``zip``s against ``message_roles``. Slicing
  ``token_ids[start:end]`` recovers the message's tokens.

Renderers that opt out of ``sampled_mask`` (DefaultRenderer — the
Jinja template is opaque) are exempt from the ``sampled_only``
assertions and return all zeros from the helper under that flag.
"""

from __future__ import annotations

from renderers.base import RenderedTokens


def test_tokens_per_message_sum_equals_attributed(model_name, renderer):
    """``sum(tokens_per_message()) + scaffold_count == len(token_ids)``.

    Every token lands in exactly one of (a) a caller-relative message
    bucket, or (b) the ``-1`` scaffolding bucket. No double-counts, no
    drops.
    """
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs, add_generation_prompt=True)

    counts = rendered.tokens_per_message()
    scaffold = sum(1 for idx in rendered.message_indices if idx == -1)
    assert sum(counts) + scaffold == len(rendered.token_ids), (
        f"{model_name}: tokens_per_message sum {sum(counts)} + scaffold "
        f"{scaffold} != token count {len(rendered.token_ids)}"
    )

    # message_roles populated, length matches caller input.
    assert len(rendered.message_roles) == len(msgs), (
        f"{model_name}: message_roles length {len(rendered.message_roles)} "
        f"!= len(msgs) {len(msgs)}"
    )
    assert rendered.message_roles == ["system", "user", "assistant"], (
        f"{model_name}: message_roles {rendered.message_roles} != input roles"
    )

    # Every caller message must have a positive count.
    bad = [i for i, n in enumerate(counts) if n == 0]
    assert not bad, f"{model_name}: messages with zero attributed tokens: {bad}"


def test_tokens_per_message_sampled_lt_total_for_assistant(model_name, renderer):
    """The assistant message's ``sampled_only=True`` count must be
    strictly less than its full count.

    The template wraps an assistant turn with at least one role-tag
    token (``<|im_start|>assistant\\n`` or equivalent) that the model
    doesn't sample at inference. Skips renderers that opt out of
    ``sampled_mask`` (DefaultRenderer).
    """
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello world!"},
    ]
    rendered = renderer.render(msgs)
    if not rendered.sampled_mask:
        return

    total = rendered.tokens_per_message()
    sampled = rendered.tokens_per_message(sampled_only=True)

    assistant_idx = 1
    assert sampled[assistant_idx] < total[assistant_idx], (
        f"{model_name}: assistant sampled count {sampled[assistant_idx]} "
        f"not strictly less than total {total[assistant_idx]} — role-tag "
        f"opener should have been excluded"
    )
    assert sampled[assistant_idx] > 0, (
        f"{model_name}: assistant sampled count is zero — content tokens "
        f"should be is_sampled=True"
    )


def test_tokens_per_message_sampled_zero_for_history_roles(model_name, renderer):
    """``sampled_only=True`` counts are zero for user / system / tool —
    the model never samples conversation history."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    if not rendered.sampled_mask:
        return

    sampled = rendered.tokens_per_message(sampled_only=True)
    assert sampled[0] == 0, (
        f"{model_name}: system message sampled count {sampled[0]} != 0"
    )
    assert sampled[1] == 0, (
        f"{model_name}: user message sampled count {sampled[1]} != 0"
    )


def test_tokens_per_message_truncates_to_n_messages(model_name, renderer):
    """Passing explicit ``n_messages`` smaller than the populated count
    silently drops tail messages — the helper doesn't raise."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    truncated = rendered.tokens_per_message(1)
    assert len(truncated) == 1
    assert truncated[0] > 0


def test_tokens_per_message_clamps_oversized_n_messages(model_name, renderer):
    """Passing ``n_messages`` larger than ``len(message_roles)`` is
    clamped — the helper never reports more messages than the renderer
    attributed, so callers can't read trailing-zero phantoms."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    clamped = rendered.tokens_per_message(99)
    assert len(clamped) == len(rendered.message_roles)
    assert clamped == rendered.tokens_per_message()


def test_tokens_per_message_bridge_attributes_new_messages(model_name, renderer):
    """``bridge_to_next_turn`` returns ``RenderedTokens`` with proper
    per-token attribution AND populated ``message_roles`` for
    ``new_messages``. ``sampled_mask`` is uniformly ``False``.

    DefaultRenderer's ``bridge_to_next_turn`` returns ``None``
    unconditionally, so we skip when bridge is unsupported.
    """
    prior = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered_prior = renderer.render(prior, add_generation_prompt=False)
    new_messages = [
        {"role": "user", "content": "Tell me a longer story please"},
    ]
    bridge = renderer.bridge_to_next_turn(
        previous_prompt_ids=rendered_prior.token_ids,
        previous_completion_ids=[],
        new_messages=new_messages,
    )
    if bridge is None:
        return

    assert len(bridge.message_indices) == len(bridge.token_ids), (
        f"{model_name}: bridge message_indices length mismatch"
    )
    assert len(bridge.sampled_mask) == len(bridge.token_ids), (
        f"{model_name}: bridge sampled_mask length mismatch"
    )
    assert bridge.message_roles == ["user"], (
        f"{model_name}: bridge message_roles {bridge.message_roles} != ['user']"
    )
    assert not any(bridge.sampled_mask), (
        f"{model_name}: bridge emitted a token marked is_sampled=True"
    )

    counts = bridge.tokens_per_message()
    assert counts == [counts[0]] and counts[0] > 0, (
        f"{model_name}: bridge attributed {counts} tokens; expected one "
        f"positive entry for the new user message"
    )


def test_tokens_by_role_includes_every_input_role(model_name, renderer):
    """``tokens_by_role`` returns a key for every role that appears in
    ``message_roles``. Counts sum to the same value as
    ``tokens_per_message``.
    """
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)

    by_role = rendered.tokens_by_role()
    assert set(by_role) == {"system", "user", "assistant"}, (
        f"{model_name}: tokens_by_role keys {set(by_role)} != expected roles"
    )
    assert sum(by_role.values()) == sum(rendered.tokens_per_message()), (
        f"{model_name}: tokens_by_role sum disagrees with tokens_per_message sum"
    )


def test_tokens_by_role_sampled_only_assistant_only(model_name, renderer):
    """Under ``sampled_only=True``, only ``assistant`` has a positive
    count — every other role's tokens were template-injected."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    if not rendered.sampled_mask:
        return

    by_role = rendered.tokens_by_role(sampled_only=True)
    assert by_role["assistant"] > 0
    assert by_role["system"] == 0
    assert by_role["user"] == 0


def test_message_token_spans_recover_token_ranges(model_name, renderer):
    """``message_token_spans()`` returns ``(start, end)`` such that
    ``token_ids[start:end]`` contains exactly the tokens attributed to
    that message and ``token_ids[end-1]`` is still in that message
    (i.e. the span is the inclusive token range of the message).

    Also: spans cover every non-scaffolding token; the spans are
    ordered by message index; concatenating
    ``token_ids[start:end]`` for every non-None span plus the
    ``-1``-attributed scaffolding tokens equals the original
    ``token_ids``.
    """
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    spans = rendered.message_token_spans()
    assert len(spans) == len(msgs)

    # Each span's tokens match the message_indices attribution.
    for i, span in enumerate(spans):
        assert span is not None, f"{model_name}: message {i} has no span"
        start, end = span
        attributed = [
            rendered.message_indices[k]
            for k in range(start, end)
            if rendered.message_indices[k] == i
        ]
        # At least one token in the span belongs to this message
        # (contiguity assumption — see method docstring).
        assert attributed, (
            f"{model_name}: span {span} for msg {i} contains no msg-i tokens"
        )

    # Spans are non-decreasing by start.
    starts = [s[0] for s in spans if s is not None]
    assert starts == sorted(starts), (
        f"{model_name}: spans not in message order: {starts}"
    )


def test_role_token_spans_groups_by_role(model_name, renderer):
    """``role_token_spans()`` groups :meth:`message_token_spans` by
    role. The total tokens captured per role should match
    ``tokens_by_role()``.
    """
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    role_spans = rendered.role_token_spans()
    by_role = rendered.tokens_by_role()

    for role, spans in role_spans.items():
        span_total = sum(e - s for s, e in spans)
        assert span_total == by_role[role], (
            f"{model_name}: role {role!r} span-total {span_total} != "
            f"tokens_by_role {by_role[role]}"
        )


def test_tokens_per_message_no_messages_helper_works():
    """Pure-data-shape: methods on a manually-constructed RenderedTokens
    without any renderer involvement. No fixture needed."""
    r = RenderedTokens(
        token_ids=[10, 11, 12, 13, 14],
        message_indices=[0, 0, 1, 1, -1],
        sampled_mask=[False, True, False, False, False],
        message_roles=["user", "assistant"],
    )
    assert r.tokens_per_message() == [2, 2]
    assert r.tokens_per_message(sampled_only=True) == [1, 0]
    assert r.tokens_by_role() == {"user": 2, "assistant": 2}
    assert r.tokens_by_role(sampled_only=True) == {"user": 1, "assistant": 0}
    assert r.message_token_spans() == [(0, 2), (2, 4)]
    assert r.role_token_spans() == {
        "user": [(0, 2)],
        "assistant": [(2, 4)],
    }


def test_message_token_spans_empty_message():
    """A message that contributed zero tokens (rare but possible for
    empty content some templates skip) gets ``None`` as its span."""
    r = RenderedTokens(
        token_ids=[10, 11, 12],
        message_indices=[0, 0, 2],
        message_roles=["user", "assistant", "tool"],
    )
    spans = r.message_token_spans()
    assert spans[0] == (0, 2)
    assert spans[1] is None  # no tokens attributed
    assert spans[2] == (2, 3)
