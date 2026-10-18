"""Tests for ``RenderedTokens.message_tool_names`` and its populating helper.

``message_tool_names`` is a per-message sidecar parallel to
``message_roles``: for each tool-role message in the rendered slice
it carries the tool function name, ``None`` everywhere else. The name
comes from ``msg["name"]`` when set, otherwise from a
``tool_call_id`` join against any prior assistant's ``tool_calls`` in
the same slice. Pure metadata — does not affect the rendered token
stream, does not mutate the caller's messages.

Unit tests below cover the join's case matrix without a tokenizer.
The single integration test runs every renderer in the conftest
matrix to catch any of the ~25 ``RenderedTokens(...)`` construction
sites that might fail to wire the field through.
"""

from __future__ import annotations

from renderers.base import extract_message_tool_names


def test_extract_empty():
    assert extract_message_tool_names([]) == []


def test_extract_caller_provided_name_wins():
    """``msg['name']`` set by the caller is used verbatim — no join attempted."""
    messages = [
        {"role": "tool", "tool_call_id": "c1", "name": "caller_set", "content": "x"},
    ]
    assert extract_message_tool_names(messages) == ["caller_set"]


def test_extract_resolves_from_prior_assistant():
    """Tool message without ``name``: recovered via tool_call_id → assistant.tool_calls."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "screenshot"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert extract_message_tool_names(messages) == [None, None, "screenshot"]


def test_extract_orphan_tool_message_is_none():
    """``tool_call_id`` matching no in-slice assistant resolves to ``None``
    (bridge case: the issuing assistant lives in the prior portion that
    ``new_messages`` doesn't cover).
    """
    messages = [{"role": "tool", "tool_call_id": "orphan", "content": "x"}]
    assert extract_message_tool_names(messages) == [None]


def test_extract_does_not_mutate_caller():
    """Caller's tool message must not gain a ``name`` field after extraction —
    the helper produces a sidecar list, not a mutated view of the input.
    """
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "f"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    extract_message_tool_names(messages)
    assert "name" not in messages[1]


def test_renderer_populates_message_tool_names(model_name, renderer):
    """Every renderer wires ``message_tool_names`` through ``RenderedTokens``.

    Catches missed wire-up at any of the ~25 ``RenderedTokens(...)``
    construction sites across concrete renderers. The input is
    spec-conformant (tool message carries ``tool_call_id`` but no
    ``name``) so the resolution path exercises the internal join.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": {}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    rt = renderer.render(messages)
    assert rt.message_tool_names == [None, None, "screenshot"], (
        f"{model_name}: got {rt.message_tool_names}"
    )
