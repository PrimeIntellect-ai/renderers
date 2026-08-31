"""Llama-3 renderer coverage.

Covers ``Llama3Renderer`` and the ``meta-llama/Llama-3.2-{1B,3B}-Instruct``
entries in ``MODEL_RENDERER_MAP``. ``load_tokenizer`` uses the
unrestricted ``unsloth/Llama-3.2-{1B,3B}-Instruct`` mirrors underneath
(verified byte-identical chat templates) so CI doesn't need an HF token
with Meta license access.
"""

from __future__ import annotations

import pytest
from parity import models_for

from renderers import Llama3Renderer, Llama3RendererConfig, create_renderer
from renderers.base import (
    MODEL_RENDERER_MAP,
    ParsedResponse,
    TOKENIZER_SOURCE_OVERRIDES,
    ToolCallParseStatus,
    load_tokenizer,
)

# Pinned date for byte-parity tests. Matches the chat template's
# strftime fallback so we don't have to override on the apply side.
_PINNED_DATE = "26 Jul 2024"

_MODEL_PAIRS = [
    (case.model, TOKENIZER_SOURCE_OVERRIDES[case.model])
    for case in models_for("llama-checkpoints")
]


@pytest.fixture(scope="module", params=_MODEL_PAIRS, ids=[m for m, _ in _MODEL_PAIRS])
def llama_pair(request):
    canonical, mirror = request.param
    tok = load_tokenizer(canonical)
    renderer = Llama3Renderer(tok, Llama3RendererConfig(date_string=_PINNED_DATE))
    return canonical, mirror, tok, renderer


# ---------------------------------------------------------------------------
# MODEL_RENDERER_MAP shape
# ---------------------------------------------------------------------------


def test_canonical_meta_llama_paths_route_to_llama_3():
    for canonical, _ in _MODEL_PAIRS:
        assert MODEL_RENDERER_MAP.get(canonical) == "llama-3", (
            f"{canonical}: expected to route to 'llama-3'"
        )


def test_create_renderer_via_explicit_config(llama_pair):
    """``Llama3RendererConfig`` resolves to Llama3Renderer in the registry."""
    _, _, tok, _ = llama_pair
    r = create_renderer(tok, Llama3RendererConfig())
    assert isinstance(r, Llama3Renderer)


def test_create_renderer_auto_resolves_after_mirror_load(llama_pair):
    """``load_tokenizer(canonical_meta_id)`` loads from the unrestricted
    mirror but preserves the canonical name needed for auto-resolution."""
    canonical, _, tok, _ = llama_pair
    assert tok.name_or_path == canonical
    assert isinstance(create_renderer(tok), Llama3Renderer)


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------


def test_default_date_matches_chat_template_strftime_fallback(llama_pair):
    """Default ``date_string`` is ``"26 Jul 2024"`` so output stays
    deterministic without an explicit override."""
    _, _, tok, _ = llama_pair
    r = Llama3Renderer(tok)
    assert r.config.date_string == _PINNED_DATE


def test_preserve_thinking_flags_are_noops(llama_pair):
    """Llama-3 has no reasoning channel, so any ``thinking_retention``
    level is accepted but never changes the token stream — the same
    never-preserves contract as Kimi-K2 / Qwen3-VL. (Cross-renderer
    coverage lives in tests/test_preserve_thinking.py.)"""
    _, _, tok, _ = llama_pair
    msgs = [
        {"role": "user", "content": "Hi."},
        {
            "role": "assistant",
            "reasoning_content": "internal musings",
            "content": "Hello!",
        },
    ]
    base = Llama3Renderer(tok).render_ids(msgs)
    for level in ("tool_cycle", "all"):
        r = Llama3Renderer(tok, Llama3RendererConfig(thinking_retention=level))
        assert r.config.thinking_retention == level
        assert r.render_ids(msgs) == base, (
            f"thinking_retention={level!r} must be a no-op for Llama-3"
        )


# ---------------------------------------------------------------------------
# Byte parity vs apply_chat_template
# ---------------------------------------------------------------------------


def _expected(tok, messages, **kwargs):
    kwargs.setdefault("add_generation_prompt", False)
    kwargs.setdefault("date_string", _PINNED_DATE)
    return list(
        tok.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
    )


def test_parity_tool_response_dict_content(llama_pair):
    """Tool response with mapping content goes through ``tojson`` in the
    template; the renderer's ``_tool_response_str`` mirrors that."""
    _, _, tok, r = llama_pair
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
        },
        {"role": "tool", "content": {"k": "v", "n": 42}},
        {"role": "assistant", "content": "ok"},
    ]
    assert r.render_ids(msgs) == _expected(tok, msgs)


def test_render_raises_on_multiple_tool_calls(llama_pair):
    """Llama-3 chat template explicitly raises on >1 tool call per turn —
    the renderer mirrors that contract."""
    _, _, _, r = llama_pair
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "f", "arguments": {}}},
                {"function": {"name": "g", "arguments": {}}},
            ],
        },
    ]
    with pytest.raises(ValueError, match="single tool call"):
        r.render_ids(msgs)


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


def _tokens_for(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def test_parse_response_plain_content(llama_pair):
    _, _, tok, r = llama_pair
    ids = _tokens_for(tok, "Hello, world!") + [r._eot]
    out = r.parse_response(ids)
    assert isinstance(out, ParsedResponse)
    assert out.content == "Hello, world!"
    assert out.tool_calls == []
    assert out.reasoning_content is None


def test_parse_response_tool_call(llama_pair):
    _, _, tok, r = llama_pair
    body = '{"name": "get_weather", "parameters": {"city": "NYC"}}'
    ids = _tokens_for(tok, body) + [r._eot]
    out = r.parse_response(ids)
    assert out.content == ""
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}


def test_parse_response_malformed_tool_call_falls_through_to_content(llama_pair):
    """A body that LOOKS like a tool call but doesn't parse should land
    in content rather than dropping silently."""
    _, _, tok, r = llama_pair
    body = '{"name": "x", broken'
    ids = _tokens_for(tok, body) + [r._eot]
    out = r.parse_response(ids)
    assert out.tool_calls == []
    assert "{" in out.content


# ---------------------------------------------------------------------------
# Bridge contract
# ---------------------------------------------------------------------------


def _simulate_prior_turn(r):
    prior = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ]
    asst = [{"role": "assistant", "content": "Hello!"}]

    prev_prompt = r.render_ids(prior, add_generation_prompt=True)
    full = r.render_ids(prior + asst, add_generation_prompt=False)
    prev_completion = list(full[len(prev_prompt) :])

    stop = set(r.get_stop_token_ids())
    last = -1
    for i in range(len(prev_completion) - 1, -1, -1):
        if prev_completion[i] in stop:
            last = i
            break
    if last >= 0:
        prev_completion = prev_completion[: last + 1]
    return prev_prompt, prev_completion


def test_bridge_extends_prev_verbatim_on_clean_stop(llama_pair):
    _, _, _, r = llama_pair
    prev_prompt, prev_completion = _simulate_prior_turn(r)
    new_messages = [{"role": "user", "content": "What's 2+2?"}]
    bridged = r.bridge_to_next_turn(prev_prompt, prev_completion, new_messages)
    assert bridged is not None
    prev = prev_prompt + prev_completion
    assert bridged.token_ids[: len(prev)] == prev
    assert len(bridged.token_ids) > len(prev)


def test_bridge_matches_fresh_render_on_clean_stop(llama_pair):
    """The whole point of the bridge: it must produce the same tokens as
    a fresh render of the full message list — except sampled tokens are
    kept verbatim rather than re-rendered."""
    _, _, _, r = llama_pair
    prior = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ]
    asst = [{"role": "assistant", "content": "Hello!"}]
    new_messages = [{"role": "user", "content": "What's 2+2?"}]

    prev_prompt, prev_completion = _simulate_prior_turn(r)
    bridged = r.bridge_to_next_turn(prev_prompt, prev_completion, new_messages)
    fresh = r.render_ids(prior + asst + new_messages, add_generation_prompt=True)
    assert bridged.token_ids == fresh


def test_bridge_rejects_assistant_in_extension(llama_pair):
    _, _, _, r = llama_pair
    prev_prompt, prev_completion = _simulate_prior_turn(r)
    bridged = r.bridge_to_next_turn(
        prev_prompt,
        prev_completion,
        [{"role": "assistant", "content": "forbidden"}],
    )
    assert bridged is None


def test_bridge_synthesises_close_on_truncation(llama_pair):
    _, _, _, r = llama_pair
    prev_prompt, prev_completion = _simulate_prior_turn(r)
    trunc = prev_completion[:-1]
    if not trunc:
        pytest.skip("simulated prior had no completion tokens to truncate")
    bridged = r.bridge_to_next_turn(
        prev_prompt, trunc, [{"role": "user", "content": "ping"}]
    )
    assert bridged is not None
    base = prev_prompt + trunc
    assert bridged.token_ids[: len(base)] == base
    assert len(bridged.token_ids) > len(base)
