"""Byte-for-byte parity: native (Rust) vs pure-Python.

For every family that has been ported to Rust, build *both* a
pure-Python renderer and a native renderer from the same tokenizer and
assert their outputs are identical across a representative set of
conversation shapes.

This complements two existing parity gates:

- ``tests/test_render_ids.py`` — Python (or, when the env var routes,
  native) vs HuggingFace's ``apply_chat_template``. Catches drift from
  the upstream reference. Run the suite with
  ``RENDERERS_NATIVE=qwen3 pytest tests/test_render_ids.py`` to exercise
  the native path through that gate.
- This file — Python vs native, holding the reference fixed. Catches
  drift between the two implementations even if HF changes its
  template. Cheaper because the HF call isn't on the path.

Both tests require a real ``tokenizer.json`` on disk. The fixtures here
skip with a clear message when the tokenizer can't be located or the
native extension isn't built — so the test file is safe to import in
sandboxed CI where neither is available.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from renderers import _native_router as router

pytestmark = pytest.mark.parity


# ── Test matrix ──────────────────────────────────────────────────────


# (model_id, family-key, extra-kwargs)
NATIVE_PARITY_FAMILIES = [
    ("Qwen/Qwen3-8B", "qwen3", {}),
]


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def native_module():
    mod = router.load_native()
    if mod is None:
        pytest.skip("renderers_native not built; run `maturin develop`")
    return mod


@pytest.fixture(scope="module", params=NATIVE_PARITY_FAMILIES, ids=lambda p: p[1])
def native_pair(request, native_module):
    """Return ``(py_renderer, native_renderer, tokenizer)`` for one family."""
    model_id, family, extra = request.param

    # Locate tokenizer.json on disk. Skip cleanly if not in HF cache —
    # this test is most useful locally with a real model snapshot.
    try:
        from renderers.base import load_tokenizer

        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        pytest.skip(f"could not load tokenizer for {model_id}: {exc}")

    try:
        tok_path = router.resolve_tokenizer_path(tokenizer)
    except Exception as exc:
        pytest.skip(f"could not resolve tokenizer.json for {model_id}: {exc}")
    if not os.path.exists(tok_path):
        pytest.skip(f"tokenizer.json missing on disk at {tok_path}")

    # Build the pure-Python renderer with the env var explicitly off so
    # the ``__new__`` routing doesn't return a native instance.
    saved = os.environ.pop("RENDERERS_NATIVE", None)
    try:
        if family == "qwen3":
            from renderers.qwen3 import Qwen3Renderer

            py_renderer = Qwen3Renderer(tokenizer, **extra)
        else:
            pytest.skip(f"no python builder wired for {family}")
    finally:
        if saved is not None:
            os.environ["RENDERERS_NATIVE"] = saved

    # Build the native renderer directly through the module surface —
    # bypasses the env-var routing entirely.
    if family == "qwen3":
        native_renderer = native_module.Renderer.qwen3(tok_path, **extra)
    else:
        pytest.skip(f"no native builder wired for {family}")

    return py_renderer, native_renderer, tokenizer


# ── Conversation fixtures (a representative cross-section) ───────────


CONVERSATIONS: list[tuple[str, list[dict[str, Any]]]] = [
    (
        "system_and_user",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ],
    ),
    (
        "single_turn",
        [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    ),
    (
        "no_system_message",
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    ),
    (
        "multi_turn",
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ],
    ),
    (
        "reasoning_content_field",
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Simple arithmetic",
                "content": "4",
            },
        ],
    ),
    (
        "tool_call_single",
        [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
        ],
    ),
    (
        "tool_call_with_response",
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "sunny, 22°C"},
            {"role": "assistant", "content": "It's sunny and 22°C in Paris."},
        ],
    ),
]


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case,messages", CONVERSATIONS, ids=lambda x: x if isinstance(x, str) else None)
def test_render_ids_parity(native_pair, case, messages):
    py_renderer, native_renderer, _tok = native_pair
    py_ids = list(py_renderer.render_ids(messages))
    rs_ids = list(native_renderer.render_ids(messages))
    assert py_ids == rs_ids, (
        f"render_ids mismatch for {case}:\n"
        f"  python: {py_ids[:30]}... (len={len(py_ids)})\n"
        f"  native: {rs_ids[:30]}... (len={len(rs_ids)})"
    )


@pytest.mark.parametrize("case,messages", CONVERSATIONS, ids=lambda x: x if isinstance(x, str) else None)
def test_render_ids_with_gen_prompt_parity(native_pair, case, messages):
    py_renderer, native_renderer, _tok = native_pair
    py_ids = list(py_renderer.render_ids(messages, add_generation_prompt=True))
    rs_ids = list(native_renderer.render_ids(messages, add_generation_prompt=True))
    assert py_ids == rs_ids


@pytest.mark.parametrize("case,messages", CONVERSATIONS, ids=lambda x: x if isinstance(x, str) else None)
def test_render_ids_with_tools_parity(native_pair, case, messages):
    py_renderer, native_renderer, _tok = native_pair
    py_ids = list(py_renderer.render_ids(messages, tools=TOOLS))
    rs_ids = list(native_renderer.render_ids(messages, tools=TOOLS))
    assert py_ids == rs_ids


@pytest.mark.parametrize("case,messages", CONVERSATIONS, ids=lambda x: x if isinstance(x, str) else None)
def test_message_indices_parity(native_pair, case, messages):
    """Per-token attribution must match — critical for training loss masks."""
    py_renderer, native_renderer, _tok = native_pair
    py_out = py_renderer.render(messages)
    rs_out = native_renderer.render(messages)
    assert list(py_out.token_ids) == list(rs_out.token_ids)
    assert list(py_out.message_indices) == list(rs_out.message_indices)


def test_stop_token_ids_parity(native_pair):
    py_renderer, native_renderer, _tok = native_pair
    assert list(py_renderer.get_stop_token_ids()) == list(
        native_renderer.get_stop_token_ids()
    )


def test_parse_response_no_tool_calls_parity(native_pair):
    """Parse a simple text completion through both."""
    py_renderer, native_renderer, _tok = native_pair
    # Render a small assistant turn, take the assistant tokens, parse.
    msgs = [{"role": "user", "content": "say hi"}]
    completion_ids = py_renderer.render_ids(
        msgs + [{"role": "assistant", "content": "Hello there!"}]
    )
    # Slice out just the assistant section by re-rendering up to the user.
    prompt_ids = py_renderer.render_ids(msgs, add_generation_prompt=True)
    assistant_ids = completion_ids[len(prompt_ids):]

    py_parsed = py_renderer.parse_response(assistant_ids)
    rs_parsed = native_renderer.parse_response(assistant_ids)
    assert py_parsed.content == rs_parsed.content
    assert (py_parsed.reasoning_content or None) == (rs_parsed.reasoning_content or None)
    assert len(py_parsed.tool_calls) == len(rs_parsed.tool_calls)


def test_bridge_to_next_turn_parity(native_pair):
    py_renderer, native_renderer, _tok = native_pair
    initial = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]
    prev_prompt_ids = py_renderer.render_ids(initial[:-1], add_generation_prompt=True)
    prev_completion_ids = py_renderer.render_ids(initial)[len(prev_prompt_ids):]
    new_messages = [{"role": "user", "content": "Tell me about Rust."}]

    py_b = py_renderer.bridge_to_next_turn(
        prev_prompt_ids, prev_completion_ids, new_messages
    )
    rs_b = native_renderer.bridge_to_next_turn(
        prev_prompt_ids, prev_completion_ids, new_messages
    )

    # Either both return None (refused) or both produce identical tokens.
    if py_b is None:
        assert rs_b is None
        return
    assert rs_b is not None
    assert list(py_b.token_ids) == list(rs_b.token_ids)


def test_bridge_refuses_assistant_in_extension(native_pair):
    py_renderer, native_renderer, _tok = native_pair
    initial = [{"role": "user", "content": "Hi"}]
    prompt_ids = py_renderer.render_ids(initial, add_generation_prompt=True)
    completion_ids = list(py_renderer.get_stop_token_ids())[:1]

    # Assistant in the extension → both must return None.
    assert (
        py_renderer.bridge_to_next_turn(
            prompt_ids,
            completion_ids,
            [{"role": "assistant", "content": "x"}],
        )
        is None
    )
    assert (
        native_renderer.bridge_to_next_turn(
            prompt_ids,
            completion_ids,
            [{"role": "assistant", "content": "x"}],
        )
        is None
    )
