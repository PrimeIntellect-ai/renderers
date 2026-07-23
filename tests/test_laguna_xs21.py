"""Laguna-XS-2.1 focused tests.

The shared matrices (conftest / config-parity) already assert byte
parity on the common shapes; this file pins the behaviours specific to
the XS-2.1 template that those matrices don't generate:

- the empty-system opt-out (and the empty ``<system></system>`` block
  under ``enable_thinking``),
- verbatim content / reasoning (no whitespace normalisation),
- reasoning gated on ``enable_thinking`` rather than on the data,
- packed tool-call args,
- parse round-trips under ``enable_thinking=True`` (the shared roundtrip
  matrix runs the default config, where the template drops reasoning at
  render time, so reasoning can't round-trip there by design),
- bridge extensions matching a fresh render byte-for-byte.
"""

from __future__ import annotations

from functools import lru_cache

from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import LagunaXS21RendererConfig
from renderers.laguna_xs2 import LagunaXS21Renderer

_MODEL = "poolside/Laguna-XS-2.1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "days": {"type": "integer", "description": "Forecast days"},
                },
                "required": ["city"],
            },
        },
    }
]


@lru_cache(maxsize=None)
def _tok():
    return load_tokenizer(_MODEL)


def _renderer(**config_kwargs) -> LagunaXS21Renderer:
    renderer = create_renderer(_tok(), LagunaXS21RendererConfig(**config_kwargs))
    assert isinstance(renderer, LagunaXS21Renderer)
    return renderer


def _expected(msgs, *, tools=None, add_generation_prompt=False, **template_kwargs):
    return list(
        _tok().apply_chat_template(
            msgs,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
    )


# ── Variant-specific byte parity ──────────────────────────────────────


def test_empty_system_opts_out_of_system_block():
    """An empty caller system message suppresses the default system
    prompt; with no tools and no thinking there is no <system> block."""
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hi"},
    ]
    r = _renderer()
    assert r.render_ids(msgs, add_generation_prompt=True) == _expected(
        msgs, add_generation_prompt=True
    )
    text = _tok().decode(r.render_ids(msgs, add_generation_prompt=True))
    assert "<system>" not in text


def test_empty_system_with_thinking_renders_empty_block():
    """enable_thinking alone forces the <system> block, even when the
    empty caller system message removed all of its content."""
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hi"},
    ]
    r = _renderer(enable_thinking=True)
    ours = r.render_ids(msgs, add_generation_prompt=True)
    assert ours == _expected(msgs, add_generation_prompt=True, enable_thinking=True)
    assert "<system></system>\n" in _tok().decode(ours)


def test_empty_system_with_tools_glues_tools_header():
    """With the system content opted out, the tools section opens the
    block directly: ``<system>### Tools`` with no separating newlines."""
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Weather?"},
    ]
    r = _renderer()
    ours = r.render_ids(msgs, tools=TOOLS, add_generation_prompt=True)
    assert ours == _expected(msgs, tools=TOOLS, add_generation_prompt=True)
    assert "<system>### Tools" in _tok().decode(ours)


def test_reasoning_dropped_without_thinking():
    """enable_thinking=False renders a bare ``</think>`` and drops the
    message's reasoning entirely."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Simple arithmetic", "content": "4"},
    ]
    r = _renderer()
    ours = r.render_ids(msgs)
    assert ours == _expected(msgs)
    assert "Simple arithmetic" not in _tok().decode(ours)


def test_reasoning_rendered_verbatim_with_thinking():
    """enable_thinking=True wraps the reasoning verbatim — whitespace
    included, empty included."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "\n  spaced reasoning  \n",
            "content": "4",
        },
    ]
    r = _renderer(enable_thinking=True)
    ours = r.render_ids(msgs)
    assert ours == _expected(msgs, enable_thinking=True)
    assert "<think>\n  spaced reasoning  \n</think>" in _tok().decode(ours)

    no_reasoning = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    ours = r.render_ids(no_reasoning)
    assert ours == _expected(no_reasoning, enable_thinking=True)
    assert "<think></think>" in _tok().decode(ours)


def test_content_rendered_verbatim():
    """Assistant content is not stripped or newline-normalised."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "\n  keep my whitespace  \n"},
    ]
    r = _renderer()
    assert r.render_ids(msgs) == _expected(msgs)


def test_multiple_tool_calls_packed_args():
    """Tool-call args pack tightly with mixed value types (strings
    verbatim, non-strings via tojson)."""
    msgs = [
        {"role": "user", "content": "Compare Tokyo and Paris"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Tokyo", "days": 3},
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                },
            ],
        },
        {"role": "tool", "content": "Sunny"},
        {"role": "tool", "content": "Rainy"},
    ]
    r = _renderer()
    ours = r.render_ids(msgs, tools=TOOLS, add_generation_prompt=True)
    assert ours == _expected(msgs, tools=TOOLS, add_generation_prompt=True)
    assert (
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
    ) in _tok().decode(ours)


def test_later_system_message_renders_in_loop():
    """A system message after the first renders inline as
    ``<system>{content}</system>``; only the leading one is absorbed
    into the header."""
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "system", "content": "Now be terse."},
        {"role": "user", "content": "Why?"},
    ]
    r = _renderer()
    assert r.render_ids(msgs, add_generation_prompt=True) == _expected(
        msgs, add_generation_prompt=True
    )


# ── Masks ─────────────────────────────────────────────────────────────


def test_assistant_prefill_tokens_unsampled():
    """The generation prompt is ``<assistant>`` plus the mode's think
    tag — in a re-render both are scaffold, everything after (through
    ``</assistant>``) is sampled, and the inter-turn newline is not."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "greeting", "content": "Hello!"},
    ]
    for enable_thinking in (False, True):
        r = _renderer(enable_thinking=enable_thinking)
        rendered = r.render(msgs)
        positions = [k for k, i in enumerate(rendered.message_indices) if i == 1]
        sampled = [rendered.sampled_mask[k] for k in positions]
        # <assistant> + (<think> | </think>) prefill.
        assert sampled[:2] == [False, False]
        # Everything between the prefill and the inter-turn newline is
        # the model's emission, ending with the sampled </assistant>.
        assert all(sampled[2:-1]), sampled
        assert sampled[-1] is False  # trailing "\n"
        assert [rendered.is_content[k] for k in positions] == sampled


# ── Parse ─────────────────────────────────────────────────────────────


def _completion_ids(text: str) -> list[int]:
    return _tok().encode(text, add_special_tokens=False)


def test_parse_no_think_completion():
    """Under the no-think gen prompt (``</think>`` prefilled), the
    completion is pure content up to ``</assistant>``."""
    parsed = _renderer().parse_response(_completion_ids("Four.</assistant>"))
    assert parsed.content == "Four."
    assert parsed.reasoning_content is None
    assert not parsed.tool_calls


def test_parse_preserves_newlines_verbatim():
    """The template renders content verbatim, so the parse must not
    strip the newlines the model emitted."""
    parsed = _renderer(enable_thinking=True).parse_response(
        _completion_ids("Chain of thought</think>\nThe answer.\n</assistant>")
    )
    assert parsed.reasoning_content == "Chain of thought"
    assert parsed.content == "\nThe answer.\n"


def test_parse_tool_call_packed_args():
    parsed = _renderer().parse_response(
        _completion_ids(
            "I will check.<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
            "</tool_call></assistant>"
        ),
        tools=TOOLS,
    )
    assert parsed.content == "I will check."
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Tokyo", "days": 3}


def test_roundtrip_reasoning_with_thinking():
    """render → parse recovers reasoning and content under
    enable_thinking=True (the mode where the template keeps them)."""
    r = _renderer(enable_thinking=True)
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    msg = {
        "role": "assistant",
        "reasoning_content": "Two plus two equals four.",
        "content": "The answer is four.",
    }
    pp = r.render_ids(prompt, add_generation_prompt=True)
    full = r.render_ids([*prompt, msg])
    completion = full[len(pp) :]
    parsed = r.parse_response(completion)
    assert parsed.reasoning_content == "Two plus two equals four."
    assert parsed.content == "The answer is four."


# ── Bridge ────────────────────────────────────────────────────────────


def _bridge_case(r, prev, asst, ext, *, tools=None, **template_kwargs):
    """Return (bridged, fresh) token ids for extending prev+asst by ext."""
    pp = r.render_ids(prev, tools=tools, add_generation_prompt=True)
    full_turn = r.render_ids([*prev, asst], tools=tools)
    # The turn ends "</assistant>\n"; the model's completion stops at
    # </assistant>, so drop the template's inter-turn newline token.
    completion = full_turn[len(pp) : -1]
    assert completion[-1] == r._assistant_end
    bridged = r.bridge_to_next_turn(pp, completion, ext, tools=tools)
    fresh = r.render_ids([*prev, asst, *ext], tools=tools, add_generation_prompt=True)
    return bridged, fresh


def test_bridge_user_extension_matches_full_render():
    prev = [{"role": "user", "content": "Hi"}]
    asst = {"role": "assistant", "content": "Hello!"}
    ext = [{"role": "user", "content": "Bye"}]
    for enable_thinking in (False, True):
        r = _renderer(enable_thinking=enable_thinking)
        bridged, fresh = _bridge_case(r, prev, asst, ext)
        assert bridged is not None
        assert bridged.token_ids == fresh


def test_bridge_tool_extension_matches_full_render():
    prev = [{"role": "user", "content": "Weather in Tokyo, then Paris?"}]
    asst = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
        ],
    }
    ext = [
        {"role": "tool", "content": "Sunny"},
        {"role": "user", "content": "And Paris?"},
        {"role": "tool", "content": "Rainy"},
    ]
    r = _renderer()
    bridged, fresh = _bridge_case(r, prev, asst, ext, tools=TOOLS)
    assert bridged is not None
    assert bridged.token_ids == fresh


def test_bridge_synthesizes_close_on_truncation():
    prev = [{"role": "user", "content": "Hi"}]
    asst = {"role": "assistant", "content": "Hello!"}
    ext = [{"role": "user", "content": "Bye"}]
    r = _renderer()
    pp = r.render_ids(prev, add_generation_prompt=True)
    full_turn = r.render_ids([*prev, asst])
    # Truncated: no </assistant> in the completion.
    completion = full_turn[len(pp) : -2]
    assert r._assistant_end not in completion
    bridged = r.bridge_to_next_turn(pp, completion, ext)
    assert bridged is not None
    # The synthesised close makes the tape identical to the clean-stop
    # bridge, which matches the fresh render byte-for-byte.
    fresh = r.render_ids([*prev, asst, *ext], add_generation_prompt=True)
    assert bridged.token_ids == fresh


def test_bridge_rejects_assistant_extension():
    r = _renderer()
    prev = [{"role": "user", "content": "Hi"}]
    pp = r.render_ids(prev, add_generation_prompt=True)
    assert (
        r.bridge_to_next_turn(
            pp, [r._assistant_end], [{"role": "assistant", "content": "x"}]
        )
        is None
    )
