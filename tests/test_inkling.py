"""Focused tests for Inkling and Inkling-Small.

The shared matrices (conftest render-parity, config-parity, multimodal) already
assert byte parity on the common shapes and image/audio parity against the
processor. This file pins the behaviours specific to Inkling's token-delimited
template that those matrices don't generate:

- the ``Thinking effort level: {N}`` line (labels + raw floats, ``0`` special case),
- effort emitted once before the first non-system message (or at the end),
- the token-delimited assistant structure and its sampled/is_content masks,
- tool-name resolution via ``tool_call_id``,
- parse round-trips (reasoning / content / native-JSON tool calls),
- bridge extensions matching a fresh render byte-for-byte,
- config validation of ``reasoning_effort``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from renderers import create_renderer
from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.configs import InklingRendererConfig
from renderers.inkling import InklingRenderer

_MODEL = "thinkingmachines/Inkling"
_SMALL_MODEL = "thinkingmachines/Inkling-Small"

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


def _renderer(**config_kwargs) -> InklingRenderer:
    renderer = create_renderer(_tok(), InklingRendererConfig(**config_kwargs))
    assert isinstance(renderer, InklingRenderer)
    return renderer


def _expected(msgs, *, tools=None, add_generation_prompt=False, **template_kwargs):
    result = _tok().apply_chat_template(
        msgs,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
        **template_kwargs,
    )
    if isinstance(result, dict):
        return list(result["input_ids"])
    if isinstance(result, str):
        return list(_tok().encode(result, add_special_tokens=False))
    return list(result)


def _decode(ids):
    return _tok().decode(ids, skip_special_tokens=False)


@pytest.mark.parametrize("checkpoint", [_MODEL, _SMALL_MODEL])
def test_auto_resolves_both_checkpoints_to_shared_renderer(checkpoint):
    renderer = create_renderer(load_tokenizer(checkpoint))
    assert isinstance(renderer, InklingRenderer)
    assert renderer.config.name == "inkling"


# ── Reasoning-effort line ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "effort,shown",
    [
        ("none", "0"),
        ("minimal", "0.1"),
        ("low", "0.2"),
        ("medium", "0.7"),
        ("high", "0.9"),
        ("max", "0.99"),
        (0.0, "0"),
        (0.5, "0.5"),
        (0.33, "0.33"),
        (0.99, "0.99"),
    ],
)
def test_effort_line_format_and_parity(effort, shown):
    msgs = [{"role": "user", "content": "Hi"}]
    r = _renderer(reasoning_effort=effort)
    ours = r.render_ids(msgs, add_generation_prompt=True)
    assert ours == _expected(msgs, add_generation_prompt=True, reasoning_effort=effort)
    assert f"Thinking effort level: {shown}<|end_message|>" in _decode(ours)


def test_default_effort_is_high():
    """No ``reasoning_effort`` set mirrors the template default (0.9)."""
    msgs = [{"role": "user", "content": "Hi"}]
    ours = _renderer().render_ids(msgs, add_generation_prompt=True)
    assert ours == _expected(msgs, add_generation_prompt=True)
    assert "Thinking effort level: 0.9<|end_message|>" in _decode(ours)


def test_effort_emitted_once_before_first_non_system():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Hi"},
    ]
    ours = _renderer().render_ids(msgs, add_generation_prompt=True)
    assert ours == _expected(msgs, add_generation_prompt=True)
    text = _decode(ours)
    # System body precedes the effort line, which precedes the user turn.
    assert (
        text.index("SYS")
        < text.index("Thinking effort level")
        < text.index("<|message_user|>")
    )
    assert text.count("Thinking effort level") == 1


def test_effort_emitted_at_end_when_all_system():
    msgs = [{"role": "system", "content": "only sys"}]
    ours = _renderer().render_ids(msgs)
    assert ours == _expected(msgs)
    text = _decode(ours)
    assert text.index("only sys") < text.index("Thinking effort level")


# ── Tool declaration precedes everything ──────────────────────────────


def test_tool_declaration_comes_first():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Weather?"},
    ]
    ours = _renderer().render_ids(msgs, tools=TOOLS, add_generation_prompt=True)
    assert ours == _expected(msgs, tools=TOOLS, add_generation_prompt=True)
    text = _decode(ours)
    assert text.startswith("<|message_system|>tool_declare<|content_xml|>")
    # The tools block precedes the system body and the effort line.
    assert (
        text.index("tool_declare")
        < text.index("SYS")
        < text.index("Thinking effort level")
    )


# ── Assistant structure + masks ───────────────────────────────────────


def test_assistant_structure_and_masks():
    msgs = [
        {"role": "user", "content": "2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "add them",
            "content": "4",
        },
    ]
    r = _renderer()
    rendered = r.render(msgs)
    assert rendered.token_ids == _expected(msgs)

    pos = [k for k, i in enumerate(rendered.message_indices) if i == 1]
    sampled = [rendered.sampled_mask[k] for k in pos]
    # Only the leading <|message_model|> (gen-prompt-equivalent) is scaffold.
    assert sampled[0] is False
    assert all(sampled[1:])
    # On assistant tokens is_content == sampled_mask by construction.
    assert [rendered.is_content[k] for k in pos] == sampled
    # The turn ends on the sampled <|content_model_end_sampling|>.
    assert rendered.token_ids[pos[-1]] == r._content_model_end_sampling
    assert rendered.sampled_mask[pos[-1]] is True


def test_tool_call_invoke_json_shape():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris", "days": 3},
                    }
                }
            ],
        },
    ]
    ours = _renderer().render_ids(msgs, tools=TOOLS)
    assert ours == _expected(msgs, tools=TOOLS)
    assert (
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        '{"name":"get_weather","args":{"city":"Paris","days":3}}<|end_message|>'
    ) in _decode(ours)


@pytest.mark.parametrize("arguments", ["not-json", "[]", ["not", "an", "object"]])
def test_tool_call_rejects_non_object_arguments(arguments):
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": arguments}}
            ],
        },
    ]
    with pytest.raises(TypeError, match="arguments must be a JSON object"):
        _renderer().render_ids(msgs, tools=TOOLS)


def test_unknown_content_part_type_raises():
    with pytest.raises(ValueError, match="Unsupported Inkling content part type"):
        _renderer().render_ids(
            [
                {
                    "role": "user",
                    "content": [{"type": "file", "file": "notes.txt"}],
                }
            ]
        )


def test_input_image_field_loads_during_render_and_bridge(monkeypatch):
    from PIL import Image

    renderer = _renderer()
    image = Image.new("RGB", (2, 2), color="navy")
    processed_images = []

    def process_image(pil, _image_hash):
        processed_images.append(pil)
        return {"pixel_values": np.zeros((1, 1), dtype=np.float32)}, 1

    monkeypatch.setattr(renderer, "_process_image", process_image)
    part = {"type": "input_image", "input_image": image}

    rendered = renderer.render([{"role": "user", "content": [part]}])
    assert rendered.multi_modal_data is not None
    assert len(rendered.multi_modal_data.mm_items["image"]) == 1

    bridged, fresh = _bridge_case(
        renderer,
        [{"role": "user", "content": "Describe this later."}],
        {"role": "assistant", "content": "Okay."},
        [{"role": "user", "content": [part]}],
    )
    assert bridged is not None
    assert bridged.token_ids == fresh
    assert bridged.multi_modal_data is not None
    assert len(bridged.multi_modal_data.mm_items["image"]) == 1
    assert processed_images and all(pil is image for pil in processed_images)


# ── Tool-name resolution ──────────────────────────────────────────────


def test_tool_name_resolved_via_tool_call_id():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]
    ours = _renderer().render_ids(msgs, tools=TOOLS)
    assert ours == _expected(msgs, tools=TOOLS)
    assert "<|message_tool|>get_weather<|content_text|>sunny<|end_message|>" in _decode(
        ours
    )


# ── Parse ─────────────────────────────────────────────────────────────


def _completion(prompt, asst, *, tools=None):
    r = _renderer()
    pp = r.render_ids(prompt, tools=tools, add_generation_prompt=True)
    full = r.render_ids([*prompt, asst], tools=tools)
    return r, full[len(pp) :]


def test_parse_reasoning_and_content():
    r, comp = _completion(
        [{"role": "user", "content": "2+2?"}],
        {
            "role": "assistant",
            "reasoning_content": "Let me think.",
            "content": "The answer is 4.",
        },
    )
    parsed = r.parse_response(comp)
    assert parsed.reasoning_content == "Let me think."
    assert parsed.content == "The answer is 4."
    assert not parsed.tool_calls


def test_parse_tool_call_preserves_types():
    r, comp = _completion(
        [{"role": "user", "content": "weather?"}],
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris", "days": 3},
                    }
                }
            ],
        },
        tools=TOOLS,
    )
    parsed = r.parse_response(comp, tools=TOOLS)
    assert parsed.content == "checking"
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris", "days": 3}  # int preserved
    assert tc.status == ToolCallParseStatus.OK
    assert tc.token_span is not None


def test_parse_multiple_tool_calls():
    r, comp = _completion(
        [{"role": "user", "content": "hi"}],
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "f", "arguments": {"a": 1}}},
                {"function": {"name": "g", "arguments": {}}},
            ],
        },
    )
    parsed = r.parse_response(comp)
    assert [tc.name for tc in parsed.tool_calls] == ["f", "g"]
    assert parsed.tool_calls[0].arguments == {"a": 1}
    assert parsed.tool_calls[1].arguments == {}


# ── Bridge ────────────────────────────────────────────────────────────


def _bridge_case(r, prev, asst, ext, *, tools=None):
    pp = r.render_ids(prev, tools=tools, add_generation_prompt=True)
    full = r.render_ids([*prev, asst], tools=tools)
    completion = full[len(pp) :]
    assert completion[-1] == r._content_model_end_sampling
    bridged = r.bridge_to_next_turn(pp, completion, ext, tools=tools)
    fresh = r.render_ids([*prev, asst, *ext], tools=tools, add_generation_prompt=True)
    return bridged, fresh


def test_bridge_user_extension_matches_fresh_render():
    r = _renderer()
    bridged, fresh = _bridge_case(
        r,
        [{"role": "user", "content": "Hi"}],
        {"role": "assistant", "content": "Hello!"},
        [{"role": "user", "content": "Bye"}],
    )
    assert bridged is not None
    assert bridged.token_ids == fresh


def test_bridge_tool_extension_matches_fresh_render():
    r = _renderer()
    bridged, fresh = _bridge_case(
        r,
        [{"role": "user", "content": "Weather in Tokyo?"}],
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
            ],
        },
        # Tool message carries an explicit name (bridge can't resolve a
        # prior-turn tool_call_id since the assistant is outside new_messages).
        [
            {"role": "tool", "name": "get_weather", "content": "Sunny"},
            {"role": "user", "content": "And Paris?"},
        ],
        tools=TOOLS,
    )
    assert bridged is not None
    assert bridged.token_ids == fresh


def test_bridge_synthesizes_close_on_truncation():
    r = _renderer()
    prev = [{"role": "user", "content": "Hi"}]
    asst = {"role": "assistant", "content": "Hello!"}
    ext = [{"role": "user", "content": "Bye"}]
    pp = r.render_ids(prev, add_generation_prompt=True)
    full = r.render_ids([*prev, asst])
    # Truncated: drop the terminating <|content_model_end_sampling|>.
    completion = full[len(pp) : -1]
    assert r._content_model_end_sampling not in completion
    bridged = r.bridge_to_next_turn(pp, completion, ext)
    assert bridged is not None
    fresh = r.render_ids([*prev, asst, *ext], add_generation_prompt=True)
    assert bridged.token_ids == fresh


def test_bridge_rejects_assistant_extension():
    r = _renderer()
    pp = r.render_ids([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert (
        r.bridge_to_next_turn(
            pp, [r._content_model_end_sampling], [{"role": "assistant", "content": "x"}]
        )
        is None
    )


def test_bridge_refuses_tool_needing_prior_name_resolution():
    """A tool message with only ``tool_call_id`` (no explicit name) can't be
    resolved from ``new_messages`` alone, so the bridge refuses (the caller
    re-renders)."""
    r = _renderer()
    pp = r.render_ids([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    bridged = r.bridge_to_next_turn(
        pp,
        [r._content_model_end_sampling],
        [{"role": "tool", "tool_call_id": "c1", "content": "sunny"}],
    )
    assert bridged is None


# ── Config validation ─────────────────────────────────────────────────


def test_config_accepts_labels_and_floats():
    for eff in ("none", "minimal", "low", "medium", "high", "max", 0.0, 0.5, 0.99):
        InklingRendererConfig(reasoning_effort=eff)


def test_audio_cache_has_an_independent_bound_and_uses_public_processor_api():
    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "audio_input_ids": np.zeros((1, 1, 80), dtype=np.int32),
                "audio_input_ids_mask": np.ones((1, 1), dtype=bool),
            }

    processor = FakeProcessor()
    renderer = InklingRenderer(
        _tok(),
        InklingRendererConfig(image_cache_max=1, audio_cache_max=2),
        processor=processor,
    )
    renderer._process_audio(np.zeros(8, dtype=np.float32), 16_000)
    renderer._process_audio(np.ones(8, dtype=np.float32), 16_000)

    assert len(renderer._audio_cache) == 2
    assert all(call["return_tensors"] == "pt" for call in processor.calls)
    assert all(call["sampling_rate"] == 16_000 for call in processor.calls)


def test_processor_unavailable_fails_lazily_with_upgrade_message(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("unknown processor")

    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", unavailable)
    renderer = _renderer()

    with pytest.raises(RuntimeError, match="Transformers >=5.14"):
        renderer._get_processor()


@pytest.mark.parametrize("bad", ["no_think", "bogus", 1.5, -0.1])
def test_config_rejects_bad_effort(bad):
    with pytest.raises(Exception):
        InklingRendererConfig(reasoning_effort=bad)


# ── Stop tokens ───────────────────────────────────────────────────────


def test_stop_tokens():
    r = _renderer()
    stop = r.get_stop_token_ids()
    assert r._content_model_end_sampling in stop
    # eos is <|content_model_end_sampling|> (config.json eos_token_id = 200006).
    assert stop[0] == r._content_model_end_sampling
