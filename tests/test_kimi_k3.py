"""Kimi-K3 renderer parity and contract tests.

K3 has no Jinja chat template — ``apply_chat_template`` runs Python shipped in the
model repo — so the reference for text renders is that call, exercised through the
pinned revision in ``TRUSTED_REVISIONS``.

Image renders deliberately diverge from the reference: the template emits a bare
``<|kimi_image_placeholder|>`` that its encoder substitutes later, whereas the
renderer emits the serving-ready media block, matching ``KimiK25Renderer``.
"""

from __future__ import annotations

import pytest

from renderers import config_from_name, create_renderer
from renderers.kimi_k3 import _RESPONSE_CHANNEL, _THINK_CHANNEL, _close_tag, _open_tag

MODEL = "moonshotai/Kimi-K3"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from renderers.base import TRUSTED_REVISIONS

    return AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, revision=TRUSTED_REVISIONS[MODEL]
    )


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return create_renderer(tokenizer, config_from_name("kimi-k3"))


def _image_part(size: tuple[int, int] = (112, 112)) -> dict:
    from PIL import Image

    return {"type": "image", "image": Image.new("RGB", size, (10, 20, 30))}


@pytest.mark.parametrize(
    "messages,tools",
    [
        ([{"role": "user", "content": "hi"}], None),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
            None,
        ),
        ([{"role": "user", "content": "weather?"}], TOOLS),
        (
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "because"},
                {"role": "user", "content": "q2"},
            ],
            None,
        ),
        (
            [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Sydney", "n": 3, "ok": True},
                            },
                        }
                    ],
                },
            ],
            TOOLS,
        ),
    ],
    ids=[
        "user-only",
        "multi-turn",
        "with-tools",
        "thinking-history",
        "tool-call-history",
    ],
)
def test_text_renders_match_the_model_encoder(tokenizer, renderer, messages, tools):
    kwargs = {"tools": tools} if tools else {}
    reference = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **kwargs
    )
    assert renderer.render_ids(
        messages, tools=tools, add_generation_prompt=True
    ) == list(reference)


def test_generation_prompt_opens_the_think_channel(tokenizer, renderer):
    ids = renderer.render_ids(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    tail = tokenizer.decode(ids, skip_special_tokens=False)
    assert tail.endswith('<|open|>message role="assistant"<|sep|><|open|>think<|sep|>')


def test_one_media_pad_per_image(tokenizer, renderer):
    """The model expands the pad server-side, so a run of pads would double-count."""
    pad = tokenizer.convert_tokens_to_ids("<|media_pad|>")
    one = renderer.render_ids(
        [{"role": "user", "content": [_image_part()]}], add_generation_prompt=True
    )
    two = renderer.render_ids(
        [{"role": "user", "content": [_image_part(), _image_part()]}],
        add_generation_prompt=True,
    )
    assert one.count(pad) == 1
    assert two.count(pad) == 2
    assert renderer.mm_token_type_id_map() == {pad: 1}


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    from renderers.base import TRUSTED_REVISIONS

    return AutoProcessor.from_pretrained(
        MODEL, trust_remote_code=True, revision=TRUSTED_REVISIONS[MODEL]
    )


@pytest.mark.parametrize(
    "size",
    [(112, 112), (224, 224), (100, 50), (1000, 300), (37, 91)],
    ids=["square-small", "square-large", "wide", "very-wide", "tall-odd"],
)
def test_image_renders_byte_identically_to_the_processor(tokenizer, processor, size):
    """K3 embeds the *source* pixel size in the block — not the patch grid.

    Asserting against the processor rather than against values this renderer computed
    is what catches a wrong dimension source; the grid and the pixel size diverge
    sharply for non-square inputs (1000x300 -> grid 22x72).
    """
    renderer = create_renderer(tokenizer, config_from_name("kimi-k3"))
    renderer._processor = processor
    messages = [{"role": "user", "content": [_image_part(size)]}]

    reference = list(processor(messages=messages, return_tensors="np")["input_ids"][0])
    assert renderer.render_ids(messages, add_generation_prompt=True) == reference


def test_image_block_carries_pixels_not_placeholder(tokenizer, processor):
    renderer = create_renderer(tokenizer, config_from_name("kimi-k3"))
    renderer._processor = processor
    text = tokenizer.decode(
        renderer.render_ids(
            [{"role": "user", "content": [_image_part((100, 50))]}],
            add_generation_prompt=True,
        ),
        skip_special_tokens=False,
    )
    assert (
        "<|media_begin|>image 100x50<|media_content|><|media_pad|><|media_end|>" in text
    )
    assert "<|kimi_image_placeholder|>" not in text


def test_render_carries_the_image_payload(renderer):
    rendered = renderer.render(
        [{"role": "user", "content": [_image_part()]}], add_generation_prompt=True
    )
    mm = rendered.multi_modal_data
    assert list(mm.mm_items) == ["image"]
    assert set(mm.mm_items["image"][0]) == {"pixel_values", "grid_thws"}
    assert len(mm.mm_hashes["image"]) == 1

    placeholder = mm.mm_placeholders["image"][0]
    assert placeholder.length == 1
    pad = renderer._media_pad
    assert rendered.token_ids[placeholder.offset] == pad
    # Only the pad counts as body; the surrounding wrap is scaffolding.
    assert rendered.is_content[placeholder.offset] is True
    assert sum(rendered.is_content) == 1


def test_bridge_shifts_image_placeholder_offsets(renderer):
    previous_prompt = renderer.render_ids(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    bridged = renderer.bridge_to_next_turn(
        previous_prompt, [], [{"role": "user", "content": [_image_part()]}]
    )
    assert bridged is not None
    placeholder = bridged.multi_modal_data.mm_placeholders["image"][0]
    assert placeholder.offset >= len(previous_prompt)
    assert bridged.token_ids[placeholder.offset] == renderer._media_pad


def test_bridge_preserves_the_sampled_prefix(renderer):
    previous_prompt = renderer.render_ids(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    previous_completion = renderer.render_ids([{"role": "assistant", "content": "x"}])[
        :6
    ]

    bridged = renderer.bridge_to_next_turn(
        previous_prompt, previous_completion, [{"role": "user", "content": "next"}]
    )

    assert bridged is not None
    base = [*previous_prompt, *previous_completion]
    assert bridged.token_ids[: len(base)] == base
    assert len(bridged.token_ids) > len(base)


def test_bridge_refuses_an_assistant_extension(renderer):
    previous_prompt = renderer.render_ids(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert (
        renderer.bridge_to_next_turn(
            previous_prompt, [], [{"role": "assistant", "content": "no"}]
        )
        is None
    )


def test_round_trips_a_tool_call(tokenizer, renderer):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Sydney"}},
            }
        ],
    }
    ids = renderer.render_ids(
        [{"role": "user", "content": "weather?"}, message], tools=TOOLS
    )
    parsed = renderer.parse_response(ids, tools=TOOLS)
    assert [call.name for call in parsed.tool_calls] == ["get_weather"]
    assert parsed.tool_calls[0].arguments == {"city": "Sydney"}


def test_rejects_an_unsupported_thinking_effort(tokenizer):
    from renderers.configs import KimiK3RendererConfig

    config = KimiK3RendererConfig(thinking_effort="max")
    object.__setattr__(
        config, "thinking_effort", "medium"
    )  # advertised but unsupported
    renderer = create_renderer(tokenizer, config)
    with pytest.raises(ValueError, match="thinking_effort"):
        renderer.render_ids(
            [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )


def test_stop_tokens_end_the_message(tokenizer, renderer):
    assert renderer.get_stop_token_ids() == [
        tokenizer.convert_tokens_to_ids("<|end_of_msg|>")
    ]


def test_emits_images_inside_tool_responses(tokenizer, processor):
    """Asserted here rather than in the shared parity suite: K3's own processor cannot
    extract images from tool-role messages, so that suite has no reference to compare."""
    renderer = create_renderer(tokenizer, config_from_name("kimi-k3"))
    renderer._processor = processor
    rendered = renderer.render(
        [
            {"role": "user", "content": "look"},
            {"role": "tool", "name": "camera", "content": [_image_part((100, 50))]},
        ],
        add_generation_prompt=True,
    )
    text = tokenizer.decode(rendered.token_ids, skip_special_tokens=False)
    assert '<|open|>message role="tool" name="camera"<|sep|>' in text
    assert (
        "<|media_begin|>image 100x50<|media_content|><|media_pad|><|media_end|>" in text
    )
    assert len(rendered.multi_modal_data.mm_placeholders["image"]) == 1


def test_parses_reasoning_from_a_sampled_completion(tokenizer, renderer):
    """A completion begins inside the think channel — the generation prompt already
    opened it — so reasoning has no opening tag to search for."""
    completion = (
        "let me think about it"
        + _close_tag(_THINK_CHANNEL)
        + _open_tag(_RESPONSE_CHANNEL)
        + "Blue"
        + _close_tag(_RESPONSE_CHANNEL)
    )
    parsed = renderer.parse_response(
        tokenizer.encode(completion, add_special_tokens=False)
    )
    assert parsed.content == "Blue"
    assert parsed.reasoning_content == "let me think about it"


def test_parses_a_full_assistant_message_with_both_tags(tokenizer, renderer):
    """A re-rendered history turn does carry the opening tag; both shapes must work."""
    text = (
        _open_tag(_THINK_CHANNEL)
        + "reasoned"
        + _close_tag(_THINK_CHANNEL)
        + _open_tag(_RESPONSE_CHANNEL)
        + "Red"
        + _close_tag(_RESPONSE_CHANNEL)
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))
    assert parsed.content == "Red"
    assert parsed.reasoning_content == "reasoned"


# Captured verbatim from the deployed model, so a rewrite of the emitted form fails here rather
# than in a rollout. Arguments arrive as separate typed blocks, never as a JSON object.
LIVE_TOOL_CALL_COMPLETION = (
    '<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>'
    '<|open|>tools<|sep|><|open|>call tool="move_gripper" index="1"<|sep|>'
    '<|open|>argument key="x" type="number"<|sep|>0<|close|>argument<|sep|>'
    '<|open|>argument key="z" type="number"<|sep|>0.15<|close|>argument<|sep|>'
    '<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>'
)


def test_parses_the_deployed_models_tool_call(tokenizer, renderer):
    parsed = renderer.parse_response(
        tokenizer.encode(LIVE_TOOL_CALL_COMPLETION, add_special_tokens=False)
    )
    assert [call.name for call in parsed.tool_calls] == ["move_gripper"]
    assert parsed.tool_calls[0].arguments == {"x": 0, "z": 0.15}


def test_round_trips_every_argument_type(renderer):
    """Types survive the render, so a re-rendered history matches what the model emitted."""
    arguments = {"s": "text", "n": 2.5, "i": 7, "b": False, "z": None, "o": {"k": 1}, "a": [1, 2]}
    ids = renderer.render_ids(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "act", "arguments": arguments}}
                ],
            },
        ]
    )
    assert renderer.parse_response(ids).tool_calls[0].arguments == arguments


def test_renders_several_calls_in_one_tools_channel(tokenizer, renderer):
    """Each call carries its 1-based index, as the model's own enumeration does."""
    calls = [
        {"type": "function", "function": {"name": "a", "arguments": {"x": 1}}},
        {"type": "function", "function": {"name": "b", "arguments": {"y": 2}}},
    ]
    text = tokenizer.decode(
        renderer.render_ids(
            [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": calls}]
        ),
        skip_special_tokens=False,
    )
    assert text.count('<|open|>tools<|sep|>') == 1
    assert '<|open|>call tool="a" index="1"<|sep|>' in text
    assert '<|open|>call tool="b" index="2"<|sep|>' in text


def test_tool_calls_use_the_packages_record_type(renderer, tokenizer):
    """The agent harness reads name and arguments off the record, so a bare dict is dropped
    silently rather than rejected."""
    from renderers.base import ParsedToolCall, ToolCallParseStatus

    parsed = renderer.parse_response(
        tokenizer.encode(LIVE_TOOL_CALL_COMPLETION, add_special_tokens=False)
    )
    call = parsed.tool_calls[0]
    assert isinstance(call, ParsedToolCall)
    assert call.status is ToolCallParseStatus.OK
    assert call.raw


def test_flags_a_call_naming_an_undeclared_tool(renderer, tokenizer):
    from renderers.base import ToolCallParseStatus

    text = (
        '<|open|>tools<|sep|><|open|>call tool="nope" index="1"<|sep|>'
        '<|open|>argument key="x" type="number"<|sep|>1<|close|>argument<|sep|>'
        '<|close|>call<|sep|><|close|>tools<|sep|>'
    )
    parsed = renderer.parse_response(
        tokenizer.encode(text, add_special_tokens=False), tools=TOOLS
    )
    assert parsed.tool_calls[0].status is ToolCallParseStatus.UNKNOWN_TOOL
