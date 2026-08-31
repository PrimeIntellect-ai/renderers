"""Focused coverage for Gemma 4's template variants and tool grammar."""

from functools import lru_cache

import pytest
from parity import models_for

from renderers import Gemma4Renderer, create_renderer
from renderers.base import MODEL_RENDERER_MAP, MULTIMODAL_MODELS, load_tokenizer
from renderers.configs import Gemma4RendererConfig


_MODELS = tuple(case.model for case in models_for("gemma-checkpoints"))


@lru_cache
def _gemma4():
    tokenizer = load_tokenizer("google/gemma-4-31B-it")
    return tokenizer, create_renderer(tokenizer)


def test_all_instruction_checkpoints_are_registered_as_image_renderers():
    for model in _MODELS:
        assert MODEL_RENDERER_MAP[model] == "gemma4"
        assert MULTIMODAL_MODELS[model] == {"image"}


def test_disabled_thinking_prefill_tracks_template_revision(monkeypatch):
    tokenizer, current_renderer = _gemma4()
    messages = [{"role": "user", "content": "Hello"}]

    current_text = tokenizer.decode(
        current_renderer.render_ids(messages, add_generation_prompt=True),
        skip_special_tokens=False,
    )
    assert current_text.endswith("<|channel>thought\n<channel|>")

    # E2B/E4B use the otherwise-identical earlier template revision, which
    # stops at the model role opener when thinking is disabled.
    monkeypatch.setattr(tokenizer, "name_or_path", "google/gemma-4-E4B-it")
    monkeypatch.setattr(tokenizer, "chat_template", "")
    earlier_renderer = Gemma4Renderer(tokenizer)
    earlier_text = tokenizer.decode(
        earlier_renderer.render_ids(messages, add_generation_prompt=True),
        skip_special_tokens=False,
    )
    assert earlier_text.endswith("<|turn>model\n")


def test_preserve_thinking_controls_derived_retention_and_rejects_conflicts():
    tokenizer, _ = _gemma4()
    preserved = Gemma4Renderer(
        tokenizer,
        Gemma4RendererConfig(enable_thinking=True, preserve_thinking=True),
    )
    assert preserved.effective_thinking_retention == "all"

    with pytest.raises(ValueError, match="preserve_thinking=True implies"):
        Gemma4RendererConfig(
            preserve_thinking=True,
            thinking_retention="tool_cycle",
        )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "content": "orphan"}],
        [
            {"role": "assistant", "content": "No call."},
            {"role": "tool", "content": "orphan"},
        ],
        [
            {
                "role": "assistant",
                "content": "",
                "tool_responses": [{"name": "legacy", "response": "done"}],
            },
            {"role": "tool", "content": "still orphaned"},
        ],
    ],
)
def test_unconsumed_tool_messages_raise(messages):
    tokenizer, renderer = _gemma4()
    with pytest.raises(ValueError, match="Unconsumed tool message"):
        renderer.render_ids(messages)


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_tool_cycle_matches_canonical_template(enable_thinking):
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(
        tokenizer, Gemma4RendererConfig(enable_thinking=enable_thinking)
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Weather in Berlin?"},
        {
            "role": "assistant",
            "reasoning_content": "I should call the weather tool."
            if not enable_thinking
            else None,
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Berlin"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"temperature": 24, "unit": "C"}',
        },
        {
            "role": "assistant",
            "reasoning_content": "I can now answer." if not enable_thinking else None,
            "content": "It is 24 C.",
        },
    ]

    expected = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        return_dict=False,
    )
    assert renderer.render_ids(messages, tools=tools) == list(expected)


def test_disabled_thinking_post_tool_completion_matches_sampled_stream():
    """A post-tool assistant message continues the existing model turn.

    The 26B/31B generation prompt prefills an empty thought channel before the
    initial completion, but the disabled-thinking post-tool prompt does not
    insert another one. A later full rerender must preserve that exact stream.
    """
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer, Gemma4RendererConfig(enable_thinking=False))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    user = {"role": "user", "content": "Weather in Berlin?"}
    tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": {"city": "Berlin"},
                },
            }
        ],
    }
    tool_response = {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"temperature": 24}',
    }
    final = {"role": "assistant", "content": "It is 24 C."}

    initial_prompt = renderer.render_ids(
        [user], tools=tools, add_generation_prompt=True
    )
    tool_call_prompt = renderer.render_ids(
        [user, tool_call], tools=tools, add_generation_prompt=True
    )
    assert tool_call_prompt[: len(initial_prompt)] == initial_prompt
    tool_call_completion = tool_call_prompt[len(initial_prompt) :]

    post_tool_prompt = renderer.bridge_to_next_turn(
        initial_prompt,
        tool_call_completion,
        [tool_response],
        tools=tools,
    )
    assert post_tool_prompt is not None

    final_completion = tokenizer.encode(final["content"], add_special_tokens=False) + [
        renderer.get_stop_token_ids()[0]
    ]
    reminder = {"role": "user", "content": "Please summarize."}
    extended_stream = renderer.bridge_to_next_turn(
        post_tool_prompt.token_ids,
        final_completion,
        [reminder],
        tools=tools,
    )
    assert extended_stream is not None
    rerendered = renderer.render_ids(
        [user, tool_call, tool_response, final, reminder],
        tools=tools,
        add_generation_prompt=True,
    )
    assert rerendered == extended_stream.token_ids


def test_parser_extracts_reasoning_and_multiple_typed_tool_calls():
    tokenizer, renderer = _gemma4()
    text = (
        "<|channel>thought\nI need two lookups.\n<channel|>"
        '<|tool_call>call:weather{city:<|"|>Berlin<|"|>,days:2}'
        "<tool_call|>"
        "<|tool_call>call:flags{enabled:true,values:[1,null]}<tool_call|>"
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert parsed.reasoning_content == "I need two lookups."
    assert parsed.content == ""
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("weather", {"city": "Berlin", "days": 2}),
        ("flags", {"enabled": True, "values": [1, None]}),
    ]


def test_parser_recovers_prompt_opened_post_tool_reasoning():
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer, Gemma4RendererConfig(enable_thinking=True))
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "weather", "arguments": {"city": "Berlin"}},
    }
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
    ]
    expected_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )
    prompt = renderer.render_ids(messages, add_generation_prompt=True)

    assert prompt == list(expected_prompt)
    assert tokenizer.decode(prompt, skip_special_tokens=False).endswith(
        "<|channel>thought\n"
    )

    completion = tokenizer.encode(
        "Need synthesize.\n<channel|>It is sunny.<turn|>",
        add_special_tokens=False,
    )

    parsed = renderer.parse_response(completion)

    assert parsed.reasoning_content == "Need synthesize."
    assert parsed.content == "It is sunny."
    assert parsed.tool_calls == []

    # Initial-turn content without a channel closer remains ordinary content.
    direct = renderer.parse_response(
        tokenizer.encode("Direct answer.<turn|>", add_special_tokens=False)
    )
    assert direct.reasoning_content is None
    assert direct.content == "Direct answer."


def _image_renderer():
    """A Gemma 4 renderer with a live ``Gemma4Processor``, or a skip."""
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    try:
        renderer._get_processor()
    except (RuntimeError, OSError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Gemma4Processor unavailable: {exc}")
    return tokenizer, renderer


def _tiny_image():
    from PIL import Image

    return Image.new("RGB", (224, 224), color=(128, 192, 255))


@pytest.mark.parametrize("size", [(224, 224), (448, 224), (224, 448)])
def test_real_processor_keeps_one_batched_row_per_image(size):
    """Guard the ``MultiModalFieldConfig.batched('image')`` contract with
    live Gemma4Processor output rather than synthetic tensor shapes."""
    from PIL import Image

    _, renderer = _image_renderer()
    image = Image.new("RGB", size, color=(128, 192, 255))
    rendered = renderer.render(
        [{"role": "user", "content": [{"type": "image", "image": image}]}]
    )
    item = rendered.multi_modal_data.mm_items["image"][0]
    placeholder = rendered.multi_modal_data.mm_placeholders["image"][0]

    assert item["pixel_values"].shape[0] == 1
    assert item["image_position_ids"].shape[0] == 1
    assert item["pixel_values"].shape[1] == item["image_position_ids"].shape[1]
    assert placeholder.length > 0


def test_schema_unified_image_parts_still_expand_image_tokens():
    """``Dataset.from_list`` unifies the Arrow schema across a content list,
    so an image part round-tripped through a dataset carries ``text: None``
    (and text parts carry ``image: None``). Dispatching on ``"text" in part``
    before classifying media would route the image into the text branch,
    emit nothing, and silently skip soft-token expansion."""
    tokenizer, renderer = _image_renderer()
    image = _tiny_image()
    plain = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]
    schema_unified = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "text": None},
                {"type": "text", "text": "What is this?", "image": None},
            ],
        }
    ]

    baseline = renderer.render(plain)
    roundtripped = renderer.render(schema_unified)

    image_id = tokenizer.convert_tokens_to_ids("<|image|>")
    assert baseline.token_ids.count(image_id) > 0
    assert roundtripped.token_ids == baseline.token_ids
    assert (
        roundtripped.multi_modal_data.mm_hashes == baseline.multi_modal_data.mm_hashes
    )
    assert (
        roundtripped.multi_modal_data.mm_placeholders
        == baseline.multi_modal_data.mm_placeholders
    )


def test_schema_unified_tool_response_image_parts_survive():
    """Same hazard on the tool-response path, which also has to keep
    accepting untyped text parts."""
    tokenizer, renderer = _image_renderer()
    image = _tiny_image()
    messages = [
        {"role": "user", "content": "Look it up."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [
                {"type": "image", "image": image, "text": None},
                {"type": "text", "text": "captured", "image": None},
            ],
        },
    ]
    rendered = renderer.render(messages)

    image_id = tokenizer.convert_tokens_to_ids("<|image|>")
    assert rendered.token_ids.count(image_id) > 0
    assert rendered.multi_modal_data.mm_hashes["image"]
    assert "captured" in tokenizer.decode(rendered.token_ids, skip_special_tokens=False)


def test_untyped_text_parts_render_in_tool_responses():
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    messages = [
        {"role": "user", "content": "Check."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "check", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [{"text": "all good"}],
        },
    ]
    text = tokenizer.decode(renderer.render_ids(messages), skip_special_tokens=False)
    assert "all good" in text


def test_system_content_lists_reject_media_parts():
    """The text-only guard must not be fooled by a schema-unified image
    part's ``text: None`` key, which would drop the image silently."""
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    messages = [
        {
            "role": "system",
            "content": [{"type": "image", "image": object(), "text": None}],
        },
        {"role": "user", "content": "Hi"},
    ]
    with pytest.raises(ValueError, match="text parts only"):
        renderer.render_ids(messages)


def test_legacy_assistant_tool_responses_preserve_mask_contract():
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    messages = [
        {"role": "user", "content": "Check."},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_responses": [{"name": "check", "response": {"ok": True}}],
        },
    ]
    rendered = renderer.render(messages)

    for index, message_index in enumerate(rendered.message_indices):
        if message_index == 1:
            assert rendered.is_content[index] == rendered.sampled_mask[index]
