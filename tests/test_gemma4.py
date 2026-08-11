"""Focused coverage for Gemma 4's template variants and tool grammar."""

from functools import lru_cache

import pytest

from renderers import Gemma4Renderer, create_renderer
from renderers.base import MODEL_RENDERER_MAP, MULTIMODAL_MODELS, load_tokenizer
from renderers.configs import Gemma4RendererConfig


_MODELS = (
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
)


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
        {"role": "assistant", "content": "It is 24 C."},
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
