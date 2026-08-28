"""DeepSeek V4 Flash 0731 reference-encoder and DSML coverage."""

from __future__ import annotations

from functools import lru_cache

import pytest
from pydantic import TypeAdapter, ValidationError

from renderers import (
    DeepSeekV4Renderer,
    DeepSeekV4RendererConfig,
    RendererConfig,
    ToolCallParseStatus,
    create_renderer,
)
from renderers.base import MODEL_RENDERER_MAP, load_tokenizer


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    }
]


@lru_cache(maxsize=1)
def _tokenizer():
    return load_tokenizer(MODEL)


def _renderer(**config_kwargs):
    return DeepSeekV4Renderer(
        _tokenizer(),
        DeepSeekV4RendererConfig(**config_kwargs),
    )


def _decode(renderer, messages, **kwargs):
    return _tokenizer().decode(
        renderer.render_ids(messages, **kwargs),
        skip_special_tokens=False,
    )


def test_registration_and_native_defaults():
    tokenizer = _tokenizer()
    renderer = create_renderer(tokenizer)

    assert tokenizer.chat_template is None
    assert MODEL_RENDERER_MAP[MODEL] == "deepseek-v4"
    assert isinstance(renderer, DeepSeekV4Renderer)
    assert renderer.config.enable_thinking is False
    assert renderer.config.drop_thinking is True
    assert renderer.config.reasoning_effort == "low"
    assert renderer.effective_thinking_retention == "tool_cycle"


def test_config_discriminator_and_template_kwarg_contract():
    parsed = TypeAdapter(RendererConfig).validate_python(
        {
            "name": "deepseek-v4",
            "enable_thinking": True,
            "drop_thinking": False,
            "reasoning_effort": "max",
        }
    )
    assert isinstance(parsed, DeepSeekV4RendererConfig)
    assert parsed.reasoning_effort == "max"

    with pytest.raises(ValidationError):
        DeepSeekV4RendererConfig(
            drop_thinking=False,
            thinking_retention="tool_cycle",
        )


def test_chat_mode_generation_prompt_matches_reference_encoder():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]

    assert _decode(_renderer(), messages, add_generation_prompt=True) == (
        f"{BOS}Be concise.{USER}Hello{ASSISTANT}</think>"
    )
    assert _decode(_renderer(), messages) == f"{BOS}Be concise.{USER}Hello"


def test_thinking_mode_drops_only_historical_reasoning_without_tools():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First"},
        {
            "role": "assistant",
            "reasoning_content": "old secret",
            "content": "First answer",
        },
        {"role": "user", "content": "Second"},
        {
            "role": "assistant",
            "reasoning_content": "current thought",
            "content": "Second answer",
        },
    ]

    assert _decode(_renderer(enable_thinking=True), messages) == (
        f"{BOS}Be concise."
        f"{USER}First{ASSISTANT}</think>First answer{EOS}"
        f"{USER}Second{ASSISTANT}<think>current thought</think>Second answer{EOS}"
    )


def test_tools_preserve_reasoning_and_use_dsml_wire_format():
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "reasoning_content": "I should call the tool.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Berlin", "days": 2},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"sun":true}'},
    ]

    text = _decode(
        _renderer(enable_thinking=True),
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )

    assert text.startswith(f"{BOS}Be helpful.\n\n## Tools\n")
    assert f"{USER}Weather?{ASSISTANT}<think>I should call the tool.</think>" in text
    assert '<｜DSML｜invoke name="weather">' in text
    assert (
        '<｜DSML｜parameter name="city" string="true">Berlin</｜DSML｜parameter>'
    ) in text
    assert (
        '<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>'
    ) in text
    assert (
        f'{EOS}{USER}<tool_result>{{"sun":true}}</tool_result>{ASSISTANT}<think>'
    ) in text


def test_parallel_tool_results_are_sorted_by_call_order():
    messages = [
        {"role": "user", "content": "Run both"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "first", "arguments": {}}},
                {"id": "b", "function": {"name": "second", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "b", "content": "second result"},
        {"role": "tool", "tool_call_id": "a", "content": "first result"},
    ]

    text = _decode(_renderer(), messages)
    assert text.index("first result") < text.index("second result")


def test_dsml_roundtrip_preserves_string_and_json_argument_types():
    renderer = _renderer(enable_thinking=True)
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "reasoning_content": "Use weather.",
            "content": "checking",
            "tool_calls": [
                {
                    "function": {
                        "name": "weather",
                        "arguments": {
                            "city": "true",
                            "days": 2,
                            "flags": [True, False],
                        },
                    }
                }
            ],
        },
    ]
    rendered = renderer.render_ids(messages)
    assistant_id = _tokenizer().encode(ASSISTANT, add_special_tokens=False)[0]
    completion_start = rendered.index(assistant_id) + 2  # skip Assistant + <think>

    parsed = renderer.parse_response(rendered[completion_start:])

    assert parsed.reasoning_content == "Use weather."
    assert parsed.content == "checking"
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.status == ToolCallParseStatus.OK
    assert call.name == "weather"
    assert call.arguments == {
        "city": "true",
        "days": 2,
        "flags": [True, False],
    }
    assert call.token_span is not None


def test_reasoning_effort_prefix_is_after_bos_and_thinking_only():
    messages = [{"role": "user", "content": "Think"}]
    high = _decode(
        _renderer(enable_thinking=True, reasoning_effort="high"),
        messages,
        add_generation_prompt=True,
    )
    chat = _decode(
        _renderer(enable_thinking=False, reasoning_effort="high"),
        messages,
        add_generation_prompt=True,
    )

    assert high.startswith(f"{BOS}Reasoning Effort: Absolute maximum")
    assert chat == f"{BOS}{USER}Think{ASSISTANT}</think>"


def test_reference_encoder_drops_stale_developer_messages_without_tools():
    messages = [
        {"role": "developer", "content": "stale internal query"},
        {"role": "assistant", "reasoning_content": "old", "content": "old answer"},
        {"role": "user", "content": "current public query"},
    ]

    text = _decode(
        _renderer(enable_thinking=True),
        messages,
        add_generation_prompt=True,
    )

    assert "stale internal query" not in text
    assert text == (f"{BOS}old answer{EOS}{USER}current public query{ASSISTANT}<think>")


def test_quick_task_token_renders_without_normal_generation_prompt():
    messages = [{"role": "user", "content": "Search?", "task": "action"}]

    assert _decode(_renderer(), messages) == (
        f"{BOS}{USER}Search?{ASSISTANT}</think><｜action｜>"
    )


def test_rendered_masks_keep_dsml_sampled_and_tool_wrappers_scaffolded():
    renderer = _renderer(enable_thinking=True)
    messages = [
        {"role": "user", "content": "Call it"},
        {
            "role": "assistant",
            "reasoning_content": "calling",
            "tool_calls": [
                {"id": "x", "function": {"name": "weather", "arguments": {}}}
            ],
        },
        {"role": "tool", "tool_call_id": "x", "content": "sunny"},
    ]
    rendered = renderer.render(messages, tools=TOOLS)

    assert len(rendered.token_ids) == len(rendered.message_indices)
    assert len(rendered.token_ids) == len(rendered.sampled_mask)
    assert len(rendered.token_ids) == len(rendered.is_content)
    assert rendered.tokens_by_role(sampled_only=True)["assistant"] > 0
    assert rendered.tokens_by_role(sampled_only=True)["tool"] == 0
    tool_content = rendered.content_mask_for_roles({"tool"})
    assert (
        _tokenizer().decode(
            [token for token, keep in zip(rendered.token_ids, tool_content) if keep],
            skip_special_tokens=False,
        )
        == "sunny"
    )


def test_bridge_extends_a_single_tool_result_exactly():
    renderer = _renderer(enable_thinking=True)
    first_messages = [
        {"role": "user", "content": "Call it"},
        {
            "role": "assistant",
            "reasoning_content": "calling",
            "tool_calls": [
                {
                    "id": "x",
                    "function": {"name": "weather", "arguments": {"city": "Rome"}},
                }
            ],
        },
    ]
    full_messages = first_messages + [
        {"role": "tool", "tool_call_id": "x", "content": "sunny"}
    ]
    prompt = renderer.render_ids(
        first_messages[:1],
        tools=TOOLS,
        add_generation_prompt=True,
    )
    full_first = renderer.render_ids(first_messages, tools=TOOLS)
    completion = full_first[len(prompt) :]

    bridged = renderer.bridge_to_next_turn(
        prompt,
        completion,
        full_messages[-1:],
        tools=TOOLS,
    )

    assert bridged is not None
    assert bridged.token_ids == renderer.render_ids(
        full_messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )
