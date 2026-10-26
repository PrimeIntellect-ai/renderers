"""DeepSeek-V4.1-Flash parity against the checkpoint's shipped Python encoder."""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache

import pytest
from huggingface_hub import hf_hub_download
from pydantic import TypeAdapter, ValidationError
from renderers import (
    DeepSeekV41Renderer,
    DeepSeekV41RendererConfig,
    RendererConfig,
    ToolCallParseStatus,
    create_renderer,
)
from renderers.base import load_tokenizer
from renderers.configs import config_from_name

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

CONVERSATIONS = {
    "user": [{"role": "user", "content": "What is 2+2?"}],
    "system_user": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    "multi_turn": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
        {"role": "user", "content": "2+2?"},
    ],
    "system_multi_turn_ending_in_assistant": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add"},
    ],
}


@lru_cache(maxsize=1)
def _tokenizer():
    return load_tokenizer(MODEL)


@lru_cache(maxsize=1)
def _reference_encoder():
    path = hf_hub_download(MODEL, "encoding/encoding.py")
    spec = importlib.util.spec_from_file_location("deepseek_v41_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode(renderer, messages, **kwargs):
    return _tokenizer().decode(
        renderer.render_ids(messages, **kwargs), skip_special_tokens=False
    )


def test_registration_and_native_defaults():
    # Selected by name only: the shared parity suites need tool support for auto-mapping.
    renderer = create_renderer(_tokenizer(), config_from_name("deepseek-v4.1"))

    assert isinstance(renderer, DeepSeekV41Renderer)
    assert renderer.config.enable_thinking is True
    assert renderer.config.reasoning_effort == "high"


def test_config_discriminator_and_effort_validation():
    parsed = TypeAdapter(RendererConfig).validate_python(
        {"name": "deepseek-v4.1", "reasoning_effort": 30}
    )
    assert isinstance(parsed, DeepSeekV41RendererConfig)
    assert parsed.reasoning_effort == 30

    with pytest.raises(ValidationError):
        DeepSeekV41RendererConfig(reasoning_effort=101)


@pytest.mark.parametrize("name", CONVERSATIONS)
@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max", 30])
def test_matches_reference_encoder(name, enable_thinking, reasoning_effort):
    messages = CONVERSATIONS[name]
    add_generation_prompt = messages[-1]["role"] != "assistant"
    renderer = DeepSeekV41Renderer(
        _tokenizer(),
        DeepSeekV41RendererConfig(
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
        ),
    )

    expected = _reference_encoder().encode_messages(
        [dict(message) for message in messages],
        thinking_mode="thinking" if enable_thinking else "chat",
        reasoning_effort=reasoning_effort,
    )

    assert (
        _decode(renderer, messages, add_generation_prompt=add_generation_prompt)
        == expected
    )


def test_inline_reasoning_moves_into_think_block():
    renderer = DeepSeekV41Renderer(_tokenizer())
    inline = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "<think>add</think>4"},
    ]
    split = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add"},
    ]

    assert renderer.render_ids(inline) == renderer.render_ids(split)


def test_rejects_mid_conversation_system():
    renderer = DeepSeekV41Renderer(_tokenizer())
    user = [{"role": "user", "content": "Hi"}]

    with pytest.raises(NotImplementedError):
        renderer.render([*user, {"role": "system", "content": "Be terse."}])


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
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {"type": "object"},
                },
                "required": ["query"],
            },
        },
    },
]


def _call(id, name, arguments):
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


AGENT_TRAJECTORY = [
    {"role": "system", "content": "You are a travel agent."},
    {"role": "user", "content": "Weather in Paris and Rome?"},
    {
        "role": "assistant",
        "reasoning_content": "Call weather for both cities.",
        "content": "",
        "tool_calls": [
            _call("a", "weather", {"city": "Paris"}),
            _call("b", "weather", {"city": "Rome", "days": 2}),
        ],
    },
    {"role": "tool", "tool_call_id": "b", "content": "Rome: sunny"},
    {"role": "tool", "tool_call_id": "a", "content": "Paris: rain"},
    {
        "role": "assistant",
        "reasoning_content": "Look for indoor activities in Paris.",
        "content": "Let me search.",
        "tool_calls": [
            _call(
                "c",
                "search",
                {"query": "Paris museums", "filters": {"open": True, "max": 3}},
            )
        ],
    },
    {"role": "tool", "tool_call_id": "c", "content": "Louvre, Orsay"},
    {
        "role": "assistant",
        "reasoning_content": "Summarize.",
        "content": "Rome is sunny; Paris is rainy, try the Louvre.",
    },
    {"role": "user", "content": "And Berlin?"},
    {
        "role": "assistant",
        "reasoning_content": "One more call.",
        "content": "",
        "tool_calls": [_call("d", "weather", {"city": "Berlin"})],
    },
    {"role": "tool", "tool_call_id": "d", "content": "Berlin: snow"},
]

TOOL_CONVERSATIONS = {
    "tool_call_without_system": [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("x", "weather", {"city": "Oslo"})],
        },
        {"role": "tool", "tool_call_id": "x", "content": '{"temp": -3}'},
    ],
    "tool_result_then_user_text": [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "reasoning_content": "check",
            "tool_calls": [_call("x", "weather", {"city": "Oslo"})],
        },
        {"role": "tool", "tool_call_id": "x", "content": "cold"},
        {"role": "user", "content": "Also, be brief."},
    ],
    "argument_encodings": [
        {"role": "system", "content": "Call tools."},
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "reasoning_content": "Many calls.",
            "tool_calls": [
                _call("1", "weather", '{"city": "Oslo", "days": 3}'),
                _call("2", "weather", json.dumps('{"city": "Lima"}')),
                _call("3", "search", "[1, 2]"),
                _call("4", "search", "not json"),
                _call("5", "search", {"query": "", "filters": None}),
            ],
        },
    ],
    "agent_trajectory": AGENT_TRAJECTORY,
    "agent_trajectory_answered": AGENT_TRAJECTORY
    + [
        {
            "role": "assistant",
            "reasoning_content": "Report.",
            "content": "Berlin has snow.",
        }
    ],
}


def _reference_messages(messages, tools):
    messages = [dict(message) for message in messages]
    if tools:
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = tools
    return messages


@pytest.mark.parametrize("name", TOOL_CONVERSATIONS)
@pytest.mark.parametrize("with_tools", [True, False])
@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("drop_thinking", [True, False])
def test_tool_conversations_match_reference_encoder(
    name, with_tools, enable_thinking, drop_thinking
):
    messages = TOOL_CONVERSATIONS[name]
    tools = TOOLS if with_tools else None
    add_generation_prompt = messages[-1]["role"] != "assistant"
    renderer = DeepSeekV41Renderer(
        _tokenizer(),
        DeepSeekV41RendererConfig(
            enable_thinking=enable_thinking, drop_thinking=drop_thinking
        ),
    )

    expected = _reference_encoder().encode_messages(
        _reference_messages(messages, tools),
        thinking_mode="thinking" if enable_thinking else "chat",
        drop_thinking=drop_thinking,
    )

    assert (
        _decode(
            renderer,
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        == expected
    )


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_agent_trajectory_extends_by_prefix_and_bridge(enable_thinking):
    renderer = DeepSeekV41Renderer(
        _tokenizer(), DeepSeekV41RendererConfig(enable_thinking=enable_thinking)
    )
    messages = TOOL_CONVERSATIONS["agent_trajectory_answered"]
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]

    for index, next_index in zip(assistant_indices, assistant_indices[1:]):
        prompt = renderer.render_ids(
            messages[:index], tools=TOOLS, add_generation_prompt=True
        )
        completed = renderer.render_ids(messages[: index + 1], tools=TOOLS)
        assert completed[: len(prompt)] == prompt

        new_messages = messages[index + 1 : next_index]
        bridged = renderer.bridge_to_next_turn(
            prompt, completed[len(prompt) :], new_messages, tools=TOOLS
        )
        # Parallel results need the issuing call order; a new user query
        # drops tool-cycle reasoning. Both re-render instead.
        declines = sum(m["role"] == "tool" for m in new_messages) > 1 or (
            enable_thinking and any(m["role"] == "user" for m in new_messages)
        )
        if declines:
            assert bridged is None
        else:
            assert bridged is not None
            assert bridged.token_ids == renderer.render_ids(
                messages[:next_index], tools=TOOLS, add_generation_prompt=True
            )


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_tool_calls_roundtrip_through_parse_response(enable_thinking):
    renderer = DeepSeekV41Renderer(
        _tokenizer(), DeepSeekV41RendererConfig(enable_thinking=enable_thinking)
    )
    messages = AGENT_TRAJECTORY[:3]
    prompt = renderer.render_ids(messages[:2], tools=TOOLS, add_generation_prompt=True)
    completed = renderer.render_ids(messages, tools=TOOLS)
    completion = completed[len(prompt) :]

    parsed = renderer.parse_response(completion, prompt_ids=prompt)

    expected_reasoning = messages[2]["reasoning_content"] if enable_thinking else None
    assert parsed.reasoning_content == expected_reasoning
    assert parsed.content == ""
    assert [(call.name, call.arguments, call.status) for call in parsed.tool_calls] == [
        ("weather", {"city": "Paris"}, ToolCallParseStatus.OK),
        ("weather", {"city": "Rome", "days": 2}, ToolCallParseStatus.OK),
    ]
    for call in parsed.tool_calls:
        start, end = call.token_span
        assert (
            _tokenizer()
            .decode(completion[start:end], skip_special_tokens=False)
            .rstrip("\n")
            == call.raw
        )


def test_parses_vllm_tool_block_without_leading_newlines():
    text = (
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="search">\n'
        '<｜DSML｜ parameter name="query" string="true">a "b"</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="filters" string="false">{"k": [1]}</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls><｜end▁of▁sentence｜>"
    )
    renderer = DeepSeekV41Renderer(
        _tokenizer(), DeepSeekV41RendererConfig(enable_thinking=False)
    )

    parsed = renderer.parse_response(
        _tokenizer().encode(text, add_special_tokens=False)
    )

    assert parsed.content == ""
    assert [(call.name, call.arguments, call.status) for call in parsed.tool_calls] == [
        ("search", {"query": 'a "b"', "filters": {"k": [1]}}, ToolCallParseStatus.OK)
    ]
