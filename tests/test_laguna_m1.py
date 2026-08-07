"""Exact chat-template parity for the official Laguna M.1 checkpoint."""

from __future__ import annotations

from functools import lru_cache

import pytest
from pydantic import TypeAdapter

from renderers import (
    LagunaM1Renderer,
    LagunaM1RendererConfig,
    RendererConfig,
    create_renderer,
)
from renderers.base import load_tokenizer

_MODEL = "poolside/Laguna-M.1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["cmd"],
            },
        },
    }
]


@lru_cache(maxsize=None)
def _tok():
    return load_tokenizer(_MODEL)


def _renderer(**kwargs) -> LagunaM1Renderer:
    renderer = create_renderer(_tok(), LagunaM1RendererConfig(**kwargs))
    assert isinstance(renderer, LagunaM1Renderer)
    return renderer


def _expected(messages, *, tools=None, add_generation_prompt=False, **kwargs):
    return list(
        _tok().apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **kwargs,
        )
    )


def test_auto_selection_and_typed_config():
    renderer = create_renderer(_tok())
    assert isinstance(renderer, LagunaM1Renderer)
    assert isinstance(renderer.config, LagunaM1RendererConfig)

    parsed = TypeAdapter(RendererConfig).validate_python(
        {"name": "laguna-m.1", "enable_thinking": True}
    )
    assert isinstance(parsed, LagunaM1RendererConfig)
    assert parsed.enable_thinking is True


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_no_system_and_generation_prompt_parity(enable_thinking):
    messages = [{"role": "user", "content": "  preserve me  "}]
    renderer = _renderer(enable_thinking=enable_thinking)
    got = renderer.render_ids(messages, add_generation_prompt=True)
    assert got == _expected(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    text = _tok().decode(got)
    assert text.startswith("〈|EOS|〉<user>\n  preserve me  \n</user>\n<assistant>\n")
    assert "<system>" not in text
    assert text.endswith("<think>" if enable_thinking else "</think>")


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_reasoning_content_history_parity_in_both_modes(enable_thinking):
    messages = [
        {"role": "user", "content": "Compute."},
        {
            "role": "assistant",
            "reasoning_content": "\n  two steps  \n",
            "content": "\n  result  \n",
        },
    ]
    got = _renderer(enable_thinking=enable_thinking).render_ids(messages)
    assert got == _expected(messages, enable_thinking=enable_thinking)
    assert "<think>\ntwo steps\n</think>\nresult\n</assistant>\n" in _tok().decode(got)


def test_reasoning_field_precedence_and_inline_think_extraction():
    precedence = [
        {"role": "user", "content": "Compute."},
        {
            "role": "assistant",
            "reasoning": "preferred",
            "reasoning_content": "ignored",
            "content": "answer",
        },
    ]
    got = _renderer(enable_thinking=True).render_ids(precedence)
    assert got == _expected(precedence, enable_thinking=True)
    text = _tok().decode(got)
    assert "preferred" in text
    assert "ignored" not in text

    empty_reasoning_wins = [
        {"role": "user", "content": "Compute."},
        {
            "role": "assistant",
            "reasoning": "",
            "reasoning_content": "also ignored",
            "content": "answer",
        },
    ]
    got = _renderer(enable_thinking=True).render_ids(empty_reasoning_wins)
    assert got == _expected(empty_reasoning_wins, enable_thinking=True)
    assert "also ignored" not in _tok().decode(got)

    inline = [
        {"role": "user", "content": "Compute."},
        {
            "role": "assistant",
            "content": "<think>\ninline reason\n</think>\nvisible answer",
        },
    ]
    assert _renderer().render_ids(inline) == _expected(inline)


def test_tools_calls_responses_and_generation_prompt_parity():
    messages = [
        {"role": "system", "content": "Use tools carefully."},
        {"role": "user", "content": "Inspect the machine."},
        {
            "role": "assistant",
            "reasoning": "Need a command.",
            "content": "Running it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": {"cmd": "uname -a", "timeout": 5},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"stdout":"Linux"}',
        },
    ]
    renderer = _renderer(enable_thinking=True)
    got = renderer.render_ids(messages, tools=TOOLS, add_generation_prompt=True)
    assert got == _expected(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    text = _tok().decode(got)
    assert "<available_tools>\n" in text
    assert (
        "<tool_call>shell\n<arg_key>cmd</arg_key>\n"
        "<arg_value>uname -a</arg_value>\n<arg_key>timeout</arg_key>\n"
        "<arg_value>5</arg_value>\n</tool_call>\n"
    ) in text
    assert '<tool_response>\n{"stdout":"Linux"}\n</tool_response>\n' in text
    assert text.endswith("<assistant>\n<think>")


def test_reasoning_content_and_tool_call_round_trip():
    renderer = _renderer(enable_thinking=True)
    prompt = [{"role": "user", "content": "Inspect the machine."}]
    assistant = {
        "role": "assistant",
        "reasoning": "Need a command.",
        "content": "Running it.",
        "tool_calls": [
            {
                "function": {
                    "name": "shell",
                    "arguments": {"cmd": "uname -a", "timeout": 5},
                }
            }
        ],
    }
    prompt_ids = renderer.render_ids(prompt, add_generation_prompt=True)
    full_ids = renderer.render_ids([*prompt, assistant])
    completion_ids = full_ids[len(prompt_ids) :]
    parsed = renderer.parse_response(completion_ids, tools=TOOLS)

    assert parsed.reasoning_content == "Need a command."
    assert parsed.content == "Running it."
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "shell"
    assert parsed.tool_calls[0].arguments == {"cmd": "uname -a", "timeout": 5}


def test_raw_assistant_round_trip_parity():
    messages = [
        {"role": "user", "content": "Continue."},
        {
            "role": "assistant",
            "content": "<think>raw reason</think>raw body</assistant>",
        },
    ]
    renderer = _renderer(enable_thinking=True, render_assistant_messages_raw=True)
    assert renderer.render_ids(messages) == _expected(
        messages,
        enable_thinking=True,
        render_assistant_messages_raw=True,
    )
