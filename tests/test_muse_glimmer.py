from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from renderers import AutoRendererConfig, MuseGlimmerRenderer, create_renderer
from renderers.configs import MuseGlimmerRendererConfig
from transformers import AutoTokenizer

MODEL = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
TODAY = "2026-08-30"


@pytest.fixture(scope="module")
def tokenizer():
    path = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(Path(path))
    tokenizer.name_or_path = MODEL
    return tokenizer


@pytest.fixture(scope="module")
def tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "weather.get",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            },
        }
    ]


def _renderer(tokenizer, **kwargs):
    return MuseGlimmerRenderer(
        tokenizer,
        MuseGlimmerRendererConfig(current_date=TODAY, **kwargs),
    )


CASES = [
    ([{"role": "user", "content": "hello"}], None, False, {}),
    (
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hello"},
        ],
        None,
        True,
        {},
    ),
    (
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B"},
                ],
            }
        ],
        None,
        True,
        {},
    ),
    (
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        None,
        False,
        {},
    ),
    (
        [
            {"role": "user", "content": "solve"},
            {"role": "assistant", "reasoning_content": "think", "content": "done"},
        ],
        None,
        False,
        {},
    ),
    ([{"role": "user", "content": "weather?"}], "tools", True, {}),
    (
        [
            {"role": "system", "content": "Reasoning effort: low."},
            {"role": "user", "content": "weather?"},
        ],
        "tools",
        False,
        {"reasoning_strength": "low"},
    ),
    (
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather.get",
                            "arguments": {"city": "Paris", "days": 2},
                        },
                    }
                ],
            },
        ],
        "tools",
        False,
        {},
    ),
    (
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather.get",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        ],
        "tools",
        True,
        {},
    ),
    ([{"role": "tool", "name": "weather.get", "content": "sunny"}], "tools", True, {}),
    (
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {
                            "name": "weather.get",
                            "arguments": {"city": "Paris"},
                        },
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {
                            "name": "weather.get",
                            "arguments": {"city": "Rome", "days": [1, 2]},
                        },
                    },
                ],
            },
        ],
        "tools",
        False,
        {},
    ),
    (
        [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
        None,
        True,
        {"knowledge_cutoff": "2025-01-01", "reasoning_strength": "medium"},
    ),
]


@pytest.mark.parametrize(
    "messages,tool_mode,add_generation_prompt,config_kwargs", CASES
)
def test_pinned_template_token_parity(
    tokenizer, tools, messages, tool_mode, add_generation_prompt, config_kwargs
):
    selected_tools = tools if tool_mode else None
    renderer = _renderer(tokenizer, **config_kwargs)
    expected = tokenizer.apply_chat_template(
        messages,
        tools=selected_tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        current_date=TODAY,
        reasoning_strength=config_kwargs.get("reasoning_strength", "high"),
        knowledge_cutoff=config_kwargs.get("knowledge_cutoff", "2026-01-04"),
    )["input_ids"]
    assert (
        renderer.render_ids(
            messages,
            tools=selected_tools,
            add_generation_prompt=add_generation_prompt,
        )
        == expected
    )


def test_auto_resolution_is_typed(tokenizer):
    renderer = create_renderer(tokenizer, AutoRendererConfig())
    assert isinstance(renderer, MuseGlimmerRenderer)
    assert isinstance(renderer.config, MuseGlimmerRendererConfig)


def test_masks_and_assistant_invariant(tokenizer, tools):
    messages = [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "reasoning_content": "check",
            "content": "",
            "tool_calls": [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "weather.get", "arguments": {"city": "Paris"}},
                }
            ],
        },
    ]
    rendered = _renderer(tokenizer).render(messages, tools=tools)
    assert (
        len(rendered.token_ids)
        == len(rendered.sampled_mask)
        == len(rendered.is_content)
    )
    assert any(rendered.sampled_mask)
    assert any(rendered.is_content)
    for index, sampled, is_content in zip(
        rendered.message_indices, rendered.sampled_mask, rendered.is_content
    ):
        if index >= 0 and messages[index]["role"] == "assistant":
            assert sampled == is_content


def test_stop_parse_and_bridge(tokenizer, tools):
    renderer = _renderer(tokenizer)
    prompt_messages = [{"role": "user", "content": "weather?"}]
    prompt = renderer.render_ids(
        prompt_messages, tools=tools, add_generation_prompt=True
    )
    completion_text = (
        " to=self<|message|>check<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    completion = tokenizer.encode(completion_text, add_special_tokens=False)
    parsed = renderer.parse_response(completion, tools=tools)
    assert parsed.reasoning_content == "check"
    assert parsed.tool_calls[0].status.value == "ok"
    assert parsed.tool_calls[0].name == "weather.get"
    assert parsed.tool_calls[0].arguments == {"city": "Paris"}
    assert renderer.get_stop_token_ids() == [tokenizer.convert_tokens_to_ids("<|eot|>")]

    new_messages = [{"role": "tool", "name": "weather.get", "content": "sunny"}]
    bridged = renderer.bridge_to_next_turn(
        prompt, completion, new_messages, tools=tools
    )
    assert bridged is not None
    full_messages = (
        prompt_messages
        + [
            {
                "role": "assistant",
                "reasoning_content": "check",
                "content": "",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {
                            "name": "weather.get",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            }
        ]
        + new_messages
    )
    assert bridged.token_ids == renderer.render_ids(
        full_messages, tools=tools, add_generation_prompt=True
    )


@pytest.mark.parametrize("kind", ["image", "video"])
def test_text_only_rejects_media(tokenizer, kind):
    with pytest.raises(ValueError, match="text-only"):
        _renderer(tokenizer).render([{"role": "user", "content": [{"type": kind}]}])
