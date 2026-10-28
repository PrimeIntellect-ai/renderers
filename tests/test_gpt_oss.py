"""GPT-OSS header coverage that the Harmony parity oracle can't provide.

The parity oracle is harmony itself, so it moves with whichever harmony build
is installed. The renderer writes tool calls the way gpt-oss samples them,
channel-first with a content type, and keeps the recipient-first header of
openai-harmony and the gpt-oss chat template for tool results. These tests pin
both, against fixed strings and against real sampled rollouts.
"""

from __future__ import annotations

import json
from pathlib import Path

from renderers import GptOssRendererConfig, create_renderer
from renderers.base import load_tokenizer

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def _call(call_id: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": arguments},
    }


def test_tool_call_headers_match_sampling_and_results_put_recipient_first():
    tok = load_tokenizer("openai/gpt-oss-20b")
    renderer = create_renderer(tok)
    messages = [
        {"role": "user", "content": "Weather in SF?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call("c1", '{"location":"SF"}'),
                _call("c2", "location = SF"),
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "20C"},
    ]

    ids = renderer.render_ids(messages, tools=_TOOLS)
    expected = tok.encode(
        "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"location":"SF"}<|call|>'
        "<|start|>assistant<|channel|>commentary to=functions.get_weather code"
        "<|message|>location = SF<|call|>"
        "<|start|>functions.get_weather to=assistant<|channel|>commentary"
        "<|message|>20C<|end|>",
        add_special_tokens=False,
    )

    assert ids[-len(expected) :] == expected


def test_rerendered_tool_calls_equal_sampled_tokens():
    """Real multi-turn Prime Agent rollouts of gpt-oss-20b, with each tool call's
    arguments as the model wrote them: re-rendering the conversation reproduces
    every sampled tool-call turn token for token, both ``json`` and ``code``."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "gpt_oss_rollouts.json").read_text()
    )
    tok = load_tokenizer("openai/gpt-oss-20b")
    renderer = create_renderer(tok, GptOssRendererConfig(**fixture["renderer"]))
    content_types = set()
    for rollout in fixture["rollouts"]:
        rendered = renderer.render(rollout["messages"], tools=rollout["tools"])
        for turn in rollout["sampled_turns"]:
            ids = [
                token
                for token, index in zip(rendered.token_ids, rendered.message_indices)
                if index == turn["message_index"]
            ]
            assert ids == turn["token_ids"], rollout["task"]
            text = tok.decode(ids)
            content_types |= {
                kind
                for kind, header in (
                    ("json", "<|constrain|>json<|message|>"),
                    ("code", " code<|message|>"),
                )
                if header in text
            }
    assert content_types == {"json", "code"}
