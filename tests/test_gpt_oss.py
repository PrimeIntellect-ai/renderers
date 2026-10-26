"""GPT-OSS header coverage that the Harmony parity oracle can't provide.

The parity oracle is harmony itself, so it moves with whichever harmony build
is installed. openai-harmony renders tool-call and tool-result headers
recipient-first; oss-harmony renders them channel-first. The renderer keeps
the recipient-first header on either build, and this test pins it.
"""

from __future__ import annotations

from renderers import create_renderer
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


def test_tool_call_and_result_headers_put_recipient_first():
    tok = load_tokenizer("openai/gpt-oss-20b")
    renderer = create_renderer(tok)
    messages = [
        {"role": "user", "content": "Weather in SF?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "20C"},
    ]

    ids = renderer.render_ids(messages, tools=_TOOLS)
    expected = tok.encode(
        "<|start|>assistant to=functions.get_weather<|channel|>commentary"
        '<|message|>{"location": "SF"}<|call|>'
        "<|start|>functions.get_weather to=assistant<|channel|>commentary"
        "<|message|>20C<|end|>",
        add_special_tokens=False,
    )

    assert ids[-len(expected) :] == expected
