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
    arguments taken from the parsed sample's ``arguments_text``: re-rendering the
    conversation reproduces every sampled tool-call turn token for token, both
    ``json`` and ``code``."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "gpt_oss_rollouts.json").read_text()
    )
    tok = load_tokenizer("openai/gpt-oss-20b")
    renderer = create_renderer(tok, GptOssRendererConfig(**fixture["renderer"]))
    content_types = set()
    for rollout in fixture["rollouts"]:
        messages = rollout["messages"]
        for turn in rollout["sampled_turns"]:
            parsed = renderer.parse_response(turn["token_ids"]).tool_calls
            calls = messages[turn["message_index"]]["tool_calls"]
            assert len(parsed) == len(calls)
            for call, tc in zip(calls, parsed):
                call["function"]["arguments"] = tc.arguments_text
        rendered = renderer.render(messages, tools=rollout["tools"])
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


def test_a_code_header_on_json_arguments_round_trips():
    """gpt-oss sometimes samples `` code`` for a call whose arguments are a JSON
    object. The parsed call carries the sampled ``content_type``, and a re-render
    given it reproduces the sampled tokens; without it the header follows the
    arguments."""
    tok = load_tokenizer("openai/gpt-oss-20b")
    renderer = create_renderer(tok, GptOssRendererConfig(reasoning_effort="low"))
    args = '{"command":"ls -R"}'
    user = {"role": "user", "content": "List the files."}
    prompt = renderer.render_ids([user], add_generation_prompt=True)
    sampled = tok.encode(
        "List files.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.bash code"
        f"<|message|>{args}<|call|>",
        add_special_tokens=False,
    )
    (call,) = renderer.parse_response(sampled, prompt_ids=prompt).tool_calls
    assert (call.arguments_text, call.content_type) == (args, "code")

    def assistant_tokens(**extra):
        message = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "List files.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_0",
                    "function": {"name": "bash", "arguments": args},
                    **extra,
                }
            ],
        }
        rendered = renderer.render([user, message])
        return [
            t for t, i in zip(rendered.token_ids, rendered.message_indices) if i == 1
        ]

    # The prompt's generation header plus the sample are the assistant turn's tokens.
    turn = prompt[len(renderer.render_ids([user])) :] + sampled
    assert assistant_tokens(content_type=call.content_type) == turn
    assert assistant_tokens() != turn
    assert "<|constrain|>json" in tok.decode(assistant_tokens())


def test_a_tool_call_header_content_type_is_parsed_as_sampled():
    tok = load_tokenizer("openai/gpt-oss-20b")
    renderer = create_renderer(tok, GptOssRendererConfig(reasoning_effort="low"))
    prompt = renderer.render_ids(
        [{"role": "user", "content": "q"}], add_generation_prompt=True
    )

    def content_type(header):
        sampled = tok.encode(
            f"Go.<|end|><|start|>assistant{header}<|message|>{{}}<|call|>",
            add_special_tokens=False,
        )
        return (
            renderer.parse_response(sampled, prompt_ids=prompt)
            .tool_calls[0]
            .content_type
        )

    assert (
        content_type("<|channel|>commentary to=functions.bash <|constrain|>json")
        == "json"
    )
    assert content_type("<|channel|>commentary to=functions.bash code") == "code"
    assert (
        content_type(" to=functions.bash<|channel|>commentary <|constrain|>json")
        == "json"
    )
    # A header without a content type has none.
    assert content_type("<|channel|>commentary to=functions.bash") is None
    assert content_type(" to=functions.bash<|channel|>commentary") is None
