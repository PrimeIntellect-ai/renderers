"""Inkling 2 attribution and completion round-trip regressions."""

import pytest

from renderers import Inkling2RendererConfig, build_training_sample, create_renderer
from renderers.base import ToolCallParseStatus
from tests.model_assets import load_test_tokenizer

pytestmark = pytest.mark.model

_MODELS = ("thinkingmachines/Inkling-2-Small", "thinkingmachines/Inkling-2-Mini")
_USER = {"role": "user", "content": "Find the result."}
_STOP = "<|content_model_end_sampling|>"


@pytest.fixture(scope="module", params=_MODELS)
def tokenizer(request):
    return load_test_tokenizer(request.param)


@pytest.mark.parametrize(
    "assistant",
    [
        {"content": "The answer"},
        {"content": [{"type": "text", "text": "One"}, {"type": "text", "text": "Two"}]},
        {"content": None},
        {"content": "The answer", "reasoning_content": "Reasoning"},
        {
            "content": None,
            "tool_calls": [{"function": {"name": "search", "arguments": {"q": "x"}}}],
        },
    ],
    ids=["text", "text-parts", "null", "reasoning", "tool"],
)
def test_sampled_mask_matches_completion(tokenizer, assistant):
    renderer = create_renderer(tokenizer)
    messages = [_USER, {"role": "assistant", **assistant}]
    prompt = renderer.render_ids([_USER], add_generation_prompt=True)
    rendered = renderer.render(messages)

    assert rendered.token_ids == renderer.render_ids(messages)
    assert rendered.token_ids[: len(prompt)] == prompt
    assert rendered.sampled_mask == [False] * len(prompt) + [True] * (
        len(rendered.token_ids) - len(prompt)
    )
    assert rendered.is_content[len(prompt) :] == rendered.sampled_mask[len(prompt) :]
    assert rendered.token_ids[-1] == tokenizer.convert_tokens_to_ids(_STOP)


def test_tool_content_loss_and_scaffold_attribution(tokenizer):
    renderer = create_renderer(tokenizer)
    messages = [
        {"role": "system", "content": "System one"},
        {"role": "system", "content": "System two"},
        _USER,
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "search", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "1",
            "content": [
                {"type": "text", "text": "Result one. "},
                {"type": "text", "text": "Result two."},
            ],
        },
        {"role": "assistant", "content": "Done"},
    ]
    rendered = renderer.render(messages)
    sample = build_training_sample(renderer, messages, content_sft_roles={"tool"})

    def content(index, mask):
        return tokenizer.decode(
            [
                token
                for token, owner, keep in zip(
                    rendered.token_ids, rendered.message_indices, mask
                )
                if owner == index and keep
            ],
            skip_special_tokens=False,
        )

    assert rendered.token_ids == renderer.render_ids(messages) == sample.token_ids
    assert content(0, rendered.is_content) == "System one"
    assert content(1, rendered.is_content) == "System two"
    assert content(4, sample.loss_mask) == "Result one. Result two."
    assert "Thinking effort level" in tokenizer.decode(
        [
            token
            for token, owner in zip(rendered.token_ids, rendered.message_indices)
            if owner == -1
        ]
    )


@pytest.mark.parametrize(
    ("assistant", "tail"),
    [
        ({"content": "The answer"}, [{"role": "user", "content": "And then?"}]),
        (
            {"content": "The answer", "reasoning_content": "Reasoning"},
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "One"},
                        {"type": "text", "text": "Two"},
                    ],
                }
            ],
        ),
        (
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {"name": "search", "arguments": {"q": "x"}},
                    }
                ],
            },
            [
                {
                    "role": "tool",
                    "name": "search",
                    "tool_call_id": "call_0",
                    "content": "Result",
                },
                {"role": "user", "content": "Go on"},
            ],
        ),
    ],
    ids=["text", "reasoning", "tool"],
)
def test_bridge_matches_full_render(tokenizer, assistant, tail):
    renderer = create_renderer(tokenizer)
    history = [_USER, {"role": "assistant", **assistant}]
    prompt = renderer.render_ids([_USER], add_generation_prompt=True)
    completion = renderer.render_ids(history)[len(prompt) :]
    bridged = renderer.bridge_to_next_turn(prompt, completion, tail)
    full = renderer.render(history + tail, add_generation_prompt=True)
    prior = len(prompt) + len(completion)

    assert bridged.token_ids == full.token_ids
    assert bridged.message_indices == [-1] * prior + [
        index - len(history) for index in full.message_indices[prior:]
    ]
    assert bridged.is_content == [False] * prior + full.is_content[prior:]
    assert not any(bridged.sampled_mask)


def test_bridge_keeps_sampled_tokens_and_declines_unprovable_extensions(tokenizer):
    renderer = create_renderer(tokenizer)
    prompt = renderer.render_ids([_USER], add_generation_prompt=True)
    # Unsorted keys and spacing: a full re-render would canonicalise this call.
    completion = tokenizer.encode(
        "<|message_model|>search<|content_invoke_tool_json|>"
        '{"name": "search", "args": {"q": "x", "a": 1}}<|end_message|>' + _STOP,
        add_special_tokens=False,
    )
    tool = {"role": "tool", "name": "search", "content": "Result"}
    bridged = renderer.bridge_to_next_turn(prompt, completion, [tool])

    assert bridged.token_ids == (
        prompt + completion + renderer.render_ids([_USER, tool])[len(prompt) :]
    )
    unnamed = {"role": "tool", "tool_call_id": "call_0", "content": "Result"}
    assistant = {"role": "assistant", "content": "x"}
    assert renderer.bridge_to_next_turn(prompt, completion[:-1], [tool]) is None
    assert renderer.bridge_to_next_turn(prompt, completion, [unnamed]) is None
    assert renderer.bridge_to_next_turn(prompt, completion, [assistant]) is None
    assert renderer.bridge_to_next_turn([], completion, [tool]) is None
    assert renderer.bridge_to_next_turn(prompt, completion, []) is None


@pytest.mark.parametrize("tool_parser", ["auto", None])
def test_reasoning_and_auto_tool_calls(tokenizer, tool_parser):
    renderer = create_renderer(
        tokenizer,
        None if tool_parser == "auto" else Inkling2RendererConfig(tool_parser=None),
    )
    assert renderer.supports_tools is (tool_parser == "auto")
    text = (
        "<|message_model|><|content_thinking|> Thinking. <|end_message|>"
        "<|message_model|><|content_text|>Answer.<|end_message|>"
    )
    if renderer.supports_tools:
        text += (
            '<|message_model|>search<|content_invoke_tool_json|>{"args":{"q":"x"}}'
            "<|end_message|>"
        )
    ids = tokenizer.encode(text + _STOP, add_special_tokens=False)
    parsed = renderer.parse_response(ids)

    assert parsed.reasoning_content == " Thinking. "
    assert parsed.content == "Answer."
    assert parsed.reasoning_complete is True
    assert [call.name for call in parsed.tool_calls] == (
        ["search"] if renderer.supports_tools else []
    )
    prefix = "<|message_model|><|content_thinking|> unfinished"
    partial = renderer.parse_response(
        tokenizer.encode(prefix, add_special_tokens=False)
    )
    assert partial.reasoning_content == " unfinished"
    assert partial.reasoning_complete is False
    assert partial.content == ""


def test_unprefixed_first_block(tokenizer):
    renderer = create_renderer(tokenizer)
    text = (
        "<|content_text|>hello<|end_message|>"
        '<|message_model|><|content_invoke_tool_json|>{"name": "search", "args": {"q": "x"}}'
        "<|end_message|>" + _STOP
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert parsed.content == "hello"
    assert [call.name for call in parsed.tool_calls] == ["search"]
    assert parsed.tool_calls[0].arguments == {"q": "x"}


def test_tool_call_inside_thinking_block(tokenizer):
    renderer = create_renderer(tokenizer)
    text = (
        "<|message_model|><|content_thinking|>reasoning"
        '<|content_invoke_tool_json|>{"name": "search", "args": {"q": "x"}}'
        "<|end_message|>" + _STOP
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert parsed.reasoning_content == "reasoning"
    assert parsed.reasoning_complete is True
    assert parsed.content == ""
    assert [(call.name, call.arguments, call.status) for call in parsed.tool_calls] == [
        ("search", {"q": "x"}, ToolCallParseStatus.OK)
    ]


@pytest.mark.parametrize(
    ("header", "payload", "closed", "status"),
    [
        ("search", '{"args":{}}', False, ToolCallParseStatus.UNCLOSED_BLOCK),
        ("search", "", True, ToolCallParseStatus.INVALID_JSON),
        ("", "{}", True, ToolCallParseStatus.MISSING_NAME),
        ("search", '{"name":123}', True, ToolCallParseStatus.MALFORMED_STRUCTURE),
        ("search", '{"args":[]}', True, ToolCallParseStatus.MALFORMED_STRUCTURE),
    ],
)
def test_malformed_tool_call_status(tokenizer, header, payload, closed, status):
    renderer = create_renderer(tokenizer)
    text = f"<|message_model|>{header}<|content_invoke_tool_json|>{payload}"
    if closed:
        text += "<|end_message|>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    (call,) = renderer.parse_response(ids).tool_calls

    assert call.status == status
    assert call.raw == payload
    assert call.token_span == (0, len(ids))


@pytest.mark.parametrize("method", ["render", "render_ids"])
def test_text_renderer_rejects_images(tokenizer, method):
    renderer = create_renderer(tokenizer)
    with pytest.raises(ValueError, match="text-only.*image"):
        getattr(renderer, method)(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "unused"}}],
                }
            ]
        )
