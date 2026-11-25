"""Token parity against Thinking Machines' published Inkling 2 renderer."""

import json
from itertools import chain

import pytest

from renderers import create_renderer
from tests.model_assets import load_test_tokenizer

pytestmark = pytest.mark.model

_SYSTEM = {"role": "system", "content": "Use tools when needed."}
_USER = {"role": "user", "content": "Find the result."}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Recherche à Zürich",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    },
]
_TOOL_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q":"Zürich","a":1}'},
        },
    ],
}
_TOOL_RESULT = {"role": "tool", "name": "search", "content": "Résultat"}
_SAMPLED_TOOL_CALL = (
    "<|message_model|>search<|content_invoke_tool_json|>"
    '{"name": "search", "args": {"q": "Zürich", "a": 1}}<|end_message|>'
)
_STOP = "<|content_model_end_sampling|>"


@pytest.fixture(scope="module")
def tml():
    return pytest.importorskip(
        "tml_renderers", reason="Install tml-renderers==0.4.2 to run reference parity"
    )


@pytest.fixture(
    scope="module",
    params=["thinkingmachines/Inkling-2-Small", "thinkingmachines/Inkling-2-Mini"],
)
def tokenizer(tml, request):
    return load_test_tokenizer(request.param)


def _token_ids(spans):
    return list(chain.from_iterable(span.span.int_tokens for span in spans))


@pytest.mark.parametrize("effort", [0.2, 0.9])
def test_generation_prompt_matches_tml(tml, tokenizer, effort):
    renderer = create_renderer(
        tokenizer, chat_template_kwargs={"reasoning_effort": effort}
    )
    reference = tml.renderers.Inkling2(effort=effort)
    history = [_SYSTEM, _USER, _TOOL_CALL, {**_TOOL_RESULT, "tool_call_id": "call_0"}]
    native = tml.chat.MessageList.from_openai_chat_dicts(history, tools=_TOOLS)
    spans, _ = reference.render_for_completion(native)

    assert renderer.get_stop_token_ids() == reference.stop()
    assert renderer.render_ids(history, tools=_TOOLS, add_generation_prompt=True) == (
        _token_ids(spans)
    )


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "Quote <|end_message|> literally."},
        {**_TOOL_RESULT, "content": "Output includes <|message_model|> verbatim."},
    ],
    ids=["user", "tool"],
)
def test_literal_delimiter_matches_tml(tml, tokenizer, message):
    renderer = create_renderer(tokenizer)
    reference = tml.renderers.Inkling2(effort=0.9)
    history = [message]
    native = tml.chat.MessageList.from_openai_chat_dicts(history)
    spans, _ = reference.render_for_completion(native)

    expected = _token_ids(spans)
    assert renderer.render_ids(history, add_generation_prompt=True) == expected
    assert renderer.render(history, add_generation_prompt=True).token_ids == expected


def test_supervised_tokens_and_weights_match_tml(tml, tokenizer):
    renderer = create_renderer(tokenizer)
    reference = tml.renderers.Inkling2(effort=0.9)
    history = [
        _SYSTEM,
        _USER,
        _TOOL_CALL,
        {**_TOOL_RESULT, "tool_call_id": "call_0"},
        {
            "role": "assistant",
            "content": "The answer.",
            "reasoning_content": "Use the result.",
        },
    ]
    native = tml.chat.MessageList.from_openai_chat_dicts(history, tools=_TOOLS)
    (example,) = reference.render_for_sft(native)
    rendered = renderer.render(history, tools=_TOOLS)

    assert rendered.token_ids == _token_ids(example.input_token_spans)
    assert rendered.sampled_mask == example.per_token_loss_weight_unshifted


@pytest.mark.parametrize(
    ("sampled_text", "tail"),
    [
        pytest.param(
            "<|message_model|><|content_thinking|>Think<|end_message|>"
            "<|message_model|><|content_text|>The answer.<|end_message|>" + _STOP,
            [{"role": "user", "content": "And then?"}],
            id="text-and-reasoning",
        ),
        pytest.param(
            _SAMPLED_TOOL_CALL + _STOP, [_TOOL_RESULT], id="noncanonical-tool-json"
        ),
        pytest.param(
            _SAMPLED_TOOL_CALL,
            [
                {
                    **_TOOL_RESULT,
                    "content": "Output includes <|message_model|> verbatim.",
                }
            ],
            id="closed-tool-without-sampling-stop",
        ),
    ],
)
def test_sampled_turn_matches_tml(tml, tokenizer, sampled_text, tail):
    renderer = create_renderer(tokenizer)
    reference = tml.renderers.Inkling2(effort=0.9)
    native = tml.chat.MessageList.from_openai_chat_dicts([_SYSTEM, _USER], tools=_TOOLS)
    prefix, parser = reference.render_for_completion(native)
    prompt = renderer.render_ids(
        [_SYSTEM, _USER], tools=_TOOLS, add_generation_prompt=True
    )
    assert prompt == _token_ids(prefix)
    completion = tokenizer.encode(sampled_text, add_special_tokens=False)
    parsed = renderer.parse_response(completion)
    reference_messages = parser.parse_tokens(completion).messages
    (reference_response,) = tml.chat.MessageList(
        reference_messages
    ).to_openai_chat_dicts()
    assert parsed.content == (reference_response["content"] or "")
    assert parsed.reasoning_content == reference_response.get("reasoning_content")
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in reference_response.get("tool_calls", [])
    ]

    sampled = tml.chat.TokenSpan(tml.chat.EncodedTextTokenSpan(completion))
    spans, suffix, _ = reference.bridge_to_next_turn(
        [*prefix, sampled],
        [*native.messages, *reference_messages],
        tml.chat.MessageList.from_openai_chat_dicts(tail),
    )
    assert _token_ids(spans) == prompt + completion + _token_ids(suffix)
    bridged = renderer.bridge_to_next_turn(prompt, completion, tail, tools=_TOOLS)
    assert bridged is not None
    assert bridged.token_ids == _token_ids(spans)
