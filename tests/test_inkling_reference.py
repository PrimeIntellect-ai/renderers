"""Text parity against Thinking Machines' published Inkling renderers."""

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
_MODEL_ROLE = "<|message_model|>"
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
    params=[
        pytest.param(("Inkling", "Inkling"), id="Inkling"),
        pytest.param(("Inkling-Small", "Inkling"), id="Inkling-Small"),
        pytest.param(("Inkling-2-Small", "Inkling2"), id="Inkling-2-Small"),
        pytest.param(("Inkling-2-Mini", "Inkling2"), id="Inkling-2-Mini"),
    ],
)
def checkpoint(tml, request):
    model, reference_name = request.param
    tokenizer = load_test_tokenizer(f"thinkingmachines/{model}")
    reference = getattr(tml.renderers, reference_name)
    prefill = (
        [tokenizer.convert_tokens_to_ids(_MODEL_ROLE)]
        if reference_name == "Inkling"
        else []
    )
    return tokenizer, reference, prefill


def _token_ids(spans):
    return list(chain.from_iterable(span.span.int_tokens for span in spans))


@pytest.mark.parametrize("effort", [0.2, 0.9])
def test_generation_prompt_matches_tml(tml, checkpoint, effort):
    tokenizer, reference_class, prefill = checkpoint
    renderer = create_renderer(
        tokenizer, chat_template_kwargs={"reasoning_effort": effort}
    )
    reference = reference_class(effort=effort)
    history = [_SYSTEM, _USER, _TOOL_CALL, {**_TOOL_RESULT, "tool_call_id": "call_0"}]
    native = tml.chat.MessageList.from_openai_chat_dicts(history, tools=_TOOLS)
    spans, _ = reference.render_for_completion(native)

    extra_stops = [tokenizer.convert_tokens_to_ids("<|endoftext|>")] if prefill else []
    assert renderer.get_stop_token_ids() == reference.stop() + extra_stops
    assert renderer.render_ids(history, tools=_TOOLS, add_generation_prompt=True) == (
        _token_ids(spans) + prefill
    )


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "Quote <|end_message|> literally."},
        {**_TOOL_RESULT, "content": "Output includes <|message_model|> verbatim."},
    ],
    ids=["user", "tool"],
)
def test_literal_delimiter_matches_tml(tml, checkpoint, message):
    tokenizer, reference_class, prefill = checkpoint
    renderer = create_renderer(tokenizer)
    reference = reference_class(effort=0.9)
    history = [message]
    native = tml.chat.MessageList.from_openai_chat_dicts(history)
    spans, _ = reference.render_for_completion(native)

    expected = _token_ids(spans) + prefill
    assert renderer.render_ids(history, add_generation_prompt=True) == expected
    assert renderer.render(history, add_generation_prompt=True).token_ids == expected


def test_supervised_tokens_and_weights_match_tml(tml, checkpoint):
    tokenizer, reference_class, prefill = checkpoint
    renderer = create_renderer(tokenizer)
    reference = reference_class(effort=0.9)
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
    expected_weights = example.per_token_loss_weight_unshifted
    if prefill:
        # Our original Inkling generation prompt supplies the first assistant role token.
        for role, span in zip(
            rendered.message_roles, rendered.message_token_spans(), strict=True
        ):
            if role == "assistant":
                assert span is not None
                start, _ = span
                assert rendered.token_ids[start : start + len(prefill)] == prefill
                expected_weights[start] = 0.0
    assert rendered.sampled_mask == expected_weights


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
            _SAMPLED_TOOL_CALL + _STOP,
            [_TOOL_RESULT],
            id="noncanonical-tool-json",
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
def test_sampled_turn_matches_tml(tml, checkpoint, sampled_text, tail):
    tokenizer, reference_class, prefill = checkpoint
    renderer = create_renderer(tokenizer)
    reference = reference_class(effort=0.9)
    native = tml.chat.MessageList.from_openai_chat_dicts([_SYSTEM, _USER], tools=_TOOLS)
    prefix, parser = reference.render_for_completion(native)
    prompt = renderer.render_ids(
        [_SYSTEM, _USER], tools=_TOOLS, add_generation_prompt=True
    )
    reference_prompt = _token_ids(prefix)
    assert prompt == reference_prompt + prefill
    completion = tokenizer.encode(sampled_text, add_special_tokens=False)
    assert completion[: len(prefill)] == prefill
    local_completion = completion[len(prefill) :]
    parsed = renderer.parse_response(local_completion, prompt_ids=prompt)
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
    assert _token_ids(spans) == reference_prompt + completion + _token_ids(suffix)
    bridged = renderer.bridge_to_next_turn(prompt, local_completion, tail, tools=_TOOLS)
    assert bridged is not None
    stop = reference.stop()
    inserted_stop = stop if prefill and completion[-1:] != stop else []
    assert bridged.token_ids == (
        prompt + local_completion + inserted_stop + _token_ids(suffix) + prefill
    )
