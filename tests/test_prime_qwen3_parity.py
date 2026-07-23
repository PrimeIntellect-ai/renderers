from functools import lru_cache

import pytest
from renderers import create_renderer
from renderers.base import load_tokenizer

MODELS = [
    "PrimeIntellect/Qwen3-0.6B",
    "PrimeIntellect/Qwen3-1.7B",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                        "default": "Paris",
                        "examples": [{"country": "FR", "city": "Paris"}],
                    },
                    "days": {"type": "integer", "minimum": 1},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

CASES = [
    pytest.param(
        [{"role": "user", "content": "Hello"}],
        None,
        True,
        id="generation-prompt",
    ),
    pytest.param(
        [
            {"role": "user", "content": "Reverse abc"},
            {"role": "assistant", "content": "cba"},
        ],
        None,
        False,
        id="plain-assistant",
    ),
    pytest.param(
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": " simple arithmetic ",
                "content": " 4 ",
            },
        ],
        None,
        False,
        id="reasoning",
    ),
    pytest.param(
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "reasoning_content": "", "content": "4"},
        ],
        None,
        False,
        id="empty-reasoning",
    ),
    pytest.param(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "<think>raw</think>\nB"},
            {"role": "user", "content": "C"},
            {
                "role": "assistant",
                "reasoning_content": "kept",
                "content": "D",
            },
        ],
        None,
        False,
        id="multi-turn-reasoning",
    ),
    pytest.param(
        [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Weather?"},
        ],
        TOOLS,
        True,
        id="tools-with-system",
    ),
    pytest.param(
        [{"role": "user", "content": "Weather?"}],
        TOOLS,
        True,
        id="tools-with-default-system",
    ),
    pytest.param(
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "reasoning_content": "not rendered in the tool-call branch",
                "content": " Checking. ",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "Paris",
                                "days": 2,
                                "options": {"units": "metric", "locale": "fr"},
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London", "days": 3},
                        }
                    },
                ],
            },
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "tool", "content": '{"temp": 15}'},
            {"role": "assistant", "content": "Paris is warmer."},
        ],
        TOOLS,
        False,
        id="tool-cycle",
    ),
]


@lru_cache(maxsize=None)
def _load(model: str):
    tokenizer = load_tokenizer(model)
    return tokenizer, create_renderer(tokenizer)


def _expected(tokenizer, messages, tools, add_generation_prompt):
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
    )
    return list(result)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("messages,tools,add_generation_prompt", CASES)
def test_prime_qwen3_matches_hf_template(
    model,
    messages,
    tools,
    add_generation_prompt,
):
    tokenizer, renderer = _load(model)

    expected = _expected(tokenizer, messages, tools, add_generation_prompt)
    actual = renderer.render_ids(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
    )

    assert actual == expected


@pytest.mark.parametrize("model", MODELS)
def test_prime_qwen3_populates_assistant_masks(model):
    _, renderer = _load(model)
    prompt = [{"role": "user", "content": "Reverse abc"}]
    messages = [*prompt, {"role": "assistant", "content": "cba"}]

    generation_prompt = renderer.render_ids(prompt, add_generation_prompt=True)
    rendered = renderer.render(messages)

    assert rendered.token_ids[: len(generation_prompt)] == generation_prompt
    assert not any(rendered.sampled_mask[: len(generation_prompt)])
    assert any(rendered.sampled_mask[len(generation_prompt) :])
    assert rendered.sampled_mask[-1] is False
    assert (
        len(rendered.sampled_mask)
        == len(rendered.is_content)
        == len(rendered.token_ids)
    )


@pytest.mark.parametrize("model", MODELS)
def test_prime_qwen3_empty_think_roundtrip(model):
    """A sampled empty ``<think></think>`` block round-trips
    parse → message → re-render byte-identically with the bridged sampled
    stream. ``parse_qwen35`` used to collapse the empty-but-present block
    to ``reasoning_content=None``; the renderer's key-presence check then
    dropped the block on re-render, forking every later re-render of the
    rollout (the template keeps reasoning on ALL historical turns, and
    renders ``reasoning_content=""`` as ``<think></think>``)."""
    tokenizer, renderer = _load(model)
    stop = renderer.get_stop_token_ids()[0]

    prompt = [{"role": "user", "content": "Reverse abc"}]
    prompt_ids = renderer.render_ids(prompt, add_generation_prompt=True)
    # Tokenize the sampled turn in context so the completion carries the
    # exact BPE merges the model would produce continuing the stream.
    sampled = "<think></think>\ncba"
    full = tokenizer.encode(
        tokenizer.decode(prompt_ids) + sampled, add_special_tokens=False
    )
    assert full[: len(prompt_ids)] == prompt_ids
    completion_ids = full[len(prompt_ids) :] + [stop]

    parsed = renderer.parse_response(completion_ids)
    assert parsed.reasoning_content == ""  # present-but-empty, not None
    assert parsed.content == "cba"

    assistant = {"role": "assistant", "content": parsed.content}
    if parsed.reasoning_content is not None:
        assistant["reasoning_content"] = parsed.reasoning_content
    reminder = {"role": "user", "content": "Now reverse def"}

    bridged = renderer.bridge_to_next_turn(prompt_ids, completion_ids, [reminder])
    assert bridged is not None
    rerender = renderer.render_ids(
        [*prompt, assistant, reminder], add_generation_prompt=True
    )
    assert bridged.token_ids == rerender, (
        f"{model}: bridged sampled stream and re-render disagree:\n"
        f"bridge:   {tokenizer.decode(bridged.token_ids)!r}\n"
        f"rerender: {tokenizer.decode(rerender)!r}"
    )


@pytest.mark.parametrize("model", MODELS)
def test_prime_qwen3_no_think_parses_to_none(model):
    """Guard: a completion with no think block at all keeps
    ``reasoning_content=None``, so the key stays absent from stored
    messages and no ``<think></think>`` is invented on re-render."""
    tokenizer, renderer = _load(model)
    stop = renderer.get_stop_token_ids()[0]

    completion_ids = tokenizer.encode("cba", add_special_tokens=False) + [stop]
    parsed = renderer.parse_response(completion_ids)
    assert parsed.reasoning_content is None
    assert parsed.content == "cba"
