"""Sampled-token stability with thinking disabled.

For the affected Qwen and Gemma 4 variants, the generation prompt prefills an
empty thinking wrapper, so that wrapper is part of every sampled turn's token
stream. In an agentic rollout the conversation keeps growing (tool responses,
budget / prune-reminder user messages, ...) and may be re-rendered from
messages. The upstream Jinja templates strip the wrapper from assistant
history, so a re-render would diverge token-for-token from the stream the model
actually sampled — forking any token-identity comparison on byte-identical
messages.

These tests pin the renderer-side guarantee that fixes this: with thinking
disabled, historical assistant turns without ``reasoning_content`` re-emit
the exact gen-prompt wrapper, so the tokens a turn was sampled with are a
prefix of every later re-render of the grown conversation, and the bridge
and a full re-render agree.

This is a deliberate, documented deviation from ``apply_chat_template`` —
see the carve-out in ``test_parity.py`` and the module
docstrings of the affected renderers.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from parity import models_for
from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import _config_class_for
from renderers.token_arrays import (
    FixedWidthArrayBuilder,
    TOKEN_IDS_DTYPE,
    encode_token_ids,
    owned_token_ids_from_array,
)

# One representative model per affected renderer family, each with
# thinking explicitly disabled.
_MODELS = [
    (
        case.model,
        _config_class_for(case.resolved_renderer)(enable_thinking=False),
    )
    for case in models_for("disabled-thinking")
]

_EMPTY_WRAPPERS = {
    "google/gemma-4-26B-A4B-it": "<|channel>thought\n<channel|>",
}


@lru_cache(maxsize=None)
def _load(model_name: str):
    return load_tokenizer(model_name)


def pytest_generate_tests(metafunc):
    if "dt_model" in metafunc.fixturenames:
        metafunc.parametrize("dt_model,dt_config", _MODELS, ids=[m for m, _ in _MODELS])


def test_sampled_stream_is_prefix_of_rerender(dt_model, dt_config):
    """The rollout invariant: every generation prompt (which ends with the
    prefilled empty wrapper) stays a byte prefix of all later re-renders of
    the grown conversation — even after new real user queries arrive."""
    tok = _load(dt_model)
    renderer = create_renderer(tok, dt_config)

    convo: list[dict] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    prev = renderer.render_ids(convo, add_generation_prompt=True)
    steps = [
        (
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "You are running low on budget."},
        ),
        (
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Submit your final answer now."},
        ),
    ]
    for assistant, user in steps:
        convo += [assistant, user]
        cur = renderer.render_ids(convo, add_generation_prompt=True)
        assert np.array_equal(cur[: len(prev)], prev), (
            f"{dt_model}: re-render after appending {assistant['content']!r} "
            "diverged from the earlier generation prompt — historical turn "
            "lost its prefilled empty think wrapper"
        )
        prev = cur


def test_wrapper_reemitted_on_historical_turn(dt_model, dt_config):
    """A historical assistant turn (before the last real user query, no
    reasoning_content) renders with the exact empty wrapper bytes the
    generation prompt prefilled."""
    tok = _load(dt_model)
    renderer = create_renderer(tok, dt_config)

    rendered = renderer.render(
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Now 3+3?"},
        ]
    )
    turn = rendered.token_ids[rendered.message_indices == 1]
    text = tok.decode(turn)
    expected_wrapper = _EMPTY_WRAPPERS.get(dt_model, "<think>\n\n</think>\n\n")
    assert expected_wrapper in text, (
        f"{dt_model}: historical assistant turn rendered without the "
        f"prefilled empty think wrapper: {text!r}"
    )


def test_bridge_and_rerender_agree(dt_model, dt_config):
    """Extending a sampled turn via the bridge and re-rendering the same
    conversation from messages must produce identical tokens — the property
    whose violation shows up as token-level forks on byte-identical
    trajectories."""
    tok = _load(dt_model)
    renderer = create_renderer(tok, dt_config)

    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    prompt_ids = renderer.render_ids(msgs, add_generation_prompt=True)
    # Simulate the sampled completion: content tokens then the stop token.
    content_ids = encode_token_ids(tok, "Paris.")
    completion_builder = FixedWidthArrayBuilder(
        TOKEN_IDS_DTYPE, initial_capacity=content_ids.size + 1
    )
    completion_builder.extend(content_ids)
    completion_builder.append(renderer.get_stop_token_ids()[0])
    completion_ids = completion_builder.finish()
    reminder = {"role": "user", "content": "You are running low on budget."}

    bridged = renderer.bridge_to_next_turn(prompt_ids, completion_ids, [reminder])
    assert bridged is not None, (
        f"{dt_model}: bridge refused with thinking disabled (implied "
        "thinking_retention='all' should allow it)"
    )

    rerender = renderer.render_ids(
        msgs + [{"role": "assistant", "content": "Paris."}, reminder],
        add_generation_prompt=True,
    )
    assert np.array_equal(bridged.token_ids, rerender), (
        f"{dt_model}: bridge-extended tokens and full re-render disagree:\n"
        f"bridge:   {tok.decode(bridged.token_ids)!r}\n"
        f"rerender: {tok.decode(rerender)!r}"
    )


def test_tool_cycle_rerender_stays_prefix_stable(dt_model, dt_config):
    """Tool-call turns (empty content, no reasoning) keep the prefilled
    wrapper too. On qwen3 the template window additionally requires
    ``is_last or reasoning_content``, so without the deviation a mid-cycle
    re-render would strip the wrapper from the tool-call turn even though
    it sits after the last real user query."""
    tok = _load(dt_model)
    renderer = create_renderer(tok, dt_config)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    msgs: list[dict] = [{"role": "user", "content": "Weather in Paris?"}]
    prev = renderer.render_ids(msgs, tools=tools, add_generation_prompt=True)

    msgs += [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
        {"role": "tool", "content": '{"temp": 20}'},
    ]
    cur = renderer.render_ids(msgs, tools=tools, add_generation_prompt=True)
    assert np.array_equal(cur[: len(prev)], prev), (
        f"{dt_model}: tool-call turn lost its prefilled empty think wrapper "
        "on re-render"
    )

    msgs += [
        {"role": "assistant", "content": "It is 20 degrees."},
        {"role": "user", "content": "You are running low on budget."},
    ]
    cur2 = renderer.render_ids(msgs, tools=tools, add_generation_prompt=True)
    assert np.array_equal(cur2[: len(cur)], cur)


def test_historical_reasoning_stays_template_faithful(dt_model, dt_config):
    """Scope guard: turns that carry ``reasoning_content`` were not sampled
    under ``enable_thinking=False``, so they keep exact template parity —
    the deviation fires only for the empty wrapper."""
    tok = _load(dt_model)
    renderer = create_renderer(tok, dt_config)

    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "Adding small ints.",
            "content": "4",
        },
        {"role": "user", "content": "Now 3+3?"},
        {
            "role": "assistant",
            "reasoning_content": "Same idea.",
            "content": "6",
        },
    ]
    ours = renderer.render_ids(msgs)
    theirs = owned_token_ids_from_array(
        "apply_chat_template",
        tok.apply_chat_template(
            msgs,
            tokenize=True,
            return_dict=False,
            return_tensors="np",
            add_generation_prompt=False,
            enable_thinking=False,
        ),
    )
    assert np.array_equal(ours, theirs)
