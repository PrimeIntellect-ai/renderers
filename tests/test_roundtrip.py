"""Reference template → live generation prefix → client → parsed response.

Only inference HTTP is replayed. Expected tokens come from the checkpoint's
chat template (or independent native encoder), never from our renderer. Every
catalog checkpoint and every declared template-setting combination participates.
"""

import json
from functools import lru_cache

import pytest
from parity import MODEL_CATALOG, TOOLS, kwarg_combinations
from reference_continuations import ANSWER, REASONING, reference_continuation
from replay import assert_evidence, replay_client

from renderers import create_renderer
from renderers.base import MODEL_RENDERER_MAP, ToolCallParseStatus, load_tokenizer
from renderers.client import generate
from renderers.configs import config_from_name


# Both matrices visit checkpoints in groups; reuse within each group only.
_tokenizer = lru_cache(maxsize=1)(load_tokenizer)
# Default has no configured tool parser; Llama's template permits one call;
# Harmony's handoff ends generation at the first call. Other formats can emit
# multiple calls in one sampled continuation.
_CASES = [
    pytest.param(case, kwargs, calls, id=f"{case.model}-{kwargs}-calls={calls}")
    for case in MODEL_CATALOG
    for kwargs in kwarg_combinations(case)
    for calls in (
        [0]
        if case.resolved_renderer == "default"
        else [0, 1]
        if case.resolved_renderer in {"llama-3", "gpt-oss"}
        else [0, 1, 2]
    )
]


def test_continuation_catalog_covers_registered_renderers():
    assert set(MODEL_RENDERER_MAP.values()) <= {
        case.resolved_renderer for case in MODEL_CATALOG
    }


def _sampled_prefix(renderer, sampled):
    # A server stops at the first configured stop token. Several templates emit
    # trailing newlines after that token, which are history formatting, not sampled.
    stops = renderer.get_stop_token_ids()
    end = next(
        (i + 1 for i, token in enumerate(sampled) if token in stops), len(sampled)
    )
    return sampled[:end]


@pytest.mark.asyncio
@pytest.mark.parametrize("case,kwargs,calls", _CASES)
async def test_reference_continuation_through_generate(case, kwargs, calls):
    tok = _tokenizer(case.model)
    renderer = create_renderer(
        tok,
        config_from_name(case.resolved_renderer),
        chat_template_kwargs=kwargs or None,
    )
    tools = TOOLS if calls else None
    messages = [{"role": "user", "content": "Check the weather."}]
    prompt, sampled, reasoning = reference_continuation(
        tok, case, kwargs, messages, tools, calls=calls
    )
    sampled = _sampled_prefix(renderer, sampled)
    completions = [(sampled, "stop")]
    if reasoning:
        # Cut the SAME reference stream inside thinking, before its close, to
        # exercise length termination with the real generation prefill.
        cut = next(
            i
            for i in range(1, len(sampled) + 1)
            if tok.decode(sampled[:i]).rstrip().endswith(REASONING)
        )
        unfinished = sampled[:cut]
        completions.append((unfinished, "length"))
    requests = []
    async with replay_client(completions, requests) as client:

        async def run():
            return await generate(
                client=client,
                renderer=renderer,
                model=case.model,
                messages=messages,
                tools=tools,
                max_prompt_len=100000,
            )

        result = await run()
        assert requests[0]["token_ids"] == prompt
        assert result["prompt_ids"] == prompt
        assert_evidence(result, sampled)
        assert result["content"].strip() == ("" if calls else ANSWER)
        assert (result["reasoning_content"] or "").strip() == reasoning
        assert result["reasoning_complete"] is (
            None if case.resolved_renderer == "default" else True
        )
        assert result["finish_reason"] == ("tool_calls" if calls else "stop")
        assert len(result["tool_calls"]) == calls
        for call, city in zip(result["tool_calls"], ["Tokyo", "Berlin"]):
            assert call.status == ToolCallParseStatus.OK
            assert call.name == "get_weather"
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            assert arguments == {"city": city}
        if reasoning:
            truncated = await run()
            assert requests[1]["token_ids"] == prompt
            assert_evidence(truncated, unfinished)
            assert truncated["content"] == ""
            assert truncated["reasoning_content"].strip() == REASONING
            assert truncated["reasoning_complete"] is False
            assert truncated["tool_calls"] == []
            assert truncated["finish_reason"] == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [case for case in MODEL_CATALOG if case.resolved_renderer != "default"],
    ids=lambda case: case.model,
)
async def test_reference_continuation_after_tool_dispatch(case):
    """The tool observation can change the next generation's reasoning prefix."""
    tok = _tokenizer(case.model)
    renderer = create_renderer(tok, config_from_name(case.resolved_renderer))
    messages = [{"role": "user", "content": "Check the weather."}]
    prompt, sampled, _ = reference_continuation(tok, case, {}, messages, TOOLS, calls=1)
    sampled = _sampled_prefix(renderer, sampled)
    requests = []
    async with replay_client([(sampled, "stop")], requests) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=case.model,
            messages=messages,
            tools=TOOLS,
            max_prompt_len=100000,
        )
    assert requests[0]["token_ids"] == prompt
    (call,) = result["tool_calls"]
    assert call.status == ToolCallParseStatus.OK
    arguments = (
        json.loads(call.arguments)
        if isinstance(call.arguments, str)
        else call.arguments
    )
    observation = {"get_weather": lambda city: f"Sunny in {city}."}[call.name](
        **arguments
    )
    assistant = {
        "role": "assistant",
        "content": result["content"],
        "tool_calls": [
            {
                "id": "functions.get_weather:0",
                "type": "function",
                "function": {"name": call.name, "arguments": arguments},
            }
        ],
    }
    if result["reasoning_content"]:
        assistant["reasoning_content"] = result["reasoning_content"]
    if case.resolved_renderer in {"deepseek-v3", "deepseek-r1"}:
        assistant["content"] = None
        assistant["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    messages += [
        assistant,
        {
            "role": "tool",
            "name": call.name,
            "tool_call_id": "functions.get_weather:0",
            "content": observation,
        },
    ]
    prompt, sampled, reasoning = reference_continuation(tok, case, {}, messages, TOOLS)
    sampled = _sampled_prefix(renderer, sampled)
    requests = []
    async with replay_client([(sampled, "stop")], requests) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=case.model,
            prompt_ids=prompt,
            messages=messages,
            tools=TOOLS,
            max_prompt_len=100000,
        )
    # Feed the exact reference sampling prompt. Historical re-rendering may
    # deliberately retain reasoning that a template drops; it is not the oracle
    # for the parser's live channel state after a tool observation.
    assert requests[0]["token_ids"] == prompt
    assert_evidence(result, sampled)
    assert result["content"].strip() == ANSWER
    assert (result["reasoning_content"] or "").strip() == reasoning
    assert result["reasoning_complete"] is True
    assert result["tool_calls"] == []
    assert result["finish_reason"] == "stop"


def teardown_module():
    """Release model resources when this module finishes, not at process exit."""
    _tokenizer.cache_clear()
