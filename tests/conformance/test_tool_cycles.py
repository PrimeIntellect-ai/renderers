"""Two tool cycles through the production client and exact reference bridges."""

import asyncio

import pytest

from renderers import config_from_name, create_renderer
from renderers.base import ToolCallParseStatus
from renderers.client import generate
from tests.conformance.harness import (
    TokenBackend,
    assert_attribution,
    assert_tokens,
    reference,
    renderer,
    tokenizer,
)
from tests.parity import MODEL_CATALOG, TOOLS


def cases():
    for case in MODEL_CATALOG:
        if "full-tool-cycle" not in case.excluded_scenarios:
            yield pytest.param(case, id=case.model)


def tool_renderer(case, kwargs):
    if case.resolved_renderer == "default":
        return create_renderer(
            tokenizer(case.model),
            config_from_name("default").model_copy(update={"tool_parser": "qwen3"}),
        )
    return renderer(case, kwargs)


def tool_turn(case, turn, city):
    call_id = (
        f"functions.get_weather:{turn}"
        if case.resolved_renderer in {"kimi-k2", "kimi-k2.5"}
        else f"call_{turn}"
    )
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Check weather.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": city}},
            }
        ],
    }
    new = [
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": call_id,
            "content": "Sunny.",
        }
    ]
    return assistant, new


def reference_turn(case, native, history, assistant, new, kwargs):
    prompt = reference(case, history, kwargs, tools=TOOLS, add_generation_prompt=True)
    finished = reference(case, history + [assistant], kwargs, tools=TOOLS)
    if finished[: len(prompt)] == prompt:
        completion = finished[len(prompt) :]
    else:
        # A closed live prefill may be absent in history. Locate the start of
        # the tool output by comparing two official completed serializations.
        assistant = dict(assistant)
        assistant.pop("reasoning_content", None)
        finished = reference(case, history + [assistant], kwargs, tools=TOOLS)
        empty = reference(
            case, history + [{"role": "assistant", "content": ""}], kwargs, tools=TOOLS
        )
        start = next(
            (i for i, (a, b) in enumerate(zip(finished, empty)) if a != b),
            min(len(finished), len(empty)),
        )
        completion = finished[start:]
    stops = set(native.get_stop_token_ids())
    end = next(
        (i + 1 for i in range(len(completion) - 1, -1, -1) if completion[i] in stops),
        None,
    )
    if end is not None:
        completion = completion[:end]
    else:
        extended = reference(case, history + [assistant] + new, kwargs, tools=TOOLS)
        stop = max(i for i, token in enumerate(extended) if token in stops)
        completion += extended[stop : stop + 1]
    return prompt, completion, assistant


def tool_extension(case, native, history, assistant, new, kwargs):
    following = reference(
        case,
        history + [assistant] + new,
        kwargs,
        tools=TOOLS,
        add_generation_prompt=True,
    )
    probe = dict(
        assistant,
        tool_calls=[
            {
                "id": "call_probe",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "BOUNDARY_PROBE_endXYZ"},
                },
            }
        ],
    )
    other = reference(
        case, history + [probe] + new, kwargs, tools=TOOLS, add_generation_prompt=True
    )
    common = 0
    while (
        common < min(len(following), len(other))
        and following[-common - 1] == other[-common - 1]
    ):
        common += 1
    suffix = following[len(following) - common :]
    stops = set(native.get_stop_token_ids())
    stop = next(i for i, token in enumerate(suffix) if token in stops)
    return suffix[stop + 1 :]


@pytest.mark.parametrize("case", tuple(cases()))
@pytest.mark.parametrize("city", ["Paris", "", "  Paris\nMünchen ☁  "])
def test_two_tool_cycles(case, city):
    kwargs = {"is_training": False} if case.resolved_renderer == "hy3" else {}
    native = tool_renderer(case, kwargs)
    history = [
        {"role": "system", "content": "Use the weather tool."},
        {"role": "user", "content": "Weather?"},
    ]
    next_prompt = None
    for turn in range(2):
        assistant, new = tool_turn(case, turn, city)
        prompt, completion, assistant = reference_turn(
            case, native, history, assistant, new, kwargs
        )
        if next_prompt is not None:
            prompt = next_prompt.token_ids
        backend = TokenBackend(prompt, completion)
        result = asyncio.run(
            generate(
                client=backend,
                renderer=native,
                messages=history,
                model=case.model,
                tools=TOOLS,
                max_prompt_len=1_000_000,
                prompt_ids=next_prompt.token_ids if next_prompt is not None else None,
                prompt_attribution=next_prompt,
            )
        )
        assert result["prompt_ids"] == prompt
        assert result["completion_ids"] == completion
        assert result["completion_logprobs"] == [-0.5] * len(completion)
        assert len(result["tool_calls"]) == 1
        call = result["tool_calls"][0]
        assert call.status == ToolCallParseStatus.OK
        assert call.name == "get_weather"
        assert call.arguments == {"city": city}
        assert result["finish_reason"] == "tool_calls"
        assistant = {
            "role": "assistant",
            "content": result["content"],
            "reasoning_content": result["reasoning_content"],
            "tool_calls": [
                {
                    "id": call.id or new[0]["tool_call_id"],
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
            ],
        }
        new[0]["tool_call_id"] = assistant["tool_calls"][0]["id"]
        bridge = native.bridge_to_next_turn(prompt, completion, new, tools=TOOLS)
        if bridge is not None:
            extension = tool_extension(case, native, history, assistant, new, kwargs)
            assert_tokens(bridge.token_ids, prompt + completion + extension, case)
            assert_attribution(bridge)
            assert not any(bridge.sampled_mask)
            next_prompt = bridge
        else:
            next_prompt = native.render(
                history + [assistant] + new, tools=TOOLS, add_generation_prompt=True
            )
            assert_tokens(
                next_prompt.token_ids,
                reference(
                    case,
                    history + [assistant] + new,
                    kwargs,
                    tools=TOOLS,
                    add_generation_prompt=True,
                ),
                case,
            )
        history += [assistant] + new
