"""Token-response replays through the real client, renderer, and turn bridge.

Only the inference HTTP boundary is replayed. Tokenization, request rendering,
response parsing, tool dispatch, and continuation all use the production path.
"""

from functools import lru_cache

import pytest
from parity import TOOLS, models_for
from replay import assert_evidence, replay_client

from renderers import create_renderer
from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.client import generate
from renderers.configs import config_from_name


@lru_cache(maxsize=1)
def _tokenizer(model):
    return load_tokenizer(model)


@pytest.fixture(
    params=[
        case
        for case in models_for("shared")
        if case.resolved_renderer.startswith("glm-")
        or case.resolved_renderer == "deepseek-v4"
    ],
    ids=lambda case: case.model,
)
def rollout(request):
    case = request.param
    tok = _tokenizer(case.model)
    config = config_from_name(case.resolved_renderer)
    updates = {"thinking_retention": "all"}
    if "enable_thinking" in type(config).model_fields:
        updates["enable_thinking"] = True
    config = config.model_copy(update=updates)
    return tok, create_renderer(tok, config), case.model


def _tool_text(model, city):
    if model.startswith("deepseek"):
        return (
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n'
            f'<｜DSML｜parameter name="city" string="true">{city}</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )
    return (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        f"<arg_value>{city}</arg_value></tool_call>"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_call", [False, True])
async def test_literal_delimiters_survive_generate_and_tool_continuation(
    rollout, with_call
):
    tok, renderer, model = rollout

    def encode(text):
        return tok.encode(text, add_special_tokens=False)

    # Split encoding deliberately produces ordinary tokens spelling delimiters.
    # These are sampled tokens, so re-encoding their decoded text is incorrect.
    literal = (
        encode("Before <thi")
        + encode("nk>literal</thi")
        + encode("nk> <tool_")
        + encode("call>example</tool_")
        + encode("call> After")
    )
    expected_content = (
        "Before <think>literal</think> <tool_call>example</tool_call> After"
    )
    assert tok.decode(literal, skip_special_tokens=False) == expected_content
    assert tok.convert_tokens_to_ids("<think>") not in literal
    assert tok.convert_tokens_to_ids("</think>") not in literal
    assert tok.convert_tokens_to_ids("<tool_call>") not in literal
    argument = "東京<think>literal</think>"
    call = _tool_text(model, argument)
    sampled = encode("\n<think>Reasoning</think>") + literal
    if with_call:
        sampled += encode(call)
    sampled += renderer.get_stop_token_ids()[:1]
    answer = encode("<think>Checked</think>Sunny.") + renderer.get_stop_token_ids()[:1]
    requests = []
    messages = [{"role": "user", "content": "Check the weather."}]
    async with replay_client([(sampled, "stop"), (answer, "stop")], requests) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=model,
            messages=messages,
            tools=TOOLS,
            max_prompt_len=100000,
        )
        assert_evidence(result, sampled)
        assert result["content"] == expected_content
        assert result["reasoning_content"] == "\nReasoning"
        assert result["reasoning_complete"] is True
        assert requests[0]["token_ids"] == renderer.render_ids(
            messages, tools=TOOLS, add_generation_prompt=True
        )
        if not with_call:
            assert result["tool_calls"] == []
            assert result["finish_reason"] == "stop"
            return
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        tool = result["tool_calls"][0]
        assert tool.status == ToolCallParseStatus.OK
        assert tool.name == "get_weather"
        assert tool.arguments == {"city": argument}
        assert tool.raw in tok.decode(sampled[slice(*tool.token_span)])
        # Dispatch the parsed call and feed its result into the next real request.
        functions = {"get_weather": lambda city: f"Sunny in {city}."}
        observation = {
            "role": "tool",
            "name": tool.name,
            "tool_call_id": "weather-1",
            "content": functions[tool.name](**tool.arguments),
        }
        bridge = renderer.bridge_to_next_turn(
            result["prompt_ids"], sampled, [observation], tools=TOOLS
        )
        assert bridge is not None
        prefix = result["prompt_ids"] + sampled
        assert bridge.token_ids[: len(prefix)] == prefix
        assert not any(bridge.sampled_mask[len(prefix) :])
        followup = await generate(
            client=client,
            renderer=renderer,
            model=model,
            messages=[observation],
            tools=TOOLS,
            prompt_ids=bridge.token_ids,
            prompt_attribution=bridge,
            max_prompt_len=100000,
        )
        assert_evidence(followup, answer)
        assert requests[1]["token_ids"] == bridge.token_ids
        assert followup["prompt_attribution"] is bridge
        assert followup["content"] == "Sunny."
        assert followup["reasoning_content"] == "Checked"
        assert followup["reasoning_complete"] is True
        assert followup["tool_calls"] == []
        assert followup["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated", [False, True])
async def test_repeated_calls_keep_argument_markup_and_model_specific_tail(
    rollout, truncated
):
    tok, renderer, model = rollout
    # An opener inside a string argument must not become an additional call.
    argument = (
        '<｜DSML｜invoke name="literal">'
        if model.startswith("deepseek")
        else "<tool_call>literal"
    )
    first = _tool_text(model, argument)
    second = _tool_text(model, "Berlin")
    if truncated:
        second = second[: second.index("Berlin") + len("Berlin")]
    text = "\n<think>First<think>Second" + first + second
    if not truncated:
        text += "After<think>More"
    sampled = tok.encode(text, add_special_tokens=False)
    requests = []
    async with replay_client([(sampled, "length")], requests) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=model,
            messages=[{"role": "user", "content": "Check two cities."}],
            tools=TOOLS,
            max_prompt_len=100000,
        )
    assert_evidence(result, sampled)
    calls = result["tool_calls"]
    assert len(calls) == 2
    assert calls[0].arguments == {"city": argument}
    assert [c.status for c in calls] == [
        ToolCallParseStatus.OK,
        ToolCallParseStatus.UNCLOSED_BLOCK if truncated else ToolCallParseStatus.OK,
    ]
    for call in calls:
        assert call.raw in tok.decode(sampled[slice(*call.token_span)])
    if not truncated:
        assert calls[1].arguments == {"city": "Berlin"}
    reopened = not truncated and not model.startswith("deepseek")
    assert result["content"] == ("After" if reopened else "")
    assert result["reasoning_content"] == "\nFirstSecond" + ("More" if reopened else "")
    assert result["reasoning_complete"] is not reopened
    assert result["finish_reason"] == "length"
    bridge = renderer.bridge_to_next_turn(
        result["prompt_ids"],
        sampled,
        [{"role": "user", "content": "Continue."}],
        tools=TOOLS,
    )
    assert bridge is not None
    prefix = result["prompt_ids"] + sampled
    assert bridge.token_ids[: len(prefix)] == prefix
    extension = bridge.token_ids[len(prefix) :]
    close = tok.encode("</think>", add_special_tokens=False)
    assert (extension[: len(close)] == close) is reopened
    assert not any(bridge.sampled_mask[len(prefix) :])


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True])
async def test_unfinished_reasoning_survives_client_and_controls_bridge(
    rollout, stopped
):
    tok, renderer, model = rollout
    sampled = tok.encode(
        "\n<think>First</think>Before<think>Unfinished", add_special_tokens=False
    )
    if stopped:
        sampled += renderer.get_stop_token_ids()[:1]
    finish_reason = "stop" if stopped else "length"
    async with replay_client([(sampled, finish_reason)], []) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=model,
            messages=[{"role": "user", "content": "Think carefully."}],
            max_prompt_len=100000,
        )
    assert_evidence(result, sampled)
    assert result["content"] == "Before"
    assert result["reasoning_content"] == "\nFirstUnfinished"
    assert result["reasoning_complete"] is False
    assert result["tool_calls"] == []
    assert result["finish_reason"] == finish_reason
    bridge = renderer.bridge_to_next_turn(
        result["prompt_ids"], sampled, [{"role": "user", "content": "Continue."}]
    )
    if stopped:
        assert bridge is None
    else:
        assert bridge is not None
        prefix = result["prompt_ids"] + sampled
        assert bridge.token_ids[: len(prefix)] == prefix
        close = tok.encode("</think>", add_special_tokens=False)
        assert bridge.token_ids[len(prefix) : len(prefix) + len(close)] == close
        assert not any(bridge.sampled_mask[len(prefix) :])


def teardown_module():
    """Release model resources when this module finishes, not at process exit."""
    _tokenizer.cache_clear()
