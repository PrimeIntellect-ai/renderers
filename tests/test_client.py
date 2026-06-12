import asyncio
import base64
import json

import httpx
import numpy as np
import pytest
from renderers.base import (
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
)
from renderers.client import generate


class _FakeRenderer:
    supports_tools = True

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        assert messages == [{"role": "user", "content": "hi"}]
        assert tools == [{"type": "function", "function": {"name": "echo"}}]
        assert add_generation_prompt is True
        # Populate the full attribution surface so the test can verify
        # ``generate`` threads it through to the result dict unchanged.
        return RenderedTokens(
            token_ids=[1, 2, 3],
            message_indices=[0, 0, -1],
            sampled_mask=[False, False, False],
            is_content=[False, True, False],
            message_roles=["user"],
        )

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def get_stop_token_ids(self):
        return [99]

    def parse_response(
        self, completion_ids: list[int], *, tools=None
    ) -> ParsedResponse:
        assert completion_ids == [7, 8]
        # Stores tools so tests can assert the client plumbed them through.
        self._last_parse_tools = tools
        return ParsedResponse(
            content="done",
            reasoning_content="think",
            tool_calls=[
                ParsedToolCall(
                    raw='{"name": "echo", "arguments": {"text": "hello"}}',
                    name="echo",
                    arguments={"text": "hello"},
                    status=ToolCallParseStatus.OK,
                )
            ],
        )


class _FakeClient:
    """Mocks AsyncOpenAI's `.post()`. The renderer client builds an absolute
    URL off ``client.base_url``, so we expose one that includes the /v1 suffix
    the OpenAI SDK normally appends."""

    def __init__(self):
        self.calls = []
        self.base_url = "http://fake-host:8000/v1"

    @staticmethod
    def _as_response(payload, cast_to):
        """Both transports request cast_to=httpx.Response; deliver the payload as
        JSON bytes off the wire so the client exercises the zero-copy
        routed_experts strip. Falls back to the raw dict otherwise."""
        if cast_to is httpx.Response:
            return httpx.Response(
                200, content=json.dumps(payload, separators=(",", ":")).encode("utf-8")
            )
        return payload

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        routed_experts = np.array([[[1]], [[2]]], dtype=np.uint8)
        payload = {
            "request_id": "gen-test",
            "choices": [
                {
                    "index": 0,
                    "token_ids": [7, 8],
                    "logprobs": {
                        "content": [
                            {"token": "token_id:7", "logprob": -0.1},
                            {"token": "token_id:8", "logprob": -0.2},
                        ]
                    },
                    "finish_reason": "stop",
                    "routed_experts": {
                        "data": base64.b64encode(routed_experts.tobytes()).decode(
                            "ascii"
                        ),
                        "shape": list(routed_experts.shape),
                    },
                }
            ],
        }
        # vLLM path requests cast_to=httpx.Response; Dynamo path uses cast_to=dict.
        if cast_to is httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )
        return payload


def test_generate_builds_request_body_and_parses_response():
    client = _FakeClient()
    renderer = _FakeRenderer()

    result = asyncio.run(
        generate(
            client=client,
            renderer=renderer,
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={"temperature": 0.3, "max_tokens": 7, "min_tokens": 2},
            cache_salt="ckpt-42",
        )
    )

    # The client must plumb `tools` through to parse_response so XML-style
    # parsers can preserve declared-string args verbatim.
    assert renderer._last_parse_tools == [
        {"type": "function", "function": {"name": "echo"}}
    ]

    assert len(client.calls) == 1
    # /inference/v1/generate is mounted at the server root, so we post to
    # an absolute URL stripped of the OpenAI SDK's automatic /v1 prefix.
    assert client.calls[0]["path"] == "http://fake-host:8000/inference/v1/generate"
    assert client.calls[0]["cast_to"] is httpx.Response
    assert client.calls[0]["body"] == {
        "model": "test-model",
        "token_ids": [1, 2, 3],
        "cache_salt": "ckpt-42",
        "sampling_params": {
            "temperature": 0.3,
            "max_tokens": 7,
            "min_tokens": 2,
            "stop_token_ids": [99],
            "logprobs": 1,
            "skip_special_tokens": False,
        },
    }
    # finish_reason promoted from "stop" → "tool_calls" because the renderer
    # extracted at least one well-formed tool call client-side.
    assert result["finish_reason"] == "tool_calls"
    assert result["content"] == "done"
    assert result["reasoning_content"] == "think"
    assert result["prompt_ids"] == [1, 2, 3]
    assert result["completion_ids"] == [7, 8]
    assert result["completion_logprobs"] == [-0.1, -0.2]
    assert result["routed_experts"]["shape"] == [2, 1, 1]
    assert isinstance(result["routed_experts"]["data"], memoryview)
    assert result["routed_experts"]["data"].tobytes() == base64.b64encode(b"\x01\x02")
    assert result["multi_modal_data"] is None
    assert result["request_id"] == "gen-test"
    # Per-token attribution from the renderer surfaces on the result so
    # downstream consumers (verifiers RendererClient → prime-rl) can
    # build selective loss masks without a second render pass.
    attr = result["prompt_attribution"]
    assert attr is not None
    assert isinstance(attr, RenderedTokens)
    assert attr.token_ids == [1, 2, 3]
    assert attr.is_content == [False, True, False]
    assert attr.sampled_mask == [False, False, False]
    assert attr.message_indices == [0, 0, -1]
    assert attr.message_roles == ["user"]
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc.name == "echo"
    assert tc.arguments == {"text": "hello"}
    assert tc.status == ToolCallParseStatus.OK


class _MalformedToolRenderer(_FakeRenderer):
    """Returns only a malformed tool-call attempt — finish_reason must stay "stop"."""

    def parse_response(
        self, completion_ids: list[int], *, tools=None
    ) -> ParsedResponse:
        return ParsedResponse(
            content="",
            reasoning_content=None,
            tool_calls=[
                ParsedToolCall(
                    raw='{"name": "echo", broken',
                    status=ToolCallParseStatus.INVALID_JSON,
                )
            ],
        )


def test_generate_does_not_promote_finish_reason_for_malformed_tool_calls():
    """A malformed tool-call attempt must NOT promote finish_reason to
    "tool_calls" — only well-formed (status=OK) calls qualify. The
    malformed attempt is still preserved in ``tool_calls`` for verifier
    inspection, but the agent loop should not treat the turn as a
    successful tool invocation.
    """
    client = _FakeClient()
    result = asyncio.run(
        generate(
            client=client,
            renderer=_MalformedToolRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
        )
    )
    assert result["finish_reason"] == "stop"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0].status == ToolCallParseStatus.INVALID_JSON


class _NoRenderRenderer(_FakeRenderer):
    def render(self, messages, *, tools=None, add_generation_prompt=False):
        raise AssertionError("prebuilt prompt ids should skip render")

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        raise AssertionError("prebuilt prompt ids should skip render_ids")


def test_generate_uses_prebuilt_prompt_ids_without_rendering():
    client = _FakeClient()

    result = asyncio.run(
        generate(
            client=client,
            renderer=_NoRenderRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            prompt_ids=[11, 12, 13],
        )
    )

    assert client.calls[0]["body"]["token_ids"] == [11, 12, 13]
    assert result["prompt_ids"] == [11, 12, 13]
    # Pre-built prompt without explicit attribution → ``None`` carried
    # through. Consumers fall back to whatever attribution-free path
    # they have (e.g. uniform completion mask).
    assert result["prompt_attribution"] is None


def test_generate_threads_prompt_attribution_through_prebuilt_prompt_path():
    """When the caller passes both ``prompt_ids`` and ``prompt_attribution``
    (the multi-turn bridge path in verifiers), ``generate`` must thread
    the attribution through to the result dict unchanged — no re-rendering,
    no per-token reshuffling. Lets downstream consumers carry the
    renderer's body/scaffold cut into the trajectory step without an
    extra render pass."""
    client = _FakeClient()
    # Caller-supplied attribution; mirrors what
    # ``RendererClient._get_incremental_prompt_ids`` returns from the
    # bridge_to_next_turn output.
    supplied = RenderedTokens(
        token_ids=[11, 12, 13],
        message_indices=[-1, 0, 0],
        sampled_mask=[False, False, False],
        is_content=[False, True, True],
        message_roles=["tool"],
    )

    result = asyncio.run(
        generate(
            client=client,
            renderer=_NoRenderRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            prompt_ids=[11, 12, 13],
            prompt_attribution=supplied,
        )
    )

    # Exact passthrough — same object, no copy / no transform.
    assert result["prompt_attribution"] is supplied


class _DynamoFakeClient(_FakeClient):
    """Dynamo-shaped response: engine fields + routed_experts under nvext (not
    choices[0]); used to prove routed_experts now surfaces on dynamo."""

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        return self._as_response(
            {
                "id": "gen-dyn",
                "choices": [
                    {
                        "index": 0,
                        "logprobs": {
                            "content": [{"logprob": -0.1}, {"logprob": -0.2}]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "nvext": {
                    "engine_data": {"completion_token_ids": [7, 8]},
                    "routed_experts": {
                        # full-sequence routing (4 rows); worker can't trim
                        "data": "AQIDBA==",
                        "shape": [4, 1, 1],
                        "start": 0,
                        "dtype": "uint8",
                    },
                },
            },
            cast_to,
        )


def test_dynamo_transport_forwards_priority_and_detokenize():
    client = _DynamoFakeClient()

    result = asyncio.run(
        generate(
            client=client,
            renderer=_FakeRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={
                "temperature": 0.3,
                "max_tokens": 7,
                "detokenize": False,
                "allowed_token_ids": [7, 8],
                "bad_words_token_ids": [[1, 2]],
            },
            cache_salt="ckpt-42",
            priority=17,
            transport="dynamo",
        )
    )

    assert len(client.calls) == 1
    assert client.calls[0]["path"] == "/chat/completions"
    assert client.calls[0]["body"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": ""}],
        "stream": False,
        "nvext": {
            "token_data": [1, 2, 3],
            "extra_fields": ["engine_data"],
            "cache_salt": "ckpt-42",
            "agent_hints": {"priority": 17},
        },
        # tools are NOT forwarded on the wire (baked into token_data instead).
        "temperature": 0.3,
        "max_completion_tokens": 7,
        "logprobs": True,
        "skip_special_tokens": False,
        "stop_token_ids": [99],
        "bad_words_token_ids": [[1, 2]],
        "allowed_token_ids": [7, 8],
        "detokenize": False,
    }
    assert result["completion_ids"] == [7, 8]
    # routed_experts surfaces on dynamo as the {data, shape, start, dtype}
    # contract. No routed_experts_prompt_start is set here (first-turn case), so
    # the renderer does NOT trim — full-sequence routing passes through with
    # start=0.
    re = result["routed_experts"]
    assert re["shape"] == [4, 1, 1] and re["start"] == 0 and re["dtype"] == "uint8"
    # data rides as a zero-copy memoryview (not json-parsed), like vllm.
    assert isinstance(re["data"], memoryview)
    assert re["data"].tobytes() == b"AQIDBA=="


class _NoCompletionIdsClient(_FakeClient):
    """Dynamo response that carries no completion token IDs."""

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        return self._as_response(
            {"request_id": "x", "choices": [{"index": 0, "finish_reason": "stop"}]},
            cast_to,
        )


def test_dynamo_transport_raises_without_completion_ids():
    with pytest.raises(RuntimeError, match="completion token IDs"):
        asyncio.run(
            generate(
                client=_NoCompletionIdsClient(),
                renderer=_FakeRenderer(),
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
                tools=[{"type": "function", "function": {"name": "echo"}}],
                sampling_params={"max_tokens": 7},
                transport="dynamo",
                max_prompt_len=10_000,
            )
        )


class _EmptyCompletionClient(_FakeClient):
    """Dynamo response with a present-but-empty completion_token_ids list."""

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        return self._as_response(
            {
                "request_id": "x",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "logprobs": {"content": []}}
                ],
                "nvext": {"engine_data": {"completion_token_ids": []}},
            },
            cast_to,
        )


class _EmptyParseRenderer(_FakeRenderer):
    def parse_response(self, completion_ids, *, tools=None) -> ParsedResponse:
        assert completion_ids == []
        return ParsedResponse(content="", reasoning_content=None, tool_calls=[])


def test_dynamo_transport_allows_present_but_empty_completion():
    """A present-but-empty completion_token_ids is a valid zero-token completion
    and must NOT raise (only an absent field raises)."""
    result = asyncio.run(
        generate(
            client=_EmptyCompletionClient(),
            renderer=_EmptyParseRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={"max_tokens": 7},
            transport="dynamo",
            max_prompt_len=10_000,
        )
    )
    assert result["completion_ids"] == []


class _BoomClient(_FakeClient):
    async def post(self, path, *, cast_to=dict, body=None, options=None):
        raise ValueError("boom")


@pytest.mark.parametrize("transport", ["vllm", "dynamo"])
def test_generate_propagates_post_errors_raw(transport):
    # POST errors must propagate unchanged (no NameError from a stale handler).
    with pytest.raises(ValueError, match="boom"):
        asyncio.run(
            generate(
                client=_BoomClient(),
                renderer=_FakeRenderer(),
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
                tools=[{"type": "function", "function": {"name": "echo"}}],
                sampling_params={"max_tokens": 7},
                transport=transport,
                max_prompt_len=10_000,
            )
        )


def test_dynamo_transport_forwards_extra_sampling_fields_and_drops_denylist():
    """F1: sampling fields outside the old allowlist (presence_penalty, stop,
    guided_json) must reach the wire; vLLM-only/internal keys are dropped."""
    client = _DynamoFakeClient()
    asyncio.run(
        generate(
            client=client,
            renderer=_FakeRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={
                "max_tokens": 7,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.25,
                "stop": ["</s>"],
                "guided_json": {"type": "object"},
                # denylisted — must NOT hit the wire
                "return_token_ids": True,
                # not a top-level field (Dynamo rejects unknown ones); routed
                # into nvext.routed_experts_prompt_start so the worker trims
                # routing engine-side
                "routed_experts_prompt_start": 3,
            },
            transport="dynamo",
            max_prompt_len=10_000,
        )
    )
    body = client.calls[0]["body"]
    assert body["presence_penalty"] == 0.5
    assert body["frequency_penalty"] == 0.25
    assert body["stop"] == ["</s>"]
    assert body["guided_json"] == {"type": "object"}
    assert "return_token_ids" not in body
    # routed_experts_prompt_start is dropped from the top level (Dynamo rejects
    # unknown top-level fields) but routed into nvext so the worker applies it to
    # SamplingParams and trims routing engine-side.
    assert "routed_experts_prompt_start" not in body
    assert body["nvext"]["routed_experts_prompt_start"] == 3
    assert "extra_args" not in body.get("nvext", {})


def test_trim_dynamo_routed_experts():
    """Client-side trim is a back-compat fallback: it trims ONLY when the worker
    returned full routing (start=0) AND the caller supplied a positive
    routed_experts_prompt_start. No-op when the worker already trimmed (start>0),
    no offset is supplied (first turn), offset is 0, or routed_experts is absent."""
    from renderers.client import _trim_dynamo_routed_experts

    def _payload(channel):
        re = {
            "data": base64.b64encode(bytes([0, 1, 2, 3, 4])).decode(),
            "shape": [5, 1, 1], "start": 0, "dtype": "uint8",
        }
        return {"nvext": {channel: {"routed_experts": re}} if channel == "engine_data"
                else {"routed_experts": re}}

    # explicit prompt_start=3 -> drop 3 rows, start=3 (engine_data channel)
    resp = _payload("engine_data")
    _trim_dynamo_routed_experts(resp, {"routed_experts_prompt_start": 3})
    re = resp["nvext"]["engine_data"]["routed_experts"]
    assert re["shape"] == [2, 1, 1] and re["start"] == 3
    assert base64.b64decode(re["data"]) == bytes([3, 4])

    # explicit prompt_start=3 (top-level routed_experts channel)
    resp2 = _payload("routed_experts")
    _trim_dynamo_routed_experts(resp2, {"routed_experts_prompt_start": 3})
    re2 = resp2["nvext"]["routed_experts"]
    assert re2["shape"] == [2, 1, 1] and re2["start"] == 3

    # data as a zero-copy memoryview (the parse keeps the blob un-decoded) still
    # trims on the back-compat path: b64decode accepts memoryview.
    resp_mv = _payload("engine_data")
    re_mv = resp_mv["nvext"]["engine_data"]["routed_experts"]
    re_mv["data"] = memoryview(re_mv["data"].encode("ascii"))
    _trim_dynamo_routed_experts(resp_mv, {"routed_experts_prompt_start": 3})
    assert re_mv["shape"] == [2, 1, 1] and re_mv["start"] == 3
    assert base64.b64decode(re_mv["data"]) == bytes([3, 4])

    # worker already trimmed engine-side (start>0) -> no-op (don't double-trim)
    resp_wt = {"nvext": {"engine_data": {"routed_experts": {
        "data": base64.b64encode(bytes([3, 4])).decode(),
        "shape": [2, 1, 1], "start": 3, "dtype": "uint8",
    }}}}
    _trim_dynamo_routed_experts(resp_wt, {"routed_experts_prompt_start": 3})
    rewt = resp_wt["nvext"]["engine_data"]["routed_experts"]
    assert rewt["shape"] == [2, 1, 1] and rewt["start"] == 3
    assert base64.b64decode(rewt["data"]) == bytes([3, 4])

    # absent start (first turn) -> NO trim, full-sequence with start=0
    resp3 = _payload("engine_data")
    _trim_dynamo_routed_experts(resp3, {})
    re3 = resp3["nvext"]["engine_data"]["routed_experts"]
    assert re3["shape"] == [5, 1, 1] and re3["start"] == 0

    # offset 0 -> no-op
    resp0 = _payload("engine_data")
    _trim_dynamo_routed_experts(resp0, {"routed_experts_prompt_start": 0})
    assert resp0["nvext"]["engine_data"]["routed_experts"]["shape"] == [5, 1, 1]

    # absent routed_experts -> no-op
    resp4 = {"nvext": {"engine_data": {}}}
    _trim_dynamo_routed_experts(resp4, {"routed_experts_prompt_start": 3})
    assert resp4 == {"nvext": {"engine_data": {}}}


def test_dynamo_parse_present_empty_engine_logprobs_raises_for_nonempty_completion():
    """A present-but-empty engine_data.completion_logprobs is authoritative for
    source selection, but nonempty completions still require aligned logprobs."""
    data = {
        "choices": [
            {
                # chat-echo logprobs (would mismatch the engine ids)
                "logprobs": {"content": [{"logprob": -9.9}, {"logprob": -8.8}]},
                "finish_reason": "stop",
            }
        ],
        "nvext": {
            "engine_data": {
                "completion_token_ids": [7, 8],
                "completion_logprobs": [],
            }
        },
    }
    from renderers.client import _TRANSPORTS

    with pytest.raises(RuntimeError, match="logprobs length"):
        _TRANSPORTS["dynamo"].parse(data)


def test_dynamo_parse_falls_back_to_engine_routed_experts_for_placeholder():
    """A non-dict top-level routed_experts placeholder must not hide the valid
    engine_data.routed_experts payload."""
    data = {
        "choices": [{"index": 0, "finish_reason": "stop"}],
        "nvext": {
            "routed_experts": "placeholder",
            "engine_data": {
                "completion_token_ids": [7, 8],
                "completion_logprobs": [-0.1, -0.2],
                "routed_experts": {
                    "data": "AQI=",
                    "shape": [2, 1, 1],
                    "start": 3,
                    "dtype": "uint8",
                },
            },
        },
    }
    from renderers.client import _TRANSPORTS

    result = _TRANSPORTS["dynamo"].parse(data)

    assert result.routed_experts == {
        "data": "AQI=",
        "shape": [2, 1, 1],
        "start": 3,
        "dtype": "uint8",
    }


def test_dynamo_transport_merges_caller_nvext():
    """F2: caller-supplied nvext is merged — extra_fields union with engine_data,
    agent_hints merged with priority, unrelated caller keys preserved."""
    client = _DynamoFakeClient()
    asyncio.run(
        generate(
            client=client,
            renderer=_FakeRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={
                "max_tokens": 7,
                "nvext": {
                    "extra_fields": ["timing"],
                    "agent_hints": {"osl": 4},
                    "annotations": ["trace"],
                },
            },
            priority=9,
            transport="dynamo",
            max_prompt_len=10_000,
        )
    )
    nvext = client.calls[0]["body"]["nvext"]
    assert nvext["token_data"] == [1, 2, 3]
    # extra_fields union preserves caller "timing" + our "engine_data"
    assert nvext["extra_fields"] == ["timing", "engine_data"]
    # agent_hints merged: caller osl kept, priority overlaid
    assert nvext["agent_hints"] == {"osl": 4, "priority": 9}
    # unrelated caller nvext keys survive
    assert nvext["annotations"] == ["trace"]


def test_dynamo_transport_routes_sampling_params_cache_salt_and_priority_to_nvext():
    """cache_salt/priority supplied inside sampling_params (not the dedicated
    kwargs) must still land in nvext, never as top-level chat fields."""
    client = _DynamoFakeClient()
    asyncio.run(
        generate(
            client=client,
            renderer=_FakeRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={"max_tokens": 7, "cache_salt": "ckpt-9", "priority": 5},
            transport="dynamo",
            max_prompt_len=10_000,
        )
    )
    body = client.calls[0]["body"]
    assert body["nvext"]["cache_salt"] == "ckpt-9"
    assert body["nvext"]["agent_hints"] == {"priority": 5}
    # neither leaks to a top-level chat field
    assert "cache_salt" not in body
    assert "priority" not in body


class _BothTokenIdsClient(_FakeClient):
    """Dynamo response carrying engine_data.completion_token_ids AND a divergent
    choices[0].token_ids — the canonical engine channel must win (F3)."""

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        return self._as_response(
            {
                "id": "gen-dyn",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [99, 99],  # divergent echo — must be ignored
                        "logprobs": {
                            "content": [{"logprob": -0.1}, {"logprob": -0.2}]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "nvext": {"engine_data": {"completion_token_ids": [7, 8]}},
            },
            cast_to,
        )


def test_dynamo_transport_prefers_engine_data_over_choices_token_ids():
    client = _BothTokenIdsClient()
    result = asyncio.run(
        generate(
            client=client,
            renderer=_FakeRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={"max_tokens": 7},
            transport="dynamo",
            max_prompt_len=10_000,
        )
    )
    assert result["completion_ids"] == [7, 8]


class _MisalignedLogprobsClient(_FakeClient):
    """Dynamo response whose logprobs length disagrees with completion_ids (F4)."""

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        return self._as_response(
            {
                "id": "gen-dyn",
                "choices": [
                    {
                        "index": 0,
                        "logprobs": {"content": [{"logprob": -0.1}]},  # only 1 logprob
                        "finish_reason": "stop",
                    }
                ],
                "nvext": {"engine_data": {"completion_token_ids": [7, 8]}},  # 2 tokens
            },
            cast_to,
        )


def test_dynamo_transport_raises_on_logprob_length_mismatch():
    with pytest.raises(RuntimeError, match="logprobs length"):
        asyncio.run(
            generate(
                client=_MisalignedLogprobsClient(),
                renderer=_FakeRenderer(),
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
                tools=[{"type": "function", "function": {"name": "echo"}}],
                sampling_params={"max_tokens": 7},
                transport="dynamo",
                max_prompt_len=10_000,
            )
        )


# ---------------------------------------------------------------------------
# Multimodal features payload.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,renderer_class_path",
    [
        ("Qwen/Qwen3-VL-4B-Instruct", "renderers.qwen3_vl:Qwen3VLRenderer"),
        ("Qwen/Qwen3.5-2B", "renderers.qwen35:Qwen35Renderer"),
    ],
    ids=["qwen3_vl", "qwen35"],
)
def test_generate_serializes_multimodal_features_for_qwen_vl_family(
    model_id, renderer_class_path
):
    """When the renderer emits ``MultiModalData``, ``generate`` translates
    it into vLLM's ``features`` payload (mm_hashes + mm_placeholders +
    base64-encoded kwargs_data) and sticks it in the request body. Covers
    every renderer routed through ``_build_qwen_vl_features``."""
    import importlib

    pytest.importorskip("torch")
    pytest.importorskip("vllm", reason="vllm needed for features serialization")

    import torch as _torch
    from renderers.base import (
        MultiModalData,
        PlaceholderRange,
        load_tokenizer,
    )

    mod_name, cls_name = renderer_class_path.split(":")
    renderer_cls = getattr(importlib.import_module(mod_name), cls_name)

    # Build a minimal real renderer so type dispatch in
    # _build_mm_features hits the qwen branch. The tokenizer is only
    # touched in __init__ to grab special-token ids; render() / etc.
    # aren't called here because we pre-supply prompt_ids + mm_data.
    tokenizer = load_tokenizer(model_id)
    renderer = renderer_cls(tokenizer)

    # Two synthetic 1×2×2 images. Field factory expects pixel_values
    # shape ``(sum_HW, embed_dim)`` and grid_thw shape ``(N, 3)``; the
    # values themselves don't matter for the encoding round-trip.
    mm_data = MultiModalData(
        mm_hashes={"image": ["aaa", "bbb"]},
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=5, length=1),
                PlaceholderRange(offset=10, length=1),
            ]
        },
        mm_items={
            "image": [
                {
                    "pixel_values": _torch.zeros(4, 8, dtype=_torch.float32),
                    "image_grid_thw": _torch.tensor([[1, 2, 2]], dtype=_torch.int64),
                },
                {
                    "pixel_values": _torch.zeros(4, 8, dtype=_torch.float32),
                    "image_grid_thw": _torch.tensor([[1, 2, 2]], dtype=_torch.int64),
                },
            ],
        },
    )

    client = _FakeClient()
    asyncio.run(
        generate(
            client=client,
            renderer=renderer,
            messages=[],
            model="qwen3-vl",
            prompt_ids=list(range(20)),
            multi_modal_data=mm_data,
            sampling_params={"max_tokens": 4},
        )
    )

    body = client.calls[0]["body"]
    assert "features" in body, "multimodal call should attach features"
    features = body["features"]
    assert features["mm_hashes"] == {"image": ["aaa", "bbb"]}
    assert features["mm_placeholders"] == {
        "image": [{"offset": 5, "length": 1}, {"offset": 10, "length": 1}],
    }
    assert "kwargs_data" in features
    assert features["kwargs_data"] is not None
    assert "image" in features["kwargs_data"]
    assert len(features["kwargs_data"]["image"]) == 2
    # Items are base64 strings (encode_mm_kwargs_item output).
    for item in features["kwargs_data"]["image"]:
        assert isinstance(item, str) and len(item) > 0


# ---------------------------------------------------------------------------
# Prompt overflow handling.
# ---------------------------------------------------------------------------


class _LongRenderer(_FakeRenderer):
    """Renders a 10-token prompt regardless of input — enough to overflow a
    small ``max_prompt_len``."""

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        from renderers.base import RenderedTokens

        return RenderedTokens(token_ids=list(range(10)))


def test_generate_raises_overlong_prompt_when_explicit_cap_exceeded():
    """Pre-flight overflow check: when an explicit ``max_prompt_len`` is set
    and the rendered prompt is longer, ``generate`` raises
    ``OverlongPromptError`` without dispatching the request to the engine."""
    from renderers.client import OverlongPromptError

    client = _FakeClient()
    renderer = _LongRenderer()

    with pytest.raises(OverlongPromptError) as excinfo:
        asyncio.run(
            generate(
                client=client,
                renderer=renderer,
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
                max_prompt_len=4,
            )
        )

    assert excinfo.value.prompt_len == 10
    assert excinfo.value.max_prompt_len == 4
    assert client.calls == [], "request must not be dispatched on pre-flight fail"


def test_generate_allows_prompt_at_max_prompt_len():
    """A prompt exactly equal to ``max_prompt_len`` is allowed (the check is
    strict ``>``); only longer prompts trip the pre-flight."""
    client = _FakeClient()
    renderer = _LongRenderer()

    result = asyncio.run(
        generate(
            client=client,
            renderer=renderer,
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            max_prompt_len=10,
        )
    )

    assert len(client.calls) == 1
    assert result["prompt_ids"] == list(range(10))


def test_generate_auto_discovers_max_prompt_len_from_models_endpoint():
    """When ``max_prompt_len`` is ``None`` (default), ``generate`` discovers
    the cap via ``GET /v1/models`` and reads ``ModelCard.max_model_len``.
    The result is cached per ``(base_url, model)`` so subsequent calls
    don't re-query."""
    from renderers.client import OverlongPromptError, _max_prompt_len_cache

    class _ClientWithModels(_FakeClient):
        def __init__(self):
            super().__init__()
            self.base_url = "http://disco-host:8000/v1"
            self.models_calls = 0

        async def get(self, path, *, cast_to):
            self.models_calls += 1
            assert path == "/models"
            return {
                "object": "list",
                "data": [
                    {"id": "test-model", "max_model_len": 4},
                    {"id": "other", "max_model_len": 999},
                ],
            }

    # Clear cache so this test isn't affected by earlier ones.
    _max_prompt_len_cache.clear()

    client = _ClientWithModels()
    renderer = _LongRenderer()

    with pytest.raises(OverlongPromptError) as excinfo:
        asyncio.run(
            generate(
                client=client,
                renderer=renderer,
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
            )
        )

    assert excinfo.value.max_prompt_len == 4
    assert excinfo.value.prompt_len == 10
    assert client.models_calls == 1, "lookup must hit /models once"
    assert client.calls == [], "pre-flight must short-circuit the request"


def test_generate_caches_max_prompt_len_lookup_failure():
    """When ``GET /v1/models`` fails (e.g. mock client without ``.get``),
    the lookup result is cached as ``None`` and the pre-flight quietly
    disables — the request still goes through, callers fall back to
    whatever reactive overflow handling they have."""
    from renderers.client import _max_prompt_len_cache

    # _FakeClient has no .get method → AttributeError → cached None.
    _max_prompt_len_cache.clear()
    client = _FakeClient()
    client.base_url = "http://no-models:8000/v1"

    result = asyncio.run(
        generate(
            client=client,
            renderer=_LongRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
        )
    )

    # Request was dispatched (no pre-flight rejection) and round-tripped.
    assert len(client.calls) == 1
    assert result["prompt_ids"] == list(range(10))
    assert _max_prompt_len_cache[("http://no-models:8000/v1", "test-model")] is None
