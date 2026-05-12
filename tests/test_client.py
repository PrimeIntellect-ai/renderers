import asyncio
import base64

import numpy as np

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
        return RenderedTokens(token_ids=[1, 2, 3])

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def get_stop_token_ids(self):
        return [99]

    def parse_response(self, completion_ids: list[int]) -> ParsedResponse:
        assert completion_ids == [7, 8]
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

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        routed_experts = np.array([[[1]], [[2]]], dtype=np.int32)
        return {
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
                        "data": base64.b85encode(routed_experts.tobytes()).decode(
                            "ascii"
                        ),
                        "shape": list(routed_experts.shape),
                    },
                }
            ],
        }


def test_generate_builds_request_body_and_parses_response():
    client = _FakeClient()

    result = asyncio.run(
        generate(
            client=client,
            renderer=_FakeRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={"temperature": 0.3, "max_tokens": 7, "min_tokens": 2},
            cache_salt="ckpt-42",
        )
    )

    assert len(client.calls) == 1
    # /inference/v1/generate is mounted at the server root, so we post to
    # an absolute URL stripped of the OpenAI SDK's automatic /v1 prefix.
    assert client.calls[0]["path"] == "http://fake-host:8000/inference/v1/generate"
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
    assert result["routed_experts"] == [[[1]], [[2]]]
    assert result["multi_modal_data"] is None
    assert result["request_id"] == "gen-test"
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc.name == "echo"
    assert tc.arguments == {"text": "hello"}
    assert tc.status == ToolCallParseStatus.OK


class _MalformedToolRenderer(_FakeRenderer):
    """Returns only a malformed tool-call attempt — finish_reason must stay "stop"."""

    def parse_response(self, completion_ids: list[int]) -> ParsedResponse:
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


# ---------------------------------------------------------------------------
# Multimodal features payload.
# ---------------------------------------------------------------------------


def test_generate_serializes_multimodal_features_for_qwen3_vl():
    """When the renderer emits ``MultiModalData``, ``generate`` translates
    it into vLLM's ``features`` payload (mm_hashes + mm_placeholders +
    base64-encoded kwargs_data) and sticks it in the request body."""
    import pytest as _pytest

    _pytest.importorskip("torch")
    _pytest.importorskip("vllm", reason="vllm needed for features serialization")

    import torch as _torch

    from renderers.base import (
        MultiModalData,
        PlaceholderRange,
        load_tokenizer,
    )
    from renderers.qwen3_vl import Qwen3VLRenderer

    # Build a minimal real Qwen3VLRenderer so type dispatch in
    # _build_mm_features hits the qwen branch. The tokenizer is only
    # touched in __init__ to grab special-token ids; render() / etc.
    # aren't called here because we pre-supply prompt_ids + mm_data.
    tokenizer = load_tokenizer("Qwen/Qwen3-VL-4B-Instruct")
    renderer = Qwen3VLRenderer(tokenizer)

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
