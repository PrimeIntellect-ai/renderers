import asyncio
import base64

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
        return RenderedTokens(token_ids=[1, 2, 3])

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
