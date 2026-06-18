import asyncio
import base64
import hashlib
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
        return httpx.Response(
            200,
            content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )


def test_run_image_dir_resolution_prefers_explicit_image_dir(tmp_path, monkeypatch):
    from renderers.mm_store import run_image_dir

    image_dir = tmp_path / "custom-images"
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(image_dir))
    monkeypatch.setenv("PRIME_RL_RUN_DIR", str(tmp_path / "run_other"))
    monkeypatch.setenv("RUN_ID", "other")

    assert run_image_dir() == image_dir.resolve()


def test_run_image_dir_resolution_owns_run_prefix(monkeypatch):
    from renderers.mm_store import run_image_dir

    monkeypatch.delenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", raising=False)
    monkeypatch.delenv("PRIME_RL_RUN_DIR", raising=False)
    monkeypatch.setenv("RUN_ID", "run_abc")

    assert run_image_dir().as_posix() == "/data/outputs/run_abc/assets/images"


class _TinyQwenTokenizer:
    unk_token_id = -1
    _specials = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|endoftext|>": 3,
        "<tool_call>": 4,
        "</tool_call>": 5,
        "<tool_response>": 6,
        "</tool_response>": 7,
        "</think>": 8,
        "<|vision_start|>": 9,
        "<|vision_end|>": 10,
        "<|image_pad|>": 11,
        "<|video_pad|>": 12,
    }

    def convert_tokens_to_ids(self, token):
        return self._specials.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        return [100 + ord(ch) % 50 for ch in text]


def test_qwen3_vl_render_emits_image_descriptor_without_processor(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    from renderers.mm_store import IMAGE_REF_PAYLOAD_KEY, IMAGE_REF_PAYLOAD_VALUE
    from renderers.qwen3_vl import Qwen3VLRenderer

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)
    renderer = Qwen3VLRenderer(_TinyQwenTokenizer())

    rendered = renderer.render(
        [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_path.as_uri()}}],
            }
        ],
        add_generation_prompt=True,
    )

    item = rendered.multi_modal_data.mm_items["image"][0]
    assert "pixel_values" not in item
    assert item["image_grid_thw"] == [[1, 16, 16]]
    assert item["raw_image_id"] == "image.png"
    assert item[IMAGE_REF_PAYLOAD_KEY] == IMAGE_REF_PAYLOAD_VALUE
    assert rendered.multi_modal_data.mm_placeholders["image"][0].length == 64


def test_generate_materialize_all_image_refs_rehydrates_descriptor_slots(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    from renderers.base import MultiModalData, ParsedResponse, PlaceholderRange
    from renderers.mm_store import split_image_ref
    from renderers.qwen3_vl import Qwen3VLRenderer

    class _RetryRenderer(Qwen3VLRenderer):
        supports_tools = True

        def get_stop_token_ids(self):
            return [99]

        def parse_response(self, completion_ids, *, tools=None):
            return ParsedResponse(content="done")

    image_dir = tmp_path / "run_retry" / "assets" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "image.png"
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_path)
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(image_dir))
    monkeypatch.setenv("RUN_ID", "retry")

    mm_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()[:32]
    mm_data = MultiModalData(
        mm_hashes={"image": [mm_hash]},
        mm_placeholders={"image": [PlaceholderRange(offset=5, length=64)]},
        mm_items={"image": [{"image_grid_thw": [[1, 16, 16]]}]},
    )
    renderer = _RetryRenderer(_TinyQwenTokenizer())
    client = _FakeClient()

    asyncio.run(
        generate(
            client=client,
            renderer=renderer,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image_path.as_uri()}}],
                }
            ],
            model="qwen3-vl",
            prompt_ids=list(range(20)),
            multi_modal_data=mm_data,
            sampling_params={"max_tokens": 4},
            materialize_all_image_refs=True,
        )
    )

    ref = client.calls[0]["body"]["features"]["kwargs_data"]["image"][0]
    run_id, _fingerprint, modality, parsed_hash, raw_image_id, grid = split_image_ref(ref)
    assert (run_id, modality, parsed_hash, raw_image_id, grid) == (
        "retry",
        "image",
        mm_hash,
        "image.png",
        [1, 16, 16],
    )


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


# ---------------------------------------------------------------------------
# Multimodal features payload.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "renderer_class_path",
    [
        "renderers.qwen3_vl:Qwen3VLRenderer",
        "renderers.qwen35:Qwen35Renderer",
    ],
    ids=["qwen3_vl", "qwen35"],
)
def test_generate_serializes_image_refs_for_qwen_vl_family(
    tmp_path, monkeypatch, renderer_class_path
):
    """When the renderer emits ``MultiModalData``, ``generate`` translates
    it into vLLM's ``features`` payload (mm_hashes + mm_placeholders +
    image-ref kwargs_data) and sticks it in the request body. Descriptor-only
    images stay ``None`` so vLLM can resolve them from its cache."""
    import importlib

    from renderers.base import (
        MultiModalData,
        ParsedResponse,
        PlaceholderRange,
    )
    from renderers.mm_store import (
        IMAGE_REF_PAYLOAD_KEY,
        IMAGE_REF_PAYLOAD_VALUE,
        image_layout_fingerprint,
        split_image_ref,
    )

    mod_name, cls_name = renderer_class_path.split(":")
    renderer_cls = getattr(importlib.import_module(mod_name), cls_name)

    class _BareRenderer(renderer_cls):
        supports_tools = True

        def get_stop_token_ids(self):
            return [99]

        def parse_response(self, completion_ids, *, tools=None):
            return ParsedResponse(content="done")

    renderer = _BareRenderer.__new__(_BareRenderer)
    image_dir = tmp_path / "run_rawtest" / "assets" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "image.png").write_bytes(b"image-bytes")
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(image_dir))
    monkeypatch.setenv("RUN_ID", "rawtest")
    fingerprint = image_layout_fingerprint(
        family="qwen_vl",
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=65536,
        max_pixels=16777216,
    )

    mm_data = MultiModalData(
        mm_hashes={"image": ["a" * 32, "b" * 32]},
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=5, length=1),
                PlaceholderRange(offset=10, length=1),
            ]
        },
        mm_items={
            "image": [
                {
                    "image_grid_thw": [[1, 2, 2]],
                    "raw_image_id": "image.png",
                    "image_layout_fingerprint": fingerprint,
                    IMAGE_REF_PAYLOAD_KEY: IMAGE_REF_PAYLOAD_VALUE,
                },
                {"image_grid_thw": [[1, 2, 2]]},
            ],
        },
    )

    client = _FakeClient()
    result = asyncio.run(
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
    assert features["mm_hashes"] == {"image": ["a" * 32, "b" * 32]}
    assert features["mm_placeholders"] == {
        "image": [{"offset": 5, "length": 1}, {"offset": 10, "length": 1}],
    }
    items = features["kwargs_data"]["image"]
    assert items[1] is None
    assert split_image_ref(items[0]) == (
        "rawtest",
        fingerprint,
        "image",
        "a" * 32,
        "image.png",
        [1, 2, 2],
    )
    assert "raw_image_id" not in result["multi_modal_data"].mm_items["image"][0]


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
