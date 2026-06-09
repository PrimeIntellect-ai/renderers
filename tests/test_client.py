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
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def get_stop_token_ids(self):
        return [99]

    def parse_response(self, completion_ids: list[int], *, tools=None) -> ParsedResponse:
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
        self.calls.append({"path": path, "cast_to": cast_to, "body": body, "options": options})
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
                        "data": base64.b64encode(routed_experts.tobytes()).decode("ascii"),
                        "shape": list(routed_experts.shape),
                    },
                }
            ],
        }
        return httpx.Response(
            200,
            content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
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
    assert renderer._last_parse_tools == [{"type": "function", "function": {"name": "echo"}}]

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

    def parse_response(self, completion_ids: list[int], *, tools=None) -> ParsedResponse:
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
    "model_id,renderer_class_path",
    [
        ("Qwen/Qwen3-VL-4B-Instruct", "renderers.qwen3_vl:Qwen3VLRenderer"),
        ("Qwen/Qwen3.5-2B", "renderers.qwen35:Qwen35Renderer"),
    ],
    ids=["qwen3_vl", "qwen35"],
)
def test_generate_serializes_multimodal_features_for_qwen_vl_family(model_id, renderer_class_path, monkeypatch):
    """When the renderer emits ``MultiModalData``, ``generate`` translates
    it into vLLM's ``features`` payload (mm_hashes + mm_placeholders +
    base64-encoded kwargs_data) and sticks it in the request body. Covers
    every renderer routed through ``_build_qwen_vl_features``. Pins the store
    mode off so it exercises the inline-base64 path (the on path, which emits
    mmfile refs, is covered by ``test_qwen_vl_features_can_emit_mmfile_refs``)."""
    import importlib

    monkeypatch.setenv("RENDERERS_MM_FEATURE_STORE_MODE", "off")
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


def test_qwen_vl_features_can_emit_mmfile_refs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("vllm", reason="vllm needed for features serialization")

    import torch as _torch
    from renderers.base import MultiModalData, PlaceholderRange
    from renderers.client import _build_qwen_vl_features

    monkeypatch.setenv("RENDERERS_MM_FEATURE_STORE_MODE", "on")
    monkeypatch.setenv("PRIME_RL_MM_FEATURE_ROOT", str(tmp_path))
    monkeypatch.setenv("RUN_ID", "mmfiletest")

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
                    "pixel_values": _torch.zeros(4, 8, dtype=_torch.float32),
                    "image_grid_thw": _torch.tensor([[1, 2, 2]], dtype=_torch.int64),
                },
                {"image_grid_thw": _torch.tensor([[1, 2, 2]], dtype=_torch.int64)},
            ]
        },
    )

    features = _build_qwen_vl_features(mm_data, spatial_merge_size=2)

    items = features["kwargs_data"]["image"]
    assert items[0].startswith("mmfile:v1:mmfiletest:")
    assert items[0].endswith(":image:" + "a" * 32)
    assert items[1] is None
    assert len(list(tmp_path.rglob("*.msgpack"))) == 1


def test_qwen_vl_features_can_emit_mmraw_refs_without_processed_payloads(tmp_path, monkeypatch):
    from renderers.base import MultiModalData, PlaceholderRange
    from renderers.client import _build_qwen_vl_features
    from renderers.mm_store import (
        MM_RAW_PAYLOAD_KEY,
        MM_RAW_PAYLOAD_VALUE,
        mm_processor_fingerprint,
        raw_image_path,
        split_mmraw_ref,
    )

    monkeypatch.setenv("RENDERERS_MM_FEATURE_STORE_MODE", "raw")
    monkeypatch.setenv("PRIME_RL_MM_FEATURE_ROOT", str(tmp_path))
    monkeypatch.setenv("RUN_ID", "rawtest")
    raw_image_path(run_id="rawtest", raw_image_id="image.png").parent.mkdir(parents=True)
    raw_image_path(run_id="rawtest", raw_image_id="image.png").write_bytes(b"not-read-by-serializer")
    fingerprint = mm_processor_fingerprint(
        family="qwen_vl",
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=65536,
        max_pixels=16777216,
    )
    mm_hash = "a" * 32
    mm_data = MultiModalData(
        mm_hashes={"image": [mm_hash, "b" * 32]},
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
                    "mm_processor_fingerprint": fingerprint,
                    MM_RAW_PAYLOAD_KEY: MM_RAW_PAYLOAD_VALUE,
                },
                {"image_grid_thw": [[1, 2, 2]]},
            ]
        },
    )

    features = _build_qwen_vl_features(mm_data, spatial_merge_size=2)

    items = features["kwargs_data"]["image"]
    assert items[1] is None
    assert split_mmraw_ref(items[0]) == (
        "rawtest",
        fingerprint,
        "image",
        mm_hash,
        "image.png",
        [1, 2, 2],
    )
    assert list(tmp_path.rglob("*.msgpack")) == []


def test_strip_pixels_removes_one_request_raw_markers():
    from renderers.base import MultiModalData
    from renderers.client import _strip_pixels
    from renderers.mm_store import MM_RAW_PAYLOAD_KEY, MM_RAW_PAYLOAD_VALUE

    mm_data = MultiModalData(
        mm_items={
            "image": [
                {
                    "image_grid_thw": [[1, 2, 2]],
                    "raw_uri": "file:///tmp/image.png",
                    "raw_image_id": "image.png",
                    "mm_processor_fingerprint": "a" * 32,
                    MM_RAW_PAYLOAD_KEY: MM_RAW_PAYLOAD_VALUE,
                }
            ]
        }
    )

    stripped = _strip_pixels(mm_data)

    assert stripped.mm_items == {"image": [{"image_grid_thw": [[1, 2, 2]]}]}


def test_qwen3_vl_raw_mode_render_does_not_process_pixels(tmp_path, monkeypatch):
    import json

    from PIL import Image
    from renderers.mm_store import MM_RAW_PAYLOAD_KEY, MM_RAW_PAYLOAD_VALUE
    from renderers.qwen3_vl import Qwen3VLRenderer

    class _Tokenizer:
        unk_token_id = -1
        _specials = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<tool_call>": 4,
            "</tool_call>": 5,
            "<tool_response>": 6,
            "</tool_response>": 7,
            "<|vision_start|>": 8,
            "<|vision_end|>": 9,
            "<|image_pad|>": 10,
            "<|video_pad|>": 11,
        }

        def __init__(self, name_or_path):
            self.name_or_path = name_or_path

        def convert_tokens_to_ids(self, token):
            return self._specials.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            return [100 + ord(ch) % 50 for ch in text]

    monkeypatch.setenv("RENDERERS_MM_FEATURE_STORE_MODE", "raw")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            }
        )
    )
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(path)
    renderer = Qwen3VLRenderer(_Tokenizer(str(model_dir)), processor=object())

    rendered = renderer.render(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{path}"}}]}],
        add_generation_prompt=True,
    )

    item = rendered.multi_modal_data.mm_items["image"][0]
    assert "pixel_values" not in item
    assert item[MM_RAW_PAYLOAD_KEY] == MM_RAW_PAYLOAD_VALUE
    assert item["raw_image_id"] == "image.png"
    assert item["image_grid_thw"] == [[1, 16, 16]]
    assert rendered.multi_modal_data.mm_placeholders["image"][0].length == 64


def test_qwen3_vl_raw_layout_matches_real_processor(tmp_path, monkeypatch):
    from huggingface_hub import try_to_load_from_cache
    from PIL import Image

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    if not isinstance(try_to_load_from_cache(model_id, "preprocessor_config.json"), str):
        pytest.skip(f"{model_id} preprocessor_config.json is not cached locally")

    transformers = pytest.importorskip("transformers")
    from renderers.base import load_tokenizer
    from renderers.qwen3_vl import Qwen3VLRenderer, describe_qwen_image_layout

    monkeypatch.setenv("RENDERERS_MM_FEATURE_STORE_MODE", "raw")
    processor = transformers.AutoProcessor.from_pretrained(model_id, local_files_only=True)
    tokenizer = load_tokenizer(model_id)
    renderer = Qwen3VLRenderer(tokenizer)

    sizes = [
        (32, 32),
        (512, 512),
        (333, 777),
        (1200, 300),
        (4096, 2048),
        (65, 97),
    ]
    for width, height in sizes:
        path = tmp_path / f"image_{width}x{height}.png"
        Image.new("RGB", (width, height), color=(width % 255, height % 255, 7)).save(path)
        part = {"type": "image_url", "image_url": {"url": f"file://{path}"}}
        desc = describe_qwen_image_layout(renderer, part)
        with Image.open(path) as image:
            expected = processor.image_processor(images=[image.convert("RGB")], return_tensors="np")["image_grid_thw"][
                0
            ].tolist()
        assert desc.image_grid_thw == [expected]
        assert desc.num_image_tokens == int(expected[0] * expected[1] * expected[2]) // (
            processor.image_processor.merge_size**2
        )


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


def test_sweep_stale_artifacts_evicts_only_stale_features_never_images(tmp_path):
    import os
    import time

    from renderers.mm_store import sweep_stale_artifacts

    run_dir = tmp_path / "run_x"
    images = run_dir / "assets" / "images"
    features = run_dir / "assets" / "mm_features" / "v1"
    images.mkdir(parents=True)
    features.mkdir(parents=True)

    stale_img = images / "stale.jpg"
    fresh_img = images / "fresh.jpg"
    stale_feat = features / "stale.msgpack"
    fresh_feat = features / "fresh.msgpack"
    for p in (stale_img, fresh_img, stale_feat, fresh_feat):
        p.write_bytes(b"x")

    old = time.time() - 10_000
    os.utime(stale_img, (old, old))
    os.utime(stale_feat, (old, old))

    deleted = sweep_stale_artifacts(run_dir, ttl_seconds=3600.0)

    # Features only: the stale feature is evicted, the fresh feature kept.
    assert deleted == 1
    assert not stale_feat.exists()
    assert fresh_feat.exists()
    # Images are NEVER swept (terminal, non-regenerable source of truth) — even a
    # stale image is retained for the whole run.
    assert stale_img.exists()
    assert fresh_img.exists()


def test_sweep_stale_artifacts_noops_on_missing_dirs(tmp_path):
    from renderers.mm_store import sweep_stale_artifacts

    assert sweep_stale_artifacts(tmp_path / "does_not_exist", ttl_seconds=1.0) == 0


def test_mmfile_ref_emit_parse_roundtrip():
    """The ref shape is defined once: split_mmfile_ref is the exact inverse of
    mmfile_ref (guards against emit/parse drift across repos)."""
    from renderers.mm_store import mmfile_ref, split_mmfile_ref

    ref = mmfile_ref(run_id="run-a", fingerprint="deadbeef", modality="image", mm_hash="abc123")
    assert ref == "mmfile:v1:run-a:deadbeef:image:abc123"
    assert split_mmfile_ref(ref) == ("run-a", "deadbeef", "image", "abc123")
    # Legacy 5-part form → run_id is None (caller supplies it).
    assert split_mmfile_ref("mmfile:v1:fp:image:hash") == (None, "fp", "image", "hash")
    for bad in ("mmfile:v2:a:b:c:d", "notmmfile:v1:a:b:c:d", "mmfile:v1:a:b"):
        with pytest.raises(ValueError):
            split_mmfile_ref(bad)


def test_mmraw_ref_emit_parse_roundtrip(tmp_path, monkeypatch):
    from renderers.mm_store import mmraw_ref, raw_image_path, split_mmraw_ref

    monkeypatch.setenv("PRIME_RL_MM_FEATURE_ROOT", str(tmp_path))
    raw_image_path(run_id="run-a", raw_image_id="abc.png").parent.mkdir(parents=True)
    ref = mmraw_ref(
        run_id="run-a",
        fingerprint="deadbeefdeadbeef",
        modality="image",
        mm_hash="a" * 32,
        raw_image_id="abc.png",
        grid_thw=[[1, 2, 2]],
    )

    assert ref == "mmraw:v1:run-a:deadbeefdeadbeef:image:" + "a" * 32 + ":abc.png:1x2x2"
    assert split_mmraw_ref(ref) == ("run-a", "deadbeefdeadbeef", "image", "a" * 32, "abc.png", [1, 2, 2])
    for bad in ("mmraw:v2:a:b:c:d:e:f", "notmmraw:v1:a:b:c:d:e:f", "mmraw:v1:a:b:c"):
        with pytest.raises(ValueError):
            split_mmraw_ref(bad)
