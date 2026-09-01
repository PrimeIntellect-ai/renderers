"""Renderer-based generate client for vLLM's /inference/v1/generate.

messages → Renderer.render_ids() → token IDs → POST /inference/v1/generate
→ completion tokens → Renderer.parse_response() → structured message
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections.abc import Mapping
from typing import Any, cast

import httpx
import numpy as np
from openai import AsyncOpenAI

from renderers.base import (
    Message,
    MultiModalData,
    RenderedTokens,
    Renderer,
    ToolCallParseStatus,
    ToolSpec,
    _require_transformers,
)
from renderers.token_arrays import (
    LOGPROBS_DTYPE,
    OFFSETS_DTYPE,
    TOKEN_IDS_DTYPE,
    require_1d_array,
    require_readonly,
    require_range_array,
)

_request_logger = logging.getLogger("renderers.client")
ROUTED_EXPERTS_DATA_PREFIX = b'"routed_experts":{"data":"'
KEPT_TOKENS_IDS_PREFIX = b'"kept_tokens":{"ids":"'
# vLLM uses this value both when sampled-token evidence is missing and as a
# lower-bound clamp, so receiving it cannot prove the real logprob was returned.
VLLM_LOGPROB_SENTINEL = -9999.0
_BASE64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


class OverlongPromptError(Exception):
    """The rendered prompt exceeds the engine's context window.

    Raised by :func:`generate` when the rendered token sequence is strictly
    longer than the resolved cap — either an explicit ``max_prompt_len`` the
    caller passed in, or the engine's ``max_model_len`` discovered via
    ``GET /v1/models``. Caught client-side before the engine ever sees the
    request, so callers route the failure to a deterministic policy (skip /
    truncate / count) instead of round-tripping through an engine 4xx.

    Named after the corresponding ``verifiers.errors.OverlongPromptError``;
    the two are distinct classes (different package hierarchies) but the
    concept is the same and downstream clients translate one to the other.
    """

    def __init__(self, *, prompt_len: int, max_prompt_len: int) -> None:
        self.prompt_len = prompt_len
        self.max_prompt_len = max_prompt_len
        super().__init__(
            f"Prompt length ({prompt_len}) exceeds maximum context length ({max_prompt_len})."
        )


class MalformedGenerateResponseError(ValueError):
    """The generate endpoint returned unusable sampled-token evidence."""


# Per-process cache of resolved engine context-length caps, keyed by
# ``(base_url, model)``. ``None`` is the "we asked the engine and it didn't
# tell us" sentinel — distinct from "key missing" (haven't asked yet). The
# lock serializes the first lookup per key; cache hits avoid the lock.
_max_prompt_len_cache: dict[tuple[str, str], int | None] = {}
_max_prompt_len_lock = asyncio.Lock()


async def _resolve_max_prompt_len(client: AsyncOpenAI, model: str) -> int | None:
    """Discover ``max_model_len`` from the engine via ``GET /v1/models``.

    OpenAI-API-compatible engines expose model metadata at this endpoint;
    vLLM extends its ``ModelCard`` with a ``max_model_len`` field. Engines
    that don't (SGLang as of this writing, third-party gateways, etc.) get
    a cached ``None`` and the pre-flight overflow check silently disables —
    callers fall back to whatever reactive handling they have for engine
    4xx, which the verifiers ``@handle_openai_overlong_prompt`` decorator
    already supplies for the prime-rl path.

    Any exception during lookup (network error, non-JSON body, attribute
    miss on a mock client in tests) is treated as "unknown cap": cached
    ``None`` so we don't retry on every call.
    """
    key = (str(getattr(client, "base_url", "")), model)
    if key in _max_prompt_len_cache:
        return _max_prompt_len_cache[key]
    async with _max_prompt_len_lock:
        if key in _max_prompt_len_cache:
            return _max_prompt_len_cache[key]
        try:
            payload = await client.get("/models", cast_to=cast(Any, dict[str, Any]))
        except Exception as exc:
            _request_logger.debug("max_prompt_len lookup failed: %s", exc)
            _max_prompt_len_cache[key] = None
            return None
        value: int | None = None
        for card in payload.get("data") or []:
            if not isinstance(card, Mapping):
                continue
            if card.get("id") != model:
                continue
            raw = card.get("max_model_len")
            if isinstance(raw, int) and raw > 0:
                value = raw
            break
        _max_prompt_len_cache[key] = value
        return value


def _strip_base64_field(raw: bytes, prefix: bytes) -> tuple[bytes, memoryview | None]:
    """Splice a large base64 string field out of raw JSON bytes.

    Avoids json-decoding megabytes of base64; the returned memoryview
    references ``raw`` and is re-inserted into the parsed payload.
    """
    data_start = raw.find(prefix)
    if data_start < 0:
        return raw, None

    data_start += len(prefix)
    data_end = raw.index(b'"', data_start)
    data = memoryview(raw)[data_start:data_end]
    stripped = raw[:data_start] + raw[data_end:]
    return stripped, data


def parse_generate_response(raw: bytes) -> dict[str, Any]:
    stripped, routed_data = _strip_base64_field(raw, ROUTED_EXPERTS_DATA_PREFIX)
    stripped, kept_ids_data = _strip_base64_field(stripped, KEPT_TOKENS_IDS_PREFIX)
    payload: dict[str, Any] = json.loads(stripped)
    if routed_data is not None:
        payload["choices"][0]["routed_experts"]["data"] = routed_data
    if kept_ids_data is not None:
        payload["choices"][0]["kept_tokens"]["ids"] = kept_ids_data
    return payload


def _encode_fixed_width_array(
    name: str,
    value: np.ndarray,
    *,
    dtype: np.dtype,
    rank: int,
    columns: int | None = None,
    minimum: int | None = None,
) -> dict[str, Any]:
    expected_dtype = np.dtype(dtype)
    if rank == 1:
        require_1d_array(name, value, dtype=expected_dtype, minimum=minimum)
        if columns is not None:
            raise ValueError(f"{name} rank-1 arrays cannot declare columns")
    elif rank == 2 and columns == 2:
        require_range_array(name, value)
    else:
        raise ValueError(f"{name} has unsupported fixed-width shape metadata")
    require_readonly(name, value)
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    raw: bytes | memoryview = b"" if value.nbytes == 0 else memoryview(value).cast("B")
    envelope: dict[str, Any] = {
        "version": 1,
        "encoding": "base64",
        "dtype": expected_dtype.str,
        "count": value.shape[0],
        "data": base64.b64encode(raw).decode("ascii"),
    }
    if columns is not None:
        envelope["columns"] = columns
    return envelope


def _decode_fixed_width_array(
    name: str,
    value: object,
    *,
    dtype: np.dtype,
    rank: int,
    columns: int | None = None,
    minimum: int | None = None,
) -> np.ndarray:
    expected_dtype = np.dtype(dtype)
    required = {"version", "encoding", "dtype", "count", "data"}
    expected = required | ({"columns"} if columns is not None else set())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MalformedGenerateResponseError(
            f"Engine response {name} has invalid envelope fields."
        )
    if value["version"] != 1 or type(value["version"]) is not int:
        raise MalformedGenerateResponseError(
            f"Engine response {name}.version must be 1."
        )
    if value["encoding"] != "base64":
        raise MalformedGenerateResponseError(
            f"Engine response {name}.encoding must be 'base64'."
        )
    if value["dtype"] != expected_dtype.str:
        raise MalformedGenerateResponseError(
            f"Engine response {name}.dtype must be {expected_dtype.str!r}."
        )
    count = value["count"]
    if type(count) is not int or count < 0:
        raise MalformedGenerateResponseError(
            f"Engine response {name}.count must be non-negative."
        )
    if columns is not None and (
        type(value["columns"]) is not int or value["columns"] != columns
    ):
        raise MalformedGenerateResponseError(
            f"Engine response {name}.columns must be {columns}."
        )
    if rank not in (1, 2) or (rank == 2 and columns is None):
        raise ValueError(f"{name} has unsupported fixed-width shape metadata")
    item_count = count * (columns if columns is not None else 1)
    expected_bytes = item_count * expected_dtype.itemsize
    encoded = value["data"]
    if not isinstance(encoded, str):
        raise MalformedGenerateResponseError(
            f"Engine response {name}.data must be canonical base64."
        )
    try:
        supplied = encoded.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MalformedGenerateResponseError(
            f"Engine response {name}.data must be canonical base64."
        ) from exc
    expected_encoded_bytes = 4 * ((expected_bytes + 2) // 3)
    if len(supplied) != expected_encoded_bytes:
        raise MalformedGenerateResponseError(
            f"Engine response {name}.data length does not match count and dtype."
        )
    remainder = expected_bytes % 3
    padding = 0 if remainder == 0 else 3 - remainder
    if (
        padding
        and (not supplied.endswith(b"=" * padding) or b"=" in supplied[:-padding])
    ) or (not padding and b"=" in supplied):
        raise MalformedGenerateResponseError(
            f"Engine response {name}.data must be canonical base64."
        )
    try:
        raw = base64.b64decode(supplied, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MalformedGenerateResponseError(
            f"Engine response {name}.data must be canonical base64."
        ) from exc
    if len(raw) != expected_bytes:
        raise MalformedGenerateResponseError(
            f"Engine response {name}.data length does not match count and dtype."
        )
    if remainder:
        final_sextet = _BASE64_ALPHABET.find(supplied[-(padding + 1)])
        unused_mask = 0x0F if remainder == 1 else 0x03
        if final_sextet < 0 or final_sextet & unused_mask:
            raise MalformedGenerateResponseError(
                f"Engine response {name}.data must be canonical base64."
            )
    result = np.frombuffer(raw, dtype=expected_dtype)
    if rank == 2:
        assert columns is not None
        result = result.reshape(count, columns)
    if minimum is not None and result.size and np.any(result < minimum):
        raise MalformedGenerateResponseError(
            f"Engine response {name} values must be >= {minimum}."
        )
    result.flags.writeable = False
    return result


def _parse_completion_logprobs(
    choice: Mapping[str, Any], completion_ids: np.ndarray
) -> np.ndarray:
    completion_logprobs = _decode_fixed_width_array(
        "choice.completion_logprobs",
        choice.get("completion_logprobs"),
        dtype=LOGPROBS_DTYPE,
        rank=1,
    )
    if completion_logprobs.size != completion_ids.size:
        raise MalformedGenerateResponseError(
            "Engine response completion token count "
            f"({completion_ids.size}) does not match logprob count ({completion_logprobs.size})."
        )
    if not np.all(np.isfinite(completion_logprobs)):
        raise MalformedGenerateResponseError(
            "Engine response completion_logprobs must be finite."
        )
    if np.any(completion_logprobs == VLLM_LOGPROB_SENTINEL):
        raise MalformedGenerateResponseError(
            "Engine response completion_logprobs does not contain sampling evidence."
        )
    return completion_logprobs


async def generate(
    *,
    client: AsyncOpenAI,
    renderer: Renderer,
    messages: list[Message],
    model: str,
    prompt_ids: np.ndarray | None = None,
    multi_modal_data: MultiModalData | None = None,
    prompt_attribution: RenderedTokens | None = None,
    tools: list[ToolSpec] | None = None,
    sampling_params: dict[str, Any] | None = None,
    cache_salt: str | None = None,
    priority: int | None = None,
    extra_headers: dict[str, str] | None = None,
    max_prompt_len: int | None = None,
    process_multimodal: bool = True,
) -> dict[str, Any]:
    """Tokenize messages, call vLLM /inference/v1/generate, parse the response.

    ``sampling_params`` is forwarded to vLLM verbatim. Two fields are always
    set by us and override caller values: ``stop_token_ids`` (from the
    renderer) and ``logprobs=1`` (we always emit completion_logprobs). Pass
    ``prompt_ids`` to skip rendering and use a prebuilt token sequence —
    pair it with ``multi_modal_data`` when the prebuilt prompt has image /
    video placeholders that need engine-side mm payload, and with
    ``prompt_attribution`` (a :class:`RenderedTokens` whose ``token_ids``
    match the passed-in ``prompt_ids``) to carry the renderer's per-token
    attribution (``is_content`` / ``sampled_mask`` / ``message_indices`` /
    ``message_roles``) into the result without re-rendering.

    For multimodal renderers (e.g. ``Qwen3VLRenderer``), the call goes
    through ``renderer.render(...)`` to recover the ``multi_modal_data``
    sidecar, then serializes it to vLLM's fixed-width ``features`` schema.
    The serializer imports ``vllm.*`` lazily so text-only consumers never
    pay for the import.

    ``max_prompt_len`` controls the pre-flight overflow check. When the
    rendered prompt is strictly longer than the cap, the request is never
    sent and ``OverlongPromptError`` is raised. If ``max_prompt_len`` is
    ``None`` (the default), the cap is auto-discovered once per
    ``(base_url, model)`` via ``GET /v1/models`` (vLLM's
    ``ModelCard.max_model_len`` extension); engines that don't expose it
    cache a ``None`` cap and the pre-flight silently disables. Engine 4xx
    that still slip through propagate raw — converting them into a domain
    error is the calling client's job (its error shape is engine-specific).
    Returns a dict with: request_id, prompt_ids, renderer_prompt_ids,
    mm_placeholders, completion_ids, completion_logprobs, content,
    reasoning_content, tool_calls, finish_reason, routed_experts,
    multi_modal_data, prompt_attribution. ``renderer_prompt_ids`` is reserved
    for a future typed deferred-multimodal transport and is currently ``None``.

    ``prompt_attribution`` is the renderer's :class:`RenderedTokens` for
    the prompt — either the one this call computed via
    ``renderer.render(...)`` or the one the caller threaded in alongside
    ``prompt_ids``. Carries ``token_ids``, ``message_indices``,
    ``sampled_mask``, ``is_content``, ``message_roles``, and
    ``multi_modal_data``, so downstream consumers (verifiers
    ``RendererClient`` → prime-rl) can build per-token loss masks
    (``content_mask_for_roles({"tool"})`` for SFT-on-tool-body,
    ``sampled_mask`` for RL trainable spans) without a second render
    pass. ``None`` when the caller passed pre-built ``prompt_ids``
    without attribution.
    """
    if tools and not getattr(renderer, "supports_tools", True):
        raise ValueError(
            f"{type(renderer).__name__} does not support tools. "
            "Choose a model-specific renderer instead of the default fallback."
        )
    if not process_multimodal:
        raise NotImplementedError(
            "The fixed-width generate protocol does not support deferred multimodal processing."
        )

    def _prepare():
        if prompt_ids is not None:
            # Caller-supplied prompt; if they also gave us pre-computed
            # attribution (e.g. the bridge path in verifiers), thread it
            # through unchanged.
            require_1d_array("prompt_ids", prompt_ids, dtype=TOKEN_IDS_DTYPE, minimum=0)
            require_readonly("prompt_ids", prompt_ids)
            if prompt_attribution is not None and not np.array_equal(
                prompt_attribution.token_ids, prompt_ids
            ):
                raise ValueError("prompt_attribution.token_ids must match prompt_ids")
            return (
                prompt_ids,
                renderer.get_stop_token_ids(),
                multi_modal_data,
                prompt_attribution,
            )
        rendered = renderer.render(
            messages,
            tools=tools,
            add_generation_prompt=True,
        )
        return (
            rendered.token_ids,
            renderer.get_stop_token_ids(),
            rendered.multi_modal_data,
            rendered,
        )

    prompt_ids, stop_token_ids, mm_data, prompt_attr = _prepare()

    if max_prompt_len is None:
        max_prompt_len = await _resolve_max_prompt_len(client, model)
    if max_prompt_len is not None and len(prompt_ids) > max_prompt_len:
        raise OverlongPromptError(
            prompt_len=len(prompt_ids), max_prompt_len=max_prompt_len
        )

    sp: dict[str, Any] = dict(sampling_params or {})
    sp["stop_token_ids"] = stop_token_ids
    sp["logprobs"] = 1
    sp.setdefault("skip_special_tokens", False)

    body: dict[str, Any] = {
        "model": model,
        "token_ids": _encode_fixed_width_array(
            "token_ids", prompt_ids, dtype=TOKEN_IDS_DTYPE, rank=1, minimum=0
        ),
        "sampling_params": sp,
    }
    features = (
        _build_mm_features(renderer, mm_data)
        if mm_data and not mm_data.is_empty()
        else None
    )
    if features is not None:
        body["features"] = features
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    if priority is not None:
        body["priority"] = priority

    # /inference/v1/generate is mounted at the server root, not under /v1
    # like the OpenAI-compatible endpoints. Build an absolute URL so the
    # AsyncOpenAI client doesn't prepend its automatic /v1.
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    endpoint = f"{base}/inference/v1/generate"
    _request_logger.debug(
        "POST %s prompt_len=%d max_tokens=%s",
        endpoint,
        len(prompt_ids),
        sp.get("max_tokens"),
    )
    post_kwargs: dict[str, Any] = {
        "cast_to": httpx.Response,
        "body": body,
    }
    if extra_headers:
        post_kwargs["options"] = cast(Any, {"headers": extra_headers})
    raw_response = await client.post(endpoint, **post_kwargs)
    data = parse_generate_response(raw_response.content)

    choice = (data.get("choices") or [{}])[0]
    completion_ids = _decode_fixed_width_array(
        "choice.token_ids",
        choice.get("token_ids"),
        dtype=TOKEN_IDS_DTYPE,
        rank=1,
        minimum=0,
    )
    raw_effective_prompt_ids = data.get("prompt_token_ids")
    if raw_effective_prompt_ids is None:
        raise MalformedGenerateResponseError(
            "Engine response must include prompt_token_ids."
        )
    effective_prompt_ids = _decode_fixed_width_array(
        "prompt_token_ids",
        raw_effective_prompt_ids,
        dtype=TOKEN_IDS_DTYPE,
        rank=1,
        minimum=0,
    )
    if not np.array_equal(effective_prompt_ids, prompt_ids):
        raise MalformedGenerateResponseError(
            "Engine response prompt_token_ids does not match the requested prompt."
        )
    raw_mm_placeholders = data.get("mm_placeholders")
    mm_placeholders = (
        {
            modality: _decode_fixed_width_array(
                f"mm_placeholders[{modality!r}]",
                envelope,
                dtype=OFFSETS_DTYPE,
                rank=2,
                columns=2,
            )
            for modality, envelope in raw_mm_placeholders.items()
        }
        if isinstance(raw_mm_placeholders, Mapping)
        else None
    )

    completion_logprobs = _parse_completion_logprobs(choice, completion_ids)

    parsed = renderer.parse_response(completion_ids, tools=tools)

    routed_experts = choice.get("routed_experts")
    kept_tokens = choice.get("kept_tokens")

    # /inference/v1/generate returns finish_reason in {"stop","length",...} —
    # never "tool_calls" (a chat-completions concept). Promote stop→tool_calls
    # when we extracted at least one well-formed tool call client-side, so
    # OpenAI-compatible agent loops continue past the tool turn instead of
    # treating the response as final. Malformed attempts (INVALID_JSON,
    # UNCLOSED_BLOCK, ...) don't qualify — those still surface on
    # ``parsed.tool_calls`` so verifiers can inspect them, but they don't
    # trigger the tool-loop continuation.
    finish_reason = choice.get("finish_reason")
    ok_tool_calls = [
        tc for tc in parsed.tool_calls if tc.status == ToolCallParseStatus.OK
    ]
    if ok_tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    return {
        "request_id": data.get("request_id") or "",
        "prompt_ids": effective_prompt_ids,
        "renderer_prompt_ids": None,
        "mm_placeholders": mm_placeholders,
        "completion_ids": completion_ids,
        "completion_logprobs": completion_logprobs,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": finish_reason,
        "routed_experts": routed_experts,
        "kept_tokens": kept_tokens,
        # The mm sidecar consumed on the request side, surfaced back so
        # callers can persist it on the trajectory step for downstream
        # multi-turn bridging and training-sample construction.
        "multi_modal_data": mm_data,
        # The renderer's per-token attribution for the prompt — either
        # the RenderedTokens computed here via renderer.render(...) or
        # the one threaded in by the caller alongside prompt_ids (the
        # bridge path). Lets downstream consumers (verifiers
        # RendererClient → prime-rl) build SFT-on-tool-body and other
        # selective loss masks without a second render pass. ``None``
        # when the caller passed prompt_ids without attribution.
        "prompt_attribution": prompt_attr,
    }


def _build_mm_features(
    renderer: Renderer,
    mm_data: MultiModalData,
) -> dict[str, Any] | None:
    """Serialize ``MultiModalData`` to vLLM's ``/inference/v1/generate`` features payload.

    vLLM's ``MultiModalFeatures`` carries three things: hashes (for cache
    lookup), placeholder positions (so the engine knows where in the
    token stream each item lives), and per-item ``MultiModalKwargsItem``
    base64-encoded. The encoding requires vLLM-side type info — what
    fields belong to each modality, how they batch — and is currently
    model-family specific. For now we dispatch on the renderer class;
    extend the dispatch table as more multimodal renderers land.

    NOTE — future engine pluggability: this encoder is vLLM-specific
    (uses ``vllm.multimodal.inputs.MultiModalKwargsItems``,
    ``vllm.entrypoints.scale_out.token_in_token_out.mm_serde.encode_mm_kwargs_item``, and
    ``_create_qwen2vl_field_factory``). When a second inference engine
    arrives (SGLang, MAX, ...) the renderer client should be parameterized
    on engine: either (a) move the encoder onto the renderer as
    ``encode_mm_for_<engine>(mm_data)`` methods, or (b) accept an
    ``Encoder`` strategy at the ``generate(...)`` call site. The data type
    (``MultiModalData``) is already framework-agnostic and does not need
    to change. Don't pre-build the abstraction with one engine in tree.
    """
    from renderers.gemma4 import Gemma4Renderer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from renderers.qwen35 import Qwen35Renderer

    renderer_cls = type(renderer)

    # Qwen3-VL and Qwen3.5 both ship ``pixel_values`` + ``image_grid_thw``
    # via the shared Qwen2-VL field factory. ``spatial_merge_size=2`` is
    # the family default and matches every Qwen-VL processor in tree.
    if issubclass(renderer_cls, (Qwen3VLRenderer, Qwen35Renderer)):
        return _build_qwen_vl_features(mm_data, spatial_merge_size=2)
    if issubclass(renderer_cls, Gemma4Renderer):
        return _build_gemma4_features(mm_data)

    raise NotImplementedError(
        f"Multimodal serialization not implemented for {renderer_cls.__name__}. "
        "Add a dispatch branch in renderers.client._build_mm_features."
    )


def _build_gemma4_features(mm_data: MultiModalData) -> dict[str, Any]:
    """vLLM features payload for Gemma 4 image inputs.

    Hugging Face names the position field ``image_position_ids`` while
    vLLM's Gemma 4 processor schema calls it ``pixel_position_ids``. Keep
    renderer output faithful to the HF processor and translate at this
    engine-specific boundary.
    """
    _require_transformers("Encoding Gemma 4 multimodal features for vLLM")
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.serve.disagg.mm_serde import (
            encode_mm_kwargs_item,
        )
        from vllm.multimodal.inputs import (
            MultiModalFieldConfig,
            MultiModalKwargsItems,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Gemma 4 multimodal generate via /inference/v1/generate requires "
            "a vLLM release with Gemma 4 support and `torch`."
        ) from exc

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }
    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        pixel_values = torch.cat(
            [torch.as_tensor(item["pixel_values"]) for item in image_items], dim=0
        )
        pixel_position_ids = torch.cat(
            [torch.as_tensor(item["image_position_ids"]) for item in image_items],
            dim=0,
        )
        hf_inputs = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
            }
        )
        field_config = {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "pixel_position_ids": MultiModalFieldConfig.batched("image"),
        }
        kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, field_config)
        out["kwargs_data"]["image"] = [
            encode_mm_kwargs_item(item) for item in kwargs_items["image"]
        ]
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        placeholders = mm_data.mm_placeholders.get("image")
        if placeholders is not None:
            out["mm_placeholders"]["image"] = _encode_fixed_width_array(
                "mm_placeholders['image']",
                placeholders,
                dtype=OFFSETS_DTYPE,
                rank=2,
                columns=2,
            )

    if not any(out["kwargs_data"].values()):
        out["kwargs_data"] = None
    return out


def _build_qwen_vl_features(
    mm_data: MultiModalData, *, spatial_merge_size: int
) -> dict[str, Any]:
    """vLLM features payload for the Qwen-VL family (Qwen2-VL / Qwen3-VL).

    Stacks per-image processor outputs back into a batched ``BatchFeature``,
    runs the Qwen2-VL field factory (shared across the family), wraps as
    ``MultiModalKwargsItems``, base64-encodes each item, and assembles a
    JSON-serializable dict matching vLLM's ``MultiModalFeatures`` schema.

    Returns ``None`` semantics live one level up — this helper assumes
    the caller already verified ``mm_data`` is non-empty.
    """
    _require_transformers("Encoding Qwen-VL multimodal features for vLLM")
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.serve.disagg.mm_serde import (
            encode_mm_kwargs_item,
        )
        from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
        from vllm.multimodal.inputs import MultiModalKwargsItems
    except ImportError as exc:
        raise RuntimeError(
            "Multimodal generate via /inference/v1/generate requires `vllm` "
            "and `torch` to encode the features payload. Install vLLM in this "
            "environment, or pre-build features upstream."
        ) from exc

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }

    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        # mm_items now ship numpy arrays (the renderer is torch-free);
        # convert at this vLLM-glue boundary where torch is already a
        # hard dependency.
        pixel_values = torch.cat(
            [torch.as_tensor(it["pixel_values"]) for it in image_items], dim=0
        )
        image_grid_thw = torch.cat(
            [torch.as_tensor(it["image_grid_thw"]) for it in image_items], dim=0
        )
        hf_inputs = BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
        )
        config = _create_qwen2vl_field_factory(spatial_merge_size)(hf_inputs)
        kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config)
        encoded = [encode_mm_kwargs_item(it) for it in kwargs_items["image"]]
        out["kwargs_data"]["image"] = encoded
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        placeholders = mm_data.mm_placeholders.get("image")
        if placeholders is not None:
            out["mm_placeholders"]["image"] = _encode_fixed_width_array(
                "mm_placeholders['image']",
                placeholders,
                dtype=OFFSETS_DTYPE,
                rank=2,
                columns=2,
            )

    # If kwargs_data is empty across all modalities, drop the key so vLLM
    # falls back to the hash-only (cache-hit) path. Otherwise hand it the
    # full payload.
    if not any(out["kwargs_data"].values()):
        out["kwargs_data"] = None

    return out
