"""Renderer-based generate client for vLLM 0.20's /inference/v1/generate.

    messages → Renderer.render_ids() → token IDs → POST /inference/v1/generate
    → completion tokens → Renderer.parse_response() → structured message

When a RendererPool is passed instead of a single Renderer, the sync tokenization
and parsing work is offloaded to threads for parallel execution across rollouts.
HuggingFace fast tokenizers release the GIL during Rust encoding, so threads
achieve real parallelism.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from openai import AsyncOpenAI

from renderers.base import (
    Message,
    MultiModalData,
    Renderer,
    RendererPool,
    ToolCallParseStatus,
    ToolSpec,
)

_request_logger = logging.getLogger("renderers.client")


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
            f"Prompt length ({prompt_len}) exceeds maximum "
            f"context length ({max_prompt_len})."
        )


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


async def _maybe_offload(renderer: Renderer | RendererPool, fn):
    """Run sync renderer work on a thread iff ``renderer`` is a pool.

    A pool's methods can block on its internal queue/lock (size>1 / size=1
    fast path respectively), so we ``asyncio.to_thread`` to avoid stalling
    the event loop. A bare ``Renderer`` runs inline — used in tests where
    event-loop responsiveness isn't a concern and the thread hop would
    be pure overhead.
    """
    if isinstance(renderer, RendererPool):
        return await asyncio.to_thread(fn)
    return fn()


async def generate(
    *,
    client: AsyncOpenAI,
    renderer: Renderer | RendererPool,
    messages: list[Message],
    model: str,
    prompt_ids: list[int] | None = None,
    multi_modal_data: MultiModalData | None = None,
    tools: list[ToolSpec] | None = None,
    sampling_params: dict[str, Any] | None = None,
    cache_salt: str | None = None,
    priority: int | None = None,
    extra_headers: dict[str, str] | None = None,
    max_prompt_len: int | None = None,
) -> dict[str, Any]:
    """Tokenize messages, call vLLM /inference/v1/generate, parse the response.

    ``sampling_params`` is forwarded to vLLM verbatim. Two fields are always
    set by us and override caller values: ``stop_token_ids`` (from the
    renderer) and ``logprobs=1`` (we always emit completion_logprobs). Pass
    ``prompt_ids`` to skip rendering and use a prebuilt token sequence —
    pair it with ``multi_modal_data`` when the prebuilt prompt has image /
    video placeholders that need engine-side mm payload.

    For multimodal renderers (e.g. ``Qwen3VLRenderer``), the call goes
    through ``renderer.render(...)`` to recover the ``multi_modal_data``
    sidecar, then serializes it to vLLM's ``features`` schema (mm_hashes,
    mm_placeholders, kwargs_data) before POSTing. The serializer imports
    ``vllm.*`` lazily so text-only consumers never pay for the import.

    ``max_prompt_len`` controls the pre-flight overflow check. When the
    rendered prompt is strictly longer than the cap, the request is never
    sent and ``OverlongPromptError`` is raised. If ``max_prompt_len`` is
    ``None`` (the default), the cap is auto-discovered once per
    ``(base_url, model)`` via ``GET /v1/models`` (vLLM's
    ``ModelCard.max_model_len`` extension); engines that don't expose it
    cache a ``None`` cap and the pre-flight silently disables. Engine 4xx
    that still slip through propagate raw — converting them into a domain
    error is the calling client's job (its error shape is engine-specific).

    Returns a dict with: request_id, prompt_ids, completion_ids,
    completion_logprobs, content, reasoning_content, tool_calls,
    finish_reason, routed_experts.
    """
    if tools and not getattr(renderer, "supports_tools", True):
        raise ValueError(
            f"{type(renderer).__name__} does not support tools. "
            "Choose a model-specific renderer instead of the default fallback."
        )

    def _prepare():
        if prompt_ids is not None:
            return list(prompt_ids), renderer.get_stop_token_ids(), multi_modal_data
        rendered = renderer.render(messages, tools=tools, add_generation_prompt=True)
        return (
            rendered.token_ids,
            renderer.get_stop_token_ids(),
            rendered.multi_modal_data,
        )

    prompt_ids, stop_token_ids, mm_data = await _maybe_offload(renderer, _prepare)

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
        "token_ids": prompt_ids,
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
        "cast_to": cast(Any, dict[str, Any]),
        "body": body,
    }
    if extra_headers:
        post_kwargs["options"] = cast(Any, {"headers": extra_headers})
    data = await client.post(endpoint, **post_kwargs)

    choice = (data.get("choices") or [{}])[0]
    completion_ids = choice.get("token_ids") or []

    parsed = await _maybe_offload(
        renderer, lambda: renderer.parse_response(completion_ids, tools=tools)
    )

    # ChatCompletionLogProbs flatten: {"content": [{"logprob": ...}, ...]}
    raw_logprobs = choice.get("logprobs") or {}
    content_lp = raw_logprobs.get("content") if isinstance(raw_logprobs, dict) else None
    completion_logprobs = [float(c.get("logprob") or 0.0) for c in content_lp or []]

    routed_experts = None
    raw_re = choice.get("routed_experts")
    if isinstance(raw_re, dict) and "data" in raw_re and "shape" in raw_re:
        routed_experts = (
            np.frombuffer(base64.b85decode(raw_re["data"]), dtype=np.int32)
            .reshape(raw_re["shape"])
            .tolist()
        )

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
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(completion_ids),
        "completion_logprobs": completion_logprobs,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": finish_reason,
        "routed_experts": routed_experts,
        # The mm sidecar consumed on the request side, surfaced back so
        # callers can persist it on the trajectory step for downstream
        # multi-turn bridging and training-sample construction.
        "multi_modal_data": mm_data,
    }


def _build_mm_features(
    renderer: Renderer | RendererPool,
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

    NOTE — future engine pluggability: this encoder is vLLM 0.20-specific
    (uses ``vllm.multimodal.inputs.MultiModalKwargsItems``,
    ``vllm.entrypoints.serve.disagg.mm_serde.encode_mm_kwargs_item``, and
    ``_create_qwen2vl_field_factory``). When a second inference engine
    arrives (SGLang, MAX, ...) the renderer client should be parameterized
    on engine: either (a) move the encoder onto the renderer as
    ``encode_mm_for_<engine>(mm_data)`` methods, or (b) accept an
    ``Encoder`` strategy at the ``generate(...)`` call site. The data type
    (``MultiModalData``) is already framework-agnostic and does not need
    to change. Don't pre-build the abstraction with one engine in tree.
    """
    from renderers.qwen3_vl import Qwen3VLRenderer
    from renderers.qwen35 import Qwen35Renderer

    # Type dispatch only needs the renderer class. Pools expose
    # ``renderer_cls`` as a snapshot attribute, so we don't have to check
    # out a slot just to read ``type(r)``.
    renderer_cls = (
        renderer.renderer_cls if isinstance(renderer, RendererPool) else type(renderer)
    )

    # Qwen3-VL and Qwen3.5 both ship ``pixel_values`` + ``image_grid_thw``
    # via the shared Qwen2-VL field factory. ``spatial_merge_size=2`` is
    # the family default and matches every Qwen-VL processor in tree.
    if issubclass(renderer_cls, (Qwen3VLRenderer, Qwen35Renderer)):
        return _build_qwen_vl_features(mm_data, spatial_merge_size=2)

    raise NotImplementedError(
        f"Multimodal serialization not implemented for {renderer_cls.__name__}. "
        "Add a dispatch branch in renderers.client._build_mm_features."
    )


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
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.serve.disagg.mm_serde import encode_mm_kwargs_item
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
        out["mm_placeholders"]["image"] = [
            {"offset": p.offset, "length": p.length}
            for p in mm_data.mm_placeholders.get("image") or []
        ]

    # If kwargs_data is empty across all modalities, drop the key so vLLM
    # falls back to the hash-only (cache-hit) path. Otherwise hand it the
    # full payload.
    if not any(out["kwargs_data"].values()):
        out["kwargs_data"] = None

    return out
