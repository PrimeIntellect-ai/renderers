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
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, OpenAIError

from renderers.base import (
    Message,
    MultiModalData,
    RenderedTokens,
    Renderer,
    RendererPool,
    ToolCallParseStatus,
    ToolSpec,
)

_request_logger = logging.getLogger("renderers.client")

# --- multimodal memory modes -------------------------------------------------
# Two independent, opt-in modes shrink the multimodal data the rollout path keeps
# and ships. Both default OFF so behaviour is unchanged unless a consumer opts in.
#
# EPHEMERAL (``RENDERERS_MM_EPHEMERAL``) — *stored* data. ``generate`` returns a
# descriptor-only ``multi_modal_data`` (image_grid_thw + mm_hashes +
# mm_placeholders, no ``pixel_values``), so the env worker / trace never retains
# decoded image tensors for the life of a rollout — resident stored mm drops from
# the processed-tensor size to a tiny descriptor (the source image still lives in
# the message). The consumer re-derives pixels downstream (training-sample
# construction) from the message images. Safe on any engine (purely client-side).
#
# HASH-ONLY (``RENDERERS_MM_HASH_CACHE``) — *transported* data. Send each distinct
# image's ``pixel_values`` once; later turns send a descriptor-only (``None``
# kwargs) slot and the engine serves it from its mm-hash cache, so the POST body
# carries only new images. REQUIRES an engine that resolves ``None`` slots from
# cache (the disagg router topology). A plain single-server vLLM 0.22 forces
# ``skip_mm_cache=True`` and crashes on an unresolved ``None`` — hence default OFF.
def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").lower() not in ("0", "false", "off", "no", "")


_MM_EPHEMERAL = _env_on("RENDERERS_MM_EPHEMERAL")
_MM_HASH_CACHE_ENABLED = _env_on("RENDERERS_MM_HASH_CACHE")
# Per-(endpoint, model) FIFO bound on remembered sent hashes. Should comfortably
# exceed the engine's mm cache so we don't claim cache hits the engine evicted;
# a stale claim is corrected by the cache-miss fallback, never a wrong result.
_MM_SENT_HASHES_MAX = int(os.environ.get("RENDERERS_MM_SENT_HASHES_MAX", "16384"))
_mm_sent: dict[tuple[str, str], "OrderedDict[str, None]"] = {}
_mm_sent_lock = threading.Lock()


def _mm_seen(key: tuple[str, str], hashes: list[str]) -> set[str]:
    """Subset of ``hashes`` already sent under ``key`` (believed engine-cached)."""
    with _mm_sent_lock:
        lru = _mm_sent.get(key)
        if not lru:
            return set()
        seen = set()
        for h in hashes:
            if h in lru:
                seen.add(h)
                lru.move_to_end(h)
        return seen


def _mm_record(key: tuple[str, str], hashes: list[str]) -> None:
    """Record ``hashes`` as sent-in-full under ``key`` (FIFO-bounded)."""
    if not hashes:
        return
    with _mm_sent_lock:
        lru = _mm_sent.setdefault(key, OrderedDict())
        for h in hashes:
            lru[h] = None
            lru.move_to_end(h)
        while len(lru) > _MM_SENT_HASHES_MAX:
            lru.popitem(last=False)


def _mm_forget(key: tuple[str, str], hashes: list[str]) -> None:
    """Drop ``hashes`` from the sent set (after a suspected engine cache miss)."""
    with _mm_sent_lock:
        lru = _mm_sent.get(key)
        if lru:
            for h in hashes:
                lru.pop(h, None)


def _strip_pixels_for(mm_data: MultiModalData, hashes: set[str]) -> MultiModalData:
    """Return ``mm_data`` with ``pixel_values`` dropped from image items whose
    hash is in ``hashes`` (descriptor — ``image_grid_thw`` etc. — retained, so
    token alignment and a later re-materialize still work)."""
    if not hashes or not mm_data.mm_items:
        return mm_data
    new_items: dict[str, list[dict[str, Any]]] = {}
    for modality, items in mm_data.mm_items.items():
        item_hashes = mm_data.mm_hashes.get(modality) or []
        rebuilt = []
        for i, item in enumerate(items):
            if i < len(item_hashes) and item_hashes[i] in hashes:
                rebuilt.append({k: v for k, v in item.items() if k != "pixel_values"})
            else:
                rebuilt.append(item)
        new_items[modality] = rebuilt
    return replace(mm_data, mm_items=new_items)
ROUTED_EXPERTS_DATA_PREFIX = b'"routed_experts":{"data":"'


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


def strip_routed_experts_data(raw: bytes) -> tuple[bytes, memoryview | None]:
    data_start = raw.find(ROUTED_EXPERTS_DATA_PREFIX)
    if data_start < 0:
        return raw, None

    data_start += len(ROUTED_EXPERTS_DATA_PREFIX)
    data_end = raw.index(b'"', data_start)
    routed_data = memoryview(raw)[data_start:data_end]
    stripped = raw[:data_start] + raw[data_end:]
    return stripped, routed_data


def parse_generate_response(raw: bytes) -> dict[str, Any]:
    stripped, routed_data = strip_routed_experts_data(raw)
    payload: dict[str, Any] = json.loads(stripped)
    if routed_data is not None:
        payload["choices"][0]["routed_experts"]["data"] = routed_data
    return payload


async def generate(
    *,
    client: AsyncOpenAI,
    renderer: Renderer | RendererPool,
    messages: list[Message],
    model: str,
    prompt_ids: list[int] | None = None,
    multi_modal_data: MultiModalData | None = None,
    prompt_attribution: RenderedTokens | None = None,
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
    video placeholders that need engine-side mm payload, and with
    ``prompt_attribution`` (a :class:`RenderedTokens` whose ``token_ids``
    match the passed-in ``prompt_ids``) to carry the renderer's per-token
    attribution (``is_content`` / ``sampled_mask`` / ``message_indices`` /
    ``message_roles``) into the result without re-rendering.

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
    finish_reason, routed_experts, multi_modal_data, prompt_attribution.

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

    def _prepare():
        if prompt_ids is not None:
            # Caller-supplied prompt; if they also gave us pre-computed
            # attribution (e.g. the bridge path in verifiers), thread it
            # through unchanged.
            return (
                list(prompt_ids),
                renderer.get_stop_token_ids(),
                multi_modal_data,
                prompt_attribution,
            )
        rendered = renderer.render(messages, tools=tools, add_generation_prompt=True)
        return (
            rendered.token_ids,
            renderer.get_stop_token_ids(),
            rendered.multi_modal_data,
            rendered,
        )

    prompt_ids, stop_token_ids, mm_data, prompt_attr = await _maybe_offload(
        renderer, _prepare
    )

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

    mm_key = (endpoint, model)
    img_hashes: list[str] = (
        list(mm_data.mm_hashes.get("image") or [])
        if mm_data is not None and not mm_data.is_empty()
        else []
    )

    # Hash-only SEND (opt-in, ``RENDERERS_MM_HASH_CACHE``): strip already-sent
    # images to descriptors so the engine serves them from its mm-hash cache and
    # only new images carry pixels. Requires an engine that resolves ``None``
    # kwargs slots from cache (the disagg router topology); a plain single-server
    # vLLM crashes on an unresolved ``None``, so this defaults OFF.
    hash_only = False
    sent_full = list(img_hashes)  # hashes shipped with full pixels this turn
    if img_hashes and _MM_HASH_CACHE_ENABLED:
        seen = _mm_seen(mm_key, img_hashes)
        if seen:
            mm_data = _strip_pixels_for(mm_data, seen)
            hash_only = True
            sent_full = [h for h in img_hashes if h not in seen]

    def _post(send_mm: MultiModalData | None):
        b = dict(body)
        feats = (
            _build_mm_features(renderer, send_mm)
            if send_mm is not None and not send_mm.is_empty()
            else None
        )
        if feats is not None:
            b["features"] = feats
        pk: dict[str, Any] = {"cast_to": httpx.Response, "body": b}
        if extra_headers:
            pk["options"] = cast(Any, {"headers": extra_headers})
        return client.post(endpoint, **pk)

    try:
        raw_response = await _post(mm_data)
        miss = hash_only and raw_response.status_code >= 400
    except OpenAIError:
        if not hash_only:
            raise
        miss = True
    if miss:
        # Suspected engine mm-cache miss for a hash-only image: forget the claim,
        # re-render the full prompt, and resend every image in full.
        _mm_forget(mm_key, img_hashes)
        _, _, mm_data, prompt_attr = await _maybe_offload(renderer, _prepare)
        raw_response = await _post(mm_data)
        sent_full = list(img_hashes)
    if _MM_HASH_CACHE_ENABLED and sent_full:
        # Remember the images we shipped in full so later turns send them
        # hash-only (engine cache hit).
        _mm_record(mm_key, sent_full)

    # Ephemeral RETURN (opt-in, ``RENDERERS_MM_EPHEMERAL``): never hand decoded
    # pixels back to the caller — the trajectory keeps a descriptor only, so the
    # env worker / trace never retains image tensors for the life of a rollout.
    # Pixels are re-derived downstream from the message images (training-sample
    # construction). Independent of the send mode above.
    if img_hashes and _MM_EPHEMERAL:
        mm_data = _strip_pixels_for(mm_data, set(img_hashes))
        if prompt_attr is not None and getattr(prompt_attr, "multi_modal_data", None) is not None:
            prompt_attr = replace(prompt_attr, multi_modal_data=mm_data)
    data = parse_generate_response(raw_response.content)

    choice = (data.get("choices") or [{}])[0]
    completion_ids = choice.get("token_ids") or []

    parsed = await _maybe_offload(
        renderer, lambda: renderer.parse_response(completion_ids, tools=tools)
    )

    # ChatCompletionLogProbs flatten: {"content": [{"logprob": ...}, ...]}
    raw_logprobs = choice.get("logprobs") or {}
    content_lp = raw_logprobs.get("content") if isinstance(raw_logprobs, dict) else None
    completion_logprobs = [float(c.get("logprob") or 0.0) for c in content_lp or []]

    routed_experts = choice.get("routed_experts")

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
        # Per-item hash-only/full: items carrying ``pixel_values`` are encoded as
        # full payloads; items stripped to a descriptor (no ``pixel_values``) get
        # a ``None`` slot, scattered back to their original positions so
        # ``kwargs_data`` stays aligned with ``mm_hashes`` / ``mm_placeholders``.
        # vLLM serves the ``None`` slots from its mm-hash cache. mm_items now ship
        # numpy arrays (the renderer is torch-free); convert at this vLLM-glue
        # boundary where torch is already a hard dependency.
        encoded: list[Any] = [None] * len(image_items)
        full_idx = [
            i for i, it in enumerate(image_items) if it.get("pixel_values") is not None
        ]
        if full_idx:
            full_items = [image_items[i] for i in full_idx]
            pixel_values = torch.cat(
                [torch.as_tensor(it["pixel_values"]) for it in full_items], dim=0
            )
            image_grid_thw = torch.cat(
                [torch.as_tensor(it["image_grid_thw"]) for it in full_items], dim=0
            )
            hf_inputs = BatchFeature(
                data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
            )
            config = _create_qwen2vl_field_factory(spatial_merge_size)(hf_inputs)
            kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config)
            for slot, item in zip(full_idx, kwargs_items["image"]):
                encoded[slot] = encode_mm_kwargs_item(item)
        out["kwargs_data"]["image"] = encoded
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        out["mm_placeholders"]["image"] = [
            {"offset": p.offset, "length": p.length}
            for p in mm_data.mm_placeholders.get("image") or []
        ]

    # If no full payload was built across any modality, drop kwargs_data so vLLM
    # takes the hash-only (cache-hit) path for the whole request. Otherwise hand
    # it the payload (with ``None`` slots for the hash-only images).
    if not any(any(item is not None for item in items) for items in out["kwargs_data"].values()):
        out["kwargs_data"] = None

    return out
