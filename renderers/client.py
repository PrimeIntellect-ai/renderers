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
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import httpx
from openai import AsyncOpenAI

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

    For multimodal renderers, the call goes
    through ``renderer.render(...)`` to recover the ``multi_modal_data``
    sidecar, then serializes it to vLLM's ``features`` schema (mm_hashes,
    mm_placeholders, kwargs_data) before POSTing. Raw image ``kwargs_data``
    slots always carry a descriptor ref — every image (current and prior
    turns) is sent as a pointer that the inference endpoint materializes.

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
            # attribution (e.g. the bridge path in verifiers), thread it through.
            prompt_mm_data = multi_modal_data
            if prompt_mm_data is None and prompt_attribution is not None:
                prompt_mm_data = prompt_attribution.multi_modal_data
            return (
                list(prompt_ids),
                renderer.get_stop_token_ids(),
                prompt_mm_data,
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

    # Every image slot carries its raw ref (the pointer) — prior-turn images
    # ride forward unchanged, and the inference endpoint materializes them.
    if mm_data is not None and not mm_data.is_empty():
        body["features"] = await _maybe_offload(
            renderer, lambda: _build_vllm_mm_features(mm_data)
        )
        if prompt_attr is not None and prompt_attr.multi_modal_data is not mm_data:
            prompt_attr = replace(prompt_attr, multi_modal_data=mm_data)
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


def _build_vllm_mm_features(mm_data: MultiModalData) -> dict[str, Any]:
    """Serialize ``MultiModalData`` to vLLM's ``/inference/v1/generate`` features payload.

    vLLM's ``MultiModalFeatures`` carries three things: hashes, placeholder
    positions (so the engine knows where in the token stream each item lives),
    and one raw ref per item. Raw multimodal descriptors use the common envelope
    emitted by renderers; family-specific geometry stays inside the descriptor
    payload and is interpreted downstream by prime-rl/vLLM adapters.
    """
    from renderers.mm_store import (
        RAW_MM_ITEM_KIND,
        raw_mm_ref,
    )

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }

    for source_modality, items in mm_data.mm_items.items():
        if not items:
            continue
        mm_hashes = list(mm_data.mm_hashes.get(source_modality) or [])
        placeholders = list(mm_data.mm_placeholders.get(source_modality) or [])
        if len(mm_hashes) != len(items) or len(placeholders) != len(items):
            raise ValueError(
                "Multimodal sidecar length mismatch: "
                f"modality={source_modality} items={len(items)} "
                f"hashes={len(mm_hashes)} placeholders={len(placeholders)}"
            )

        for idx, item in enumerate(items):
            if item.get("kind") != RAW_MM_ITEM_KIND:
                raise NotImplementedError(
                    "renderers.client.generate() requires raw multimodal "
                    "descriptor envelopes (multimodal_output='raw'); "
                    f"got item keys {sorted(item)} for modality {source_modality!r}."
                )
            feature_modality = item.get("vllm_modality") or source_modality
            if not isinstance(feature_modality, str) or not feature_modality:
                raise ValueError("raw multimodal item has invalid vllm_modality")

            raw_image_uri = item.get("raw_image_uri")
            family = item.get("family")
            fingerprint = item.get("layout_fingerprint")
            payload = item.get("payload")
            if not isinstance(raw_image_uri, str) or not raw_image_uri:
                raise ValueError("raw multimodal item is missing raw_image_uri")
            if not isinstance(family, str) or not family:
                raise ValueError("raw multimodal item is missing family")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("raw multimodal item is missing layout_fingerprint")
            if not isinstance(payload, dict):
                raise ValueError("raw multimodal item payload must be a dict")

            out["mm_hashes"].setdefault(feature_modality, []).append(mm_hashes[idx])
            out["mm_placeholders"].setdefault(feature_modality, []).append(
                {"offset": placeholders[idx].offset, "length": placeholders[idx].length}
            )
            out["kwargs_data"].setdefault(feature_modality, []).append(
                raw_mm_ref(
                    family=family,
                    fingerprint=fingerprint,
                    modality=feature_modality,
                    mm_hash=mm_hashes[idx],
                    raw_image_uri=raw_image_uri,
                    payload=payload,
                )
            )

    return out
