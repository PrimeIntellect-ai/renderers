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
import math
from collections.abc import Mapping
from itertools import islice
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

_request_logger = logging.getLogger("renderers.client")
ROUTED_EXPERTS_DATA_PREFIX = b'"routed_experts":{"data":"'
# vLLM uses this value both when sampled-token evidence is missing and as a
# lower-bound clamp, so receiving it cannot prove the real logprob was returned.
VLLM_LOGPROB_SENTINEL = -9999.0


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


def _strip_base64_fields(raw: bytes, prefix: bytes) -> tuple[bytes, list[memoryview]]:
    """Splice a large base64 string field out of raw JSON bytes.

    Avoids json-decoding megabytes of base64; the returned memoryview
    references ``raw`` and is re-inserted into the parsed payload.
    """
    chunks = []
    fields = []
    cursor = 0
    while (start := raw.find(prefix, cursor)) >= 0:
        start += len(prefix)
        end = raw.index(b'"', start)
        chunks.append(raw[cursor:start])
        fields.append(memoryview(raw)[start:end])
        cursor = end
    if not fields:
        return raw, fields
    chunks.append(raw[cursor:])
    return b"".join(chunks), fields


def parse_generate_response(
    raw: bytes, *, compact_logprobs: bool = False
) -> dict[str, Any]:
    stripped, routed_data = _strip_base64_fields(raw, ROUTED_EXPERTS_DATA_PREFIX)
    compact_data = []
    if compact_logprobs:
        stripped, compact_data = _strip_base64_fields(
            stripped, b'"compact_logprobs":{"data":"'
        )
    payload: dict[str, Any] = json.loads(stripped)
    for field, buffers in (
        ("routed_experts", routed_data),
        ("compact_logprobs", compact_data),
    ):
        if buffers:
            pending = iter(buffers)
            for choice in payload["choices"]:
                if (
                    isinstance(choice.get(field), dict)
                    and choice[field].get("data") == ""
                ):
                    choice[field]["data"] = next(pending)
    return payload


def _parse_compact_logprobs(
    choice: Mapping[str, Any], completion_ids: list[int], k: int | None
) -> tuple[list[float], list[list[int]] | None, list[list[float]] | None]:
    packed = choice.get("compact_logprobs")
    if not isinstance(packed, Mapping) or packed.get("format") != "u32-f32-le-v1":
        raise MalformedGenerateResponseError("Unsupported compact logprob format")
    offsets = packed.get("offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != len(completion_ids) + 1
        or any(type(x) is not int or x < 0 for x in offsets)
        or offsets[0] != 0
        or any(end <= start for start, end in zip(offsets, offsets[1:]))
    ):
        raise MalformedGenerateResponseError("Invalid compact logprob offsets")
    num_top = packed.get("num_top_logprobs")
    if type(num_top) is not int or num_top < -1:
        raise MalformedGenerateResponseError("Invalid compact logprob candidate count")
    try:
        raw = base64.b64decode(packed["data"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise MalformedGenerateResponseError("Invalid compact logprob base64") from exc
    count = offsets[-1]
    if len(raw) != count * 8:
        raise MalformedGenerateResponseError("Invalid compact logprob buffer length")
    ids = np.frombuffer(raw, dtype="<u4", count=count)
    values = np.frombuffer(raw, dtype="<f4", count=count, offset=count * 4)
    if not np.isfinite(values).all() or (values > 1e-6).any():
        raise MalformedGenerateResponseError("Invalid compact logprob values")
    sampled = []
    top_ids: list[list[int]] = []
    top_logprobs: list[list[float]] = []
    sampling_mask = (
        _parse_sampling_mask(choice, completion_ids) if k is not None else None
    )
    for index, (token_id, start, end) in enumerate(
        zip(completion_ids, offsets, offsets[1:])
    ):
        # FlatLogprobs can repeat the sampled ID. Match vLLM's dict order/value semantics.
        row = dict(zip(ids[start:end].tolist(), values[start:end].tolist()))
        if token_id not in row or row[token_id] == VLLM_LOGPROB_SENTINEL:
            raise MalformedGenerateResponseError(
                "Missing compact sampled-token evidence"
            )
        sampled.append(row[token_id])
        if k is not None:
            head = dict(islice(row.items(), max(num_top, 1)))
            selected_ids, selected_logprobs = _select_sampler_head(
                head,
                token_id,
                row[token_id],
                k,
                sampling_mask[index] if sampling_mask is not None else None,
            )
            top_ids.append(selected_ids)
            top_logprobs.append(selected_logprobs)
    return (
        sampled,
        top_ids if k is not None else None,
        top_logprobs if k is not None else None,
    )


def _parse_completion_logprobs(
    choice: Mapping[str, Any], completion_ids: list[int]
) -> list[float]:
    raw_logprobs = choice.get("logprobs")
    if not isinstance(raw_logprobs, Mapping):
        raise MalformedGenerateResponseError(
            "Engine response choice.logprobs must be an object."
        )

    content = raw_logprobs.get("content")
    if not isinstance(content, list):
        raise MalformedGenerateResponseError(
            "Engine response choice.logprobs.content must be a list."
        )
    if len(content) != len(completion_ids):
        raise MalformedGenerateResponseError(
            "Engine response completion token count "
            f"({len(completion_ids)}) does not match logprob count ({len(content)})."
        )

    completion_logprobs: list[float] = []
    for index, entry in enumerate(content):
        if not isinstance(entry, Mapping):
            raise MalformedGenerateResponseError(
                f"Engine response choice.logprobs.content[{index}] must be an object."
            )
        expected_token = f"token_id:{completion_ids[index]}"
        if entry.get("token") != expected_token:
            raise MalformedGenerateResponseError(
                f"Engine response choice.logprobs.content[{index}].token must be {expected_token!r}."
            )
        raw_logprob = entry.get("logprob")
        if isinstance(raw_logprob, bool) or not isinstance(raw_logprob, (int, float)):
            raise MalformedGenerateResponseError(
                f"Engine response choice.logprobs.content[{index}].logprob must be a number."
            )
        try:
            logprob = float(raw_logprob)
        except OverflowError as exc:
            raise MalformedGenerateResponseError(
                f"Engine response choice.logprobs.content[{index}].logprob must be finite."
            ) from exc
        if not math.isfinite(logprob):
            raise MalformedGenerateResponseError(
                f"Engine response choice.logprobs.content[{index}].logprob must be finite."
            )
        if logprob == VLLM_LOGPROB_SENTINEL:
            raise MalformedGenerateResponseError(
                f"Engine response choice.logprobs.content[{index}].logprob does not contain sampling evidence."
            )
        completion_logprobs.append(logprob)
    return completion_logprobs


def _parse_sampling_mask(
    choice: Mapping[str, Any], completion_ids: list[int]
) -> list[list[int]] | None:
    sampling_mask = choice.get("sampling_mask")
    if sampling_mask is None:
        return None
    if not isinstance(sampling_mask, list) or len(sampling_mask) != len(completion_ids):
        raise MalformedGenerateResponseError("Sampling mask token count mismatch")
    for sampled_id, support in zip(completion_ids, sampling_mask, strict=True):
        if (
            not isinstance(support, list)
            or not support
            or any(type(i) is not int or i < 0 for i in support)
            or len(set(support)) != len(support)
            or sampled_id not in support
        ):
            raise MalformedGenerateResponseError("Invalid recorded sampling support")
    return sampling_mask


def _select_sampler_head(
    head: dict[int, float],
    sampled_id: int,
    sampled_logprob: float,
    k: int,
    support: list[int] | None,
) -> tuple[list[int], list[float]]:
    """Keep an action-independent head, or the complete recorded replay support."""
    if sampled_id in head and not math.isclose(
        head[sampled_id], sampled_logprob, rel_tol=1e-6, abs_tol=1e-5
    ):
        raise MalformedGenerateResponseError(
            "Sampler head and sampled-token logprob disagree"
        )
    # vLLM serializes filtered-out (-inf) candidates as -9999.
    ranked = sorted(
        ((i, p) for i, p in head.items() if p != VLLM_LOGPROB_SENTINEL),
        key=lambda x: (-x[1], x[0]),
    )
    if support is not None:
        support_set = set(support)
        if not support_set.issubset(i for i, _ in ranked):
            raise MalformedGenerateResponseError(
                "Missing sampling-support probabilities; increase sampling.logprobs for cutoff ties"
            )
        ranked = [(i, p) for i, p in ranked if i in support_set]
        if not math.isclose(
            math.fsum(math.exp(p) for _, p in ranked), 1.0, rel_tol=0, abs_tol=1e-4
        ):
            raise MalformedGenerateResponseError(
                "Sampler probabilities must normalize over the recorded support"
            )
    elif len(ranked) > k and ranked[k - 1][1] == ranked[k][1]:
        boundary = ranked[k][1]
        ranked = [(i, p) for i, p in ranked if p > boundary]
    else:
        ranked = ranked[:k]
    if not ranked or math.fsum(math.exp(p) for _, p in ranked) > 1.0001:
        raise MalformedGenerateResponseError("Invalid sampler head probability mass")
    if support is None and len(head) < k + 2:
        raise MalformedGenerateResponseError(
            "Engine returned fewer top logprobs than requested"
        )
    return [i for i, _ in ranked], [p for _, p in ranked]


def _parse_completion_top_logprobs(
    choice: Mapping[str, Any], completion_ids: list[int], k: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Select the head from k+2 candidates, or keep every recorded support token."""
    content = choice["logprobs"]["content"]
    sampling_mask = _parse_sampling_mask(choice, completion_ids)
    top_ids: list[list[int]] = []
    top_logprobs: list[list[float]] = []
    for index, (sampled_id, entry) in enumerate(
        zip(completion_ids, content, strict=True)
    ):
        candidates = entry.get("top_logprobs")
        if not isinstance(candidates, list) or not candidates:
            raise MalformedGenerateResponseError("Missing requested top logprobs")
        head: dict[int, float] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise MalformedGenerateResponseError("Top logprob must be an object")
            token, logprob = candidate.get("token"), candidate.get("logprob")
            if (
                not isinstance(token, str)
                or not token.startswith("token_id:")
                or not token[9:].isdigit()
            ):
                raise MalformedGenerateResponseError(
                    "Top logprob token must be token_id:<nonnegative integer>"
                )
            if (
                isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
                or not math.isfinite(logprob)
            ):
                raise MalformedGenerateResponseError("Top logprob must be finite")
            if logprob > 1e-6:
                raise MalformedGenerateResponseError("Top logprob cannot be positive")
            token_id = int(token[9:])
            if token_id in head:
                raise MalformedGenerateResponseError("Duplicate top logprob token id")
            head[token_id] = float(logprob)
        selected_ids, selected_logprobs = _select_sampler_head(
            head,
            sampled_id,
            entry["logprob"],
            k,
            sampling_mask[index] if sampling_mask is not None else None,
        )
        top_ids.append(selected_ids)
        top_logprobs.append(selected_logprobs)
    return top_ids, top_logprobs


async def generate(
    *,
    client: AsyncOpenAI,
    renderer: Renderer,
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
    process_multimodal: bool = True,
) -> dict[str, Any]:
    """Tokenize messages, call vLLM /inference/v1/generate, parse the response.

    ``sampling_params`` is forwarded to vLLM, with ``stop_token_ids`` taken
    from the renderer and ``logprobs`` set to at least one for sampled-token
    evidence. A positive integer requests a top-k head; we ask the engine
    for two extra candidates to select that head independently of the sampled
    action, excluding ties at its boundary. When the engine returns sampling
    masks, capture every recorded support probability, including cutoff ties.
    Head requests opt into compact numeric HTTP logprobs; set
    ``extra_args.prl_compact_logprobs=False`` to use native JSON. Servers without
    the extension can still return native JSON. Sampled-only requests keep the
    native path by default.
    Pass
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
    With ``process_multimodal=False``, rendering skips image processing and
    the request carries ``content_parts`` with ``return_token_ids=true``;
    vLLM returns the expanded prompt and multimodal placeholder ranges.

    ``max_prompt_len`` controls the pre-flight overflow check. When the
    rendered prompt is strictly longer than the cap, the request is never
    sent and ``OverlongPromptError`` is raised. If ``max_prompt_len`` is
    ``None`` (the default), the cap is auto-discovered once per
    ``(base_url, model)`` via ``GET /v1/models`` (vLLM's
    ``ModelCard.max_model_len`` extension); engines that don't expose it
    cache a ``None`` cap and the pre-flight silently disables. Engine 4xx
    that still slip through propagate raw — converting them into a domain
    error is the calling client's job (its error shape is engine-specific).
    Calls with ``process_multimodal=False`` skip this pre-flight because only
    vLLM knows the expanded prompt length.

    Returns a dict with: request_id, prompt_ids, renderer_prompt_ids,
    mm_placeholders, completion_ids, completion_logprobs, content,
    reasoning_content, tool_calls, finish_reason, routed_experts,
    multi_modal_data, prompt_attribution. When ``sampling_params["logprobs"]``
    is a positive int, also ``completion_top_ids`` / ``completion_top_logprobs``
    (original sampler probabilities on the head or complete replay support).
    ``renderer_prompt_ids`` is the
    unexpanded logical prompt when ``process_multimodal=False`` and ``None``
    otherwise.

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
    if not process_multimodal and not getattr(
        renderer, "supports_process_multimodal", False
    ):
        raise NotImplementedError(
            f"{type(renderer).__name__} does not support process_multimodal=False"
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
        render_kwargs: dict[str, Any] = {}
        if not process_multimodal:
            render_kwargs["process_multimodal"] = False
        rendered = renderer.render(
            messages,
            tools=tools,
            add_generation_prompt=True,
            **render_kwargs,
        )
        return (
            rendered.token_ids,
            renderer.get_stop_token_ids(),
            rendered.multi_modal_data,
            rendered,
        )

    prompt_ids, stop_token_ids, mm_data, prompt_attr = _prepare()

    if process_multimodal:
        if max_prompt_len is None:
            max_prompt_len = await _resolve_max_prompt_len(client, model)
        if max_prompt_len is not None and len(prompt_ids) > max_prompt_len:
            raise OverlongPromptError(
                prompt_len=len(prompt_ids), max_prompt_len=max_prompt_len
            )

    sp: dict[str, Any] = dict(sampling_params or {})
    sp["stop_token_ids"] = stop_token_ids
    requested_logprobs = sp.get("logprobs")
    head_size = (
        requested_logprobs
        if type(requested_logprobs) is int and requested_logprobs > 0
        else None
    )
    sp["logprobs"] = head_size + 2 if head_size is not None else 1
    if head_size is not None:
        # Flat storage avoids millions of live Logprob objects for long completions.
        sp.setdefault("flat_logprobs", True)
        sp["extra_args"] = {
            "prl_compact_logprobs": True,
            **(sp.get("extra_args") or {}),
        }
        if not sp.get("stop"):
            sp.setdefault("detokenize", False)
    sp.setdefault("skip_special_tokens", False)

    body: dict[str, Any] = {
        "model": model,
        "token_ids": prompt_ids,
        "sampling_params": sp,
    }
    content_parts = _content_parts(messages) if not process_multimodal else None
    features = (
        _build_mm_features(renderer, mm_data)
        if process_multimodal and mm_data and not mm_data.is_empty()
        else None
    )
    if content_parts:
        body["content_parts"] = content_parts
        body["return_token_ids"] = True
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
    data = parse_generate_response(
        raw_response.content,
        compact_logprobs=bool((sp.get("extra_args") or {}).get("prl_compact_logprobs")),
    )

    choice = (data.get("choices") or [{}])[0]
    completion_ids = choice.get("token_ids") or []
    effective_prompt_ids = data.get("prompt_token_ids")
    if content_parts and not isinstance(effective_prompt_ids, list):
        raise MalformedGenerateResponseError(
            "Engine response must include prompt_token_ids when process_multimodal=False."
        )
    mm_placeholders = data.get("mm_placeholders")
    if content_parts and not isinstance(mm_placeholders, dict):
        raise MalformedGenerateResponseError(
            "Engine response must include mm_placeholders when process_multimodal=False."
        )

    # Top-k sampler head (ids + logprobs), only parsed when the caller asked
    # for more than the sampled token. Score centering consumes it to cancel
    # trainer/sampler drift on off-policy rollouts.
    completion_top_ids = None
    completion_top_logprobs = None
    if choice.get("compact_logprobs") is not None:
        completion_logprobs, completion_top_ids, completion_top_logprobs = (
            _parse_compact_logprobs(choice, completion_ids, head_size)
        )
    else:
        completion_logprobs = _parse_completion_logprobs(choice, completion_ids)
        if head_size is not None:
            completion_top_ids, completion_top_logprobs = (
                _parse_completion_top_logprobs(choice, completion_ids, head_size)
            )

    parsed = renderer.parse_response(
        completion_ids,
        prompt_ids=list(effective_prompt_ids or prompt_ids),
        tools=tools,
    )

    routed_experts = choice.get("routed_experts")
    # vLLM's native kept-set sampling masks (``--return-sampling-mask``):
    # one list of surviving vocab ids per completion token.
    sampling_mask = choice.get("sampling_mask")

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
        "usage": data.get("usage"),
        "prompt_ids": list(effective_prompt_ids or prompt_ids),
        "renderer_prompt_ids": list(prompt_ids) if content_parts else None,
        "mm_placeholders": mm_placeholders,
        "completion_ids": list(completion_ids),
        "completion_logprobs": completion_logprobs,
        "completion_top_ids": completion_top_ids,
        "completion_top_logprobs": completion_top_logprobs,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": finish_reason,
        "reasoning_complete": parsed.reasoning_complete,
        "routed_experts": routed_experts,
        "sampling_mask": sampling_mask,
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


_MEDIA_URL_TYPES = {
    "image": "image_url",
    "image_url": "image_url",
    "audio": "audio_url",
    "audio_url": "audio_url",
    "video": "video_url",
    "video_url": "video_url",
}


def _content_parts(messages: list[Message]) -> list[dict[str, Any]]:
    """Flatten raw media in prompt order for vLLM's token generate endpoint."""
    parts: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            part_type = part.get("type")
            if part_type is None:
                part_type = next(
                    (key for key in _MEDIA_URL_TYPES if part.get(key)), None
                )
            if not isinstance(part_type, str) or part_type not in _MEDIA_URL_TYPES:
                continue
            source = part.get(part_type)
            if source is None:
                source = part.get(_MEDIA_URL_TYPES[part_type]) or part.get("url")
            url = source.get("url") if isinstance(source, Mapping) else source
            if not isinstance(url, str) or not url:
                raise ValueError(f"{part_type} content part is missing a URL")
            parts.append({"type": _MEDIA_URL_TYPES[part_type], "url": url})
    return parts


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
    from renderers.nemotron3 import Nemotron35Renderer
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
    if issubclass(renderer_cls, Nemotron35Renderer):
        return _build_nemotron35_features(mm_data)

    raise NotImplementedError(
        f"Multimodal serialization not implemented for {renderer_cls.__name__}. "
        "Add a dispatch branch in renderers.client._build_mm_features."
    )


def _build_nemotron35_features(mm_data: MultiModalData) -> dict[str, Any]:
    """Encode dynamic-resolution Nemotron image inputs for vLLM."""
    _require_transformers("Encoding Nemotron 3.5 multimodal features for vLLM")
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
            encode_mm_kwargs_item,
        )
        from vllm.multimodal.inputs import (
            MultiModalFieldConfig,
            MultiModalKwargsItems,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron 3.5 multimodal generate requires vLLM and torch."
        ) from exc

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }
    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        pixel_values_flat = []
        imgs_sizes = []
        num_tokens_per_image = []
        for item in image_items:
            pixel_values = torch.as_tensor(item["pixel_values"])
            if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
                raise ValueError(
                    "Each Nemotron pixel_values item must have shape (1, C, H, W)."
                )
            sizes = item["imgs_sizes"]
            token_counts = item["num_tokens"]
            if not (len(sizes) == len(token_counts) == 1):
                raise ValueError(
                    "Each Nemotron image item must contain one size and token count."
                )
            pixel_values_flat.append(pixel_values[0])
            imgs_sizes.append(tuple(int(value) for value in sizes[0]))
            num_tokens_per_image.append(int(token_counts[0]))

        hf_inputs = BatchFeature(
            data={
                "pixel_values_flat": pixel_values_flat,
                "imgs_sizes": imgs_sizes,
                "num_tokens_per_image": num_tokens_per_image,
            }
        )
        field_config = {
            "pixel_values_flat": MultiModalFieldConfig.batched("image"),
            "num_tokens_per_image": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
            "imgs_sizes": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }
        kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, field_config)
        out["kwargs_data"]["image"] = [
            encode_mm_kwargs_item(item) for item in kwargs_items["image"]
        ]
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        out["mm_placeholders"]["image"] = [
            {"offset": placeholder.offset, "length": placeholder.length}
            for placeholder in mm_data.mm_placeholders.get("image") or []
        ]

    if not any(out["kwargs_data"].values()):
        out["kwargs_data"] = None
    return out


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
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
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
        out["mm_placeholders"]["image"] = [
            {"offset": placeholder.offset, "length": placeholder.length}
            for placeholder in mm_data.mm_placeholders.get("image") or []
        ]

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
        from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
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
