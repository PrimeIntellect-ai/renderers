"""Renderer-based generate client for vLLM + Dynamo.

Two transports, selected per-call via ``transport=`` parameter:

    "vllm" (default)
        messages → Renderer.render_ids() → token IDs → POST /inference/v1/generate
        → completion tokens → Renderer.parse_response() → structured message
        vLLM's TITO surface (server.py mounts the route in prime-rl).

    "dynamo"
        messages → Renderer.render_ids() → token IDs → POST /v1/chat/completions
        with ``nvext.token_data`` + ``nvext.extra_fields=["engine_data"]``
        → completion tokens via ``nvext.engine_data.completion_token_ids``
        → Renderer.parse_response() → structured message
        Dynamo has no ``/inference/v1/generate`` route; this branch posts to
        the standard OpenAI chat-completions surface and reads the engine
        token IDs back via the ``nvext.engine_data`` channel.

When a RendererPool is passed instead of a single Renderer, the sync tokenization
and parsing work is offloaded to threads for parallel execution across rollouts.
HuggingFace fast tokenizers release the GIL during Rust encoding, so threads
achieve real parallelism.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

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


# Public type alias; matches verifiers.types.RendererTransport string set.
RendererTransport = Literal["vllm", "dynamo"]

# Keys never forwarded to Dynamo at the top level: vLLM/prime-only fields its
# strict validator rejects (mirrors the token client's drop set). ``priority``
# is routed into nvext.agent_hints and ``routed_experts_prompt_start`` into
# nvext.routed_experts_prompt_start instead (the worker applies the latter to
# SamplingParams so vLLM trims routing engine-side). ``max_tokens`` and ``nvext``
# are handled explicitly and skipped separately.
_DYNAMO_DROP_KEYS = frozenset(
    {
        "return_token_ids",
        "spaces_between_special_tokens",
        "priority",
        "routed_experts_prompt_start",
    }
)

# Absolute /inference/v1/generate URLs, cached per client base_url.
_vllm_endpoint_cache: dict[str, str] = {}


def _vllm_endpoint(base_url: str) -> str:
    """Absolute ``/inference/v1/generate`` URL for ``base_url`` (cached).

    The route is mounted at the server root, not under /v1, so strip the
    client's trailing /v1 and build an absolute URL — otherwise AsyncOpenAI
    prepends its automatic /v1.
    """
    endpoint = _vllm_endpoint_cache.get(base_url)
    if endpoint is None:
        endpoint = f"{base_url.rstrip('/').removesuffix('/v1')}/inference/v1/generate"
        _vllm_endpoint_cache[base_url] = endpoint
    return endpoint


def _flatten_chat_logprobs(choice: Mapping[str, Any]) -> list[float]:
    """Flatten ChatCompletionLogProbs ``{"content": [{"logprob": ...}, ...]}``."""
    raw = choice.get("logprobs") or {}
    content = raw.get("content") if isinstance(raw, dict) else None
    return [float(c.get("logprob") or 0.0) for c in content or []]


@dataclass(frozen=True)
class _WireResult:
    """Normalized fields extracted from a backend's raw response."""

    completion_ids: list[int]
    completion_logprobs: list[float]
    routed_experts: Any
    request_id: str
    finish_reason: str | None


class _Transport(ABC):
    """Per-backend request/response strategy for :func:`generate`."""

    @abstractmethod
    async def post(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_ids: list[int],
        sp: dict[str, Any],
        renderer: Renderer | RendererPool,
        mm_data: MultiModalData | None,
        cache_salt: str | None,
        priority: int | None,
        extra_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Build the wire body, POST it, and return the decoded response dict."""

    @abstractmethod
    def parse(self, data: dict[str, Any]) -> _WireResult:
        """Extract normalized completion fields from the backend response."""


class _VllmGenerateTransport(_Transport):
    """vLLM TITO surface: ``POST /inference/v1/generate``."""

    async def post(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_ids: list[int],
        sp: dict[str, Any],
        renderer: Renderer | RendererPool,
        mm_data: MultiModalData | None,
        cache_salt: str | None,
        priority: int | None,
        extra_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        features = (
            _build_mm_features(renderer, mm_data)
            if mm_data and not mm_data.is_empty()
            else None
        )
        body: dict[str, Any] = {
            "model": model,
            "token_ids": prompt_ids,
            "sampling_params": sp,
        }
        if features is not None:
            body["features"] = features
        if cache_salt is not None:
            body["cache_salt"] = cache_salt
        if priority is not None:
            body["priority"] = priority

        endpoint = _vllm_endpoint(str(client.base_url))
        _request_logger.debug(
            "POST %s prompt_len=%d max_tokens=%s",
            endpoint,
            len(prompt_ids),
            sp.get("max_tokens"),
        )
        post_kwargs: dict[str, Any] = {"cast_to": httpx.Response, "body": body}
        if extra_headers:
            post_kwargs["options"] = cast(Any, {"headers": extra_headers})
        raw_response = await client.post(endpoint, **post_kwargs)
        return parse_generate_response(raw_response.content)

    def parse(self, data: dict[str, Any]) -> _WireResult:
        choice = (data.get("choices") or [{}])[0]
        return _WireResult(
            completion_ids=list(choice.get("token_ids") or []),
            completion_logprobs=_flatten_chat_logprobs(choice),
            routed_experts=choice.get("routed_experts"),
            request_id=data.get("request_id") or "",
            finish_reason=choice.get("finish_reason"),
        )


class _DynamoChatTransport(_Transport):
    """NVIDIA Dynamo: ``POST /v1/chat/completions`` with the nvext envelope.

    ``nvext.token_data`` carries the pre-tokenized prompt; ``extra_fields=
    ["engine_data"]`` opts into the completion-IDs/logprobs/routed_experts
    channel. routed_experts is normalized to the ``{data, shape, start,
    dtype}`` contract.
    """

    async def post(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_ids: list[int],
        sp: dict[str, Any],
        renderer: Renderer | RendererPool,
        mm_data: MultiModalData | None,
        cache_salt: str | None,
        priority: int | None,
        extra_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        # TODO: Implement multimodal support for dynamo transport.
        if mm_data is not None and not mm_data.is_empty():
            raise NotImplementedError(
                "Multimodal renderers are not yet supported on the dynamo "
                "transport. Use vllm or stay on the token-client TITO "
                "path for VLMs."
            )
        body = self._build_body(model, prompt_ids, sp, cache_salt, priority)
        post_kwargs: dict[str, Any] = {"cast_to": httpx.Response, "body": body}
        if extra_headers:
            post_kwargs["options"] = cast(Any, {"headers": extra_headers})
        # Engine 4xx propagate raw (matches the vLLM path).
        raw_response = await client.post("/chat/completions", **post_kwargs)
        # Keep routed_experts blob as zero-copy memoryview (avoids event-loop json.loads).
        resp = _parse_dynamo_response(raw_response.content)
        # Back-compat trim: no-op when worker already trimmed engine-side (start>0).
        _trim_dynamo_routed_experts(resp, sp)
        return resp

    @staticmethod
    def _build_body(
        model: str,
        prompt_ids: list[int],
        sp: dict[str, Any],
        cache_salt: str | None,
        priority: int | None,
    ) -> dict[str, Any]:
        # kwargs win over sampling_params; both route into nvext on Dynamo.
        if cache_salt is None:
            cache_salt = sp.get("cache_salt")
        if priority is None:
            priority = sp.get("priority")

        # Merge caller nvext; layer required fields on top.
        nvext: dict[str, Any] = dict(sp.get("nvext") or {})
        nvext["token_data"] = list(prompt_ids)
        extra_fields = list(nvext.get("extra_fields") or [])
        # Only "engine_data": routed_experts nests inside it, so requesting
        # the dedicated field separately would duplicate the blob on the wire.
        if "engine_data" not in extra_fields:
            extra_fields.append("engine_data")
        nvext["extra_fields"] = extra_fields
        if cache_salt is not None:
            nvext["cache_salt"] = cache_salt
        if priority is not None:
            agent_hints = dict(nvext.get("agent_hints") or {})
            agent_hints["priority"] = priority
            nvext["agent_hints"] = agent_hints
        # Rides nvext; Dynamo rejects unknown top-level chat fields.
        reps = sp.get("routed_experts_prompt_start")
        if reps is not None:
            nvext["routed_experts_prompt_start"] = reps

        # messages is a placeholder stub the OpenAI schema requires but Dynamo
        # ignores. tools are baked into token_data; forwarding the renderer
        # ToolSpec (not the OpenAI tool shape) would 400.
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "stream": False,
            "nvext": nvext,
        }
        if sp.get("max_tokens") is not None:
            body["max_completion_tokens"] = sp["max_tokens"]

        # Forward every other non-None sampling field (denylist, not allowlist)
        # so caller-requested params (presence_penalty, stop, guided_*, ...) are
        # not silently dropped. Mirrors the token client's body construction.
        for key, value in sp.items():
            if value is None or key in _DYNAMO_DROP_KEYS or key in body:
                continue
            if key in ("nvext", "max_tokens", "cache_salt"):
                continue  # handled above (cache_salt -> nvext; priority denylisted)
            if key == "logprobs":
                # vLLM takes logprobs=N (int); Dynamo's chat schema wants the
                # OpenAI bool + top_logprobs split.
                body["logprobs"] = True
                if isinstance(value, int) and value > 1:
                    body["top_logprobs"] = value
            else:
                body[key] = value
        return body

    def parse(self, data: dict[str, Any]) -> _WireResult:
        choice = (data.get("choices") or [{}])[0]
        nvext = data.get("nvext") or {}
        engine = nvext.get("engine_data") or {}

        # Canonical Dynamo channel first (nvext.engine_data, then top-level
        # nvext), then the OpenAI-extended choices[0].token_ids. The
        # engine channel is authoritative — choices[0].token_ids may be a
        # detokenize-then-retokenize echo that differs from what was sampled.
        completion_ids = None
        present = False
        for src in (engine, nvext):
            if src.get("completion_token_ids") is not None:
                completion_ids = src["completion_token_ids"]
                present = True
                break
        if not present and choice.get("token_ids") is not None:
            completion_ids = choice["token_ids"]
            present = True
        if not present:
            # Field absent (vs. an empty completion) — usually a missing
            # nvext.extra_fields=["engine_data"] opt-in.
            raise RuntimeError(
                "dynamo response carried no completion token IDs "
                "(expected nvext.engine_data.completion_token_ids)."
            )
        completion_ids = list(completion_ids or [])

        # Prefer engine_data.completion_logprobs — the same authoritative source
        # as the engine completion_token_ids used above — so logprobs stay
        # positionally aligned with the ids. The choices[0] chat logprobs are a
        # detokenize/retokenize echo that can diverge from the sampled ids, which
        # would misalign while still passing the length check below. Fall back to
        # the chat logprobs only when the engine channel is absent. A present
        # empty engine list is still authoritative for source selection and must
        # NOT fall through to the chat echo, but it is only valid for a
        # zero-token completion.
        engine_logprobs = engine.get("completion_logprobs")
        if engine_logprobs is not None:
            logprobs = [float(x) for x in engine_logprobs]
        else:
            logprobs = _flatten_chat_logprobs(choice)
        # Logprobs are indexed positionally against completion_ids downstream;
        # a length mismatch would silently misalign tokens and logprobs or
        # produce samples that prime-rl later rejects.
        if len(logprobs) != len(completion_ids):
            raise RuntimeError(
                f"dynamo logprobs length ({len(logprobs)}) does not match "
                f"completion token count ({len(completion_ids)})."
            )

        # Prefer nvext.routed_experts, fall back to engine_data.routed_experts.
        routed_experts = nvext.get("routed_experts")
        if not isinstance(routed_experts, Mapping):
            routed_experts = engine.get("routed_experts")
        routed_experts = _normalize_routed_experts(routed_experts)

        return _WireResult(
            completion_ids=completion_ids,
            completion_logprobs=logprobs,
            routed_experts=routed_experts,
            request_id=data.get("request_id") or data.get("id") or "",
            finish_reason=choice.get("finish_reason"),
        )


_TRANSPORTS: dict[str, _Transport] = {
    "vllm": _VllmGenerateTransport(),
    "dynamo": _DynamoChatTransport(),
}


def _normalize_routed_experts(payload: Any) -> dict[str, Any] | None:
    """Validate/normalize a dynamo routed_experts payload to the
    ``{data, shape, start, dtype}`` contract.

    Defaults ``start=0`` and ``dtype="uint8"`` for back-compat with payloads
    serialized before those fields existed; raises a clear ``RuntimeError`` for
    a non-contract payload (string/map, wrong rank) so the failure surfaces here
    with context instead of as a ``TypeError``/``KeyError`` in trajectory
    processing.
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or "data" not in payload or "shape" not in payload:
        raise RuntimeError(
            "dynamo routed_experts must be a mapping with 'data' and "
            f"'shape'; got {type(payload).__name__}"
        )
    shape = payload["shape"]
    if not (isinstance(shape, (list, tuple)) and len(shape) == 3):
        raise RuntimeError(
            "dynamo routed_experts 'shape' must be 3-D [seq, layers, topk]; "
            f"got {shape!r}"
        )
    return {
        "data": payload["data"],
        "shape": [int(d) for d in shape],
        "start": int(payload.get("start", 0)),
        "dtype": payload.get("dtype", "uint8"),
    }


_ROUTED_EXPERTS_ITEMSIZE = {"uint8": 1, "uint16": 2, "int16": 2, "int32": 4}


def _trim_dynamo_routed_experts(resp: Any, sp: dict[str, Any]) -> None:
    """Client-side trim of a dynamo routed_experts payload, in place.

    This is a **back-compat fallback**. The renderer now forwards
    ``routed_experts_prompt_start`` via ``nvext`` (see ``_build_body``), so a
    current Dynamo worker trims the leading prompt rows engine-side (vLLM) and
    stamps the payload's ``start`` > 0 — in which case this is a no-op. We only
    trim here when the worker returned FULL-sequence routing with ``start == 0``
    (an older worker that ignored the nvext field) and the caller supplied a
    positive ``routed_experts_prompt_start``: drop that many leading rows and set
    ``start``. No-op when routed_experts is absent/empty, the worker already
    trimmed (``start`` > 0), or no positive offset is supplied (first-turn
    requests keep full-sequence routing with ``start=0``).
    """
    if not isinstance(resp, Mapping):
        return
    nvext = resp.get("nvext")
    if not isinstance(nvext, Mapping):
        return
    routed = nvext.get("routed_experts")
    if not isinstance(routed, dict):
        engine = nvext.get("engine_data")
        routed = engine.get("routed_experts") if isinstance(engine, Mapping) else None
    if not isinstance(routed, dict):
        return
    data = routed.get("data")
    shape = routed.get("shape")
    # data may be a zero-copy memoryview/bytes (engine_data blob kept un-parsed)
    # or a str; base64.b64decode accepts all three.
    if not isinstance(data, (str, bytes, memoryview)) or not (
        isinstance(shape, (list, tuple)) and len(shape) == 3
    ):
        return

    offset = sp.get("routed_experts_prompt_start")
    if offset is None:
        return
    # Worker already trimmed engine-side (stamped start > 0) — don't double-trim.
    if int(routed.get("start") or 0) != 0:
        return
    offset = max(0, min(int(offset), int(shape[0])))
    if offset == 0:
        return

    dtype = routed.get("dtype", "uint8")
    itemsize = _ROUTED_EXPERTS_ITEMSIZE.get(dtype)
    if itemsize is None:
        raise RuntimeError(
            f"unknown routed_experts dtype {dtype!r}; "
            f"expected one of {sorted(_ROUTED_EXPERTS_ITEMSIZE)}"
        )
    row_size = int(shape[1]) * int(shape[2]) * itemsize
    trimmed = base64.b64decode(data)[offset * row_size :]
    routed["data"] = base64.b64encode(trimmed).decode("ascii")
    routed["shape"] = [int(shape[0]) - offset, int(shape[1]), int(shape[2])]
    routed["start"] = offset


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


def _parse_dynamo_response(raw: bytes) -> dict[str, Any]:
    """Parse a dynamo response, keeping the routed_experts base64 blob as a
    zero-copy ``memoryview`` instead of decoding it through ``json.loads`` (a
    large blob would block the event loop). Mirrors ``parse_generate_response``,
    but re-attaches the blob at the nvext location (``nvext.routed_experts`` or
    ``nvext.engine_data.routed_experts``) rather than ``choices[0]``."""
    stripped, routed_data = strip_routed_experts_data(raw)
    payload: dict[str, Any] = json.loads(stripped)
    if routed_data is not None:
        nvext = payload.get("nvext")
        if isinstance(nvext, dict):
            routed = nvext.get("routed_experts")
            if not isinstance(routed, dict):
                engine = nvext.get("engine_data")
                routed = (
                    engine.get("routed_experts") if isinstance(engine, dict) else None
                )
            if isinstance(routed, dict):
                routed["data"] = routed_data
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
    transport: RendererTransport = "vllm",
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

    impl = _TRANSPORTS.get(transport)
    if impl is None:
        raise ValueError(f"Unknown renderer transport: {transport!r}")
    data = await impl.post(
        client=client,
        model=model,
        prompt_ids=prompt_ids,
        sp=sp,
        renderer=renderer,
        mm_data=mm_data,
        cache_salt=cache_salt,
        priority=priority,
        extra_headers=extra_headers,
    )
    wire = impl.parse(data)

    parsed = await _maybe_offload(
        renderer, lambda: renderer.parse_response(wire.completion_ids, tools=tools)
    )

    # /inference/v1/generate never returns "tool_calls", so promote
    # stop→tool_calls when we parsed tool calls client-side (keeps agent
    # loops going). No-op on dynamo, which can return it directly.
    finish_reason = wire.finish_reason
    ok_tool_calls = [
        tc for tc in parsed.tool_calls if tc.status == ToolCallParseStatus.OK
    ]
    if ok_tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    return {
        "request_id": wire.request_id,
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(wire.completion_ids),
        "completion_logprobs": wire.completion_logprobs,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": finish_reason,
        "routed_experts": wire.routed_experts,
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

    NOTE — future engine pluggability: this encoder is vLLM-specific
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
