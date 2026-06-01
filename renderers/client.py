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
import contextlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from pathlib import Path
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
from renderers.mm_store import (
    build_mm_feature_envelope,
    mm_feature_envelope_matches,
    mm_feature_fingerprint,
    mm_feature_path,
    mmfile_ref,
    run_id_from_env,
)

_request_logger = logging.getLogger("renderers.client")
ROUTED_EXPERTS_DATA_PREFIX = b'"routed_experts":{"data":"'
_MM_MAX_INFLIGHT_ENV = "RENDERERS_MM_MAX_INFLIGHT"
_DEFAULT_MM_MAX_INFLIGHT = 4
_mm_payload_semaphores: dict[tuple[int, int], asyncio.Semaphore] = {}
_MM_FEATURE_STORE_MODE_ENV = "RENDERERS_MM_FEATURE_STORE_MODE"


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
        super().__init__(f"Prompt length ({prompt_len}) exceeds maximum context length ({max_prompt_len}).")


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


def _mm_max_inflight() -> int | None:
    raw = os.getenv(_MM_MAX_INFLIGHT_ENV)
    if raw is None:
        return _DEFAULT_MM_MAX_INFLIGHT
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MM_MAX_INFLIGHT
    if value < 1:
        return None
    return value


def _mm_feature_store_mode() -> str:
    """Offload processed mm-feature payloads to disk and ship ``mmfile`` refs
    (``on``, the default) or inline them as base64 in the request (``off``)."""
    mode = os.getenv(_MM_FEATURE_STORE_MODE_ENV, "on").strip().lower()
    if mode in {"", "0", "false", "off", "disabled", "none", "no"}:
        return "off"
    if mode in {"1", "true", "on", "enabled", "yes"}:
        return "on"
    raise ValueError(f"Invalid {_MM_FEATURE_STORE_MODE_ENV}={mode!r}; expected on or off.")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _existing_mm_feature_valid(
    path: Path, *, run_id: str, fingerprint: str, modality: str, mm_hash: str, placeholder_length: int
) -> bool:
    try:
        import msgpack

        with path.open("rb") as f:
            packed = f.read()
        artifact = msgpack.unpackb(packed, raw=False)
        envelope = artifact.get("envelope") if isinstance(artifact, dict) else None
        payload = artifact.get("payload") if isinstance(artifact, dict) else None
        if not isinstance(envelope, dict) or not isinstance(payload, bytes):
            return False
        # Validate placeholder_length too: vLLM checks it on load, but the envelope
        # match doesn't — so a stale artifact with the right hash/fingerprint but a
        # wrong placeholder_length would fail in vLLM and never get repaired (we'd
        # keep skipping the rewrite). Treat a mismatch as invalid → rewrite.
        if envelope.get("placeholder_length") != int(placeholder_length):
            return False
        return mm_feature_envelope_matches(
            envelope,
            run_id=run_id,
            fingerprint=fingerprint,
            modality=modality,
            mm_hash=mm_hash,
            payload=payload,
        )
    except Exception:
        return False


def _write_mm_feature_artifact(
    *,
    run_id: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    payload: bytes,
    placeholder_length: int,
) -> str:
    import msgpack

    path = mm_feature_path(run_id=run_id, fingerprint=fingerprint, modality=modality, mm_hash=mm_hash)
    if path.exists() and _existing_mm_feature_valid(
        path,
        run_id=run_id,
        fingerprint=fingerprint,
        modality=modality,
        mm_hash=mm_hash,
        placeholder_length=placeholder_length,
    ):
        # Recurring image: the feature is already on disk and valid. Refresh its
        # mtime so the orchestrator's last-use feature sweep treats it as hot —
        # otherwise a frequently-reused feature keeps its first-write mtime, ages
        # past the TTL while still being referenced, and gets swept (forcing an
        # expensive force_full_pixels reprocess on the next reference). Suppress
        # OSError in case a concurrent sweep just removed it — the rewrite below
        # would handle that on a subsequent call anyway.
        with contextlib.suppress(OSError):
            os.utime(path, None)
        return mmfile_ref(run_id=run_id, fingerprint=fingerprint, modality=modality, mm_hash=mm_hash)

    envelope = build_mm_feature_envelope(
        run_id=run_id,
        fingerprint=fingerprint,
        modality=modality,
        mm_hash=mm_hash,
        payload=payload,
        placeholder_length=placeholder_length,
    )
    packed = msgpack.packb({"envelope": envelope, "payload": payload}, use_bin_type=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(packed)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise

    return mmfile_ref(run_id=run_id, fingerprint=fingerprint, modality=modality, mm_hash=mm_hash)


@contextlib.asynccontextmanager
async def _limit_mm_payloads(mm_data: MultiModalData | None) -> AsyncIterator[None]:
    if mm_data is None or mm_data.is_empty():
        yield
        return

    limit = _mm_max_inflight()
    if limit is None:
        yield
        return

    loop = asyncio.get_running_loop()
    key = (id(loop), limit)
    semaphore = _mm_payload_semaphores.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _mm_payload_semaphores[key] = semaphore

    async with semaphore:
        yield


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
    force_full_pixels: bool = False,
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

    ``force_full_pixels`` selects the multimodal serialization mode. When
    ``False`` (default), images that arrive descriptor-only (no
    ``pixel_values`` — typically prior-turn images carried through a bridge)
    are sent hash-only on the assumption the engine still has them cached,
    while new-turn images (which carry ``pixel_values``) are sent in full.
    When ``True``, ``materialize_pixels`` re-attaches pixels for every image
    and the whole prompt is sent in full — the caller's cache-miss fallback
    after a hash-only request is rejected by the engine.

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

    prompt_ids, stop_token_ids, mm_data, prompt_attr = await _maybe_offload(renderer, _prepare)

    if max_prompt_len is None:
        max_prompt_len = await _resolve_max_prompt_len(client, model)
    if max_prompt_len is not None and len(prompt_ids) > max_prompt_len:
        raise OverlongPromptError(prompt_len=len(prompt_ids), max_prompt_len=max_prompt_len)

    sp: dict[str, Any] = dict(sampling_params or {})
    sp["stop_token_ids"] = stop_token_ids
    sp["logprobs"] = 1
    sp.setdefault("skip_special_tokens", False)

    body: dict[str, Any] = {
        "model": model,
        "token_ids": prompt_ids,
        "sampling_params": sp,
    }

    # Multimodal: ``mm_data`` carried into the rollout is descriptor-only
    # (no ``pixel_values``) so the env worker never retains decoded image
    # tensors. Re-attach pixels for the POST via ``materialize_pixels``
    # (cache hit, else reprocess from the message base64), build the engine
    # features, then strip pixels again so the value handed back to the
    # trajectory stays descriptor-only.
    def _features_and_descriptor_mm() -> "tuple[dict[str, Any] | None, MultiModalData | None]":
        if mm_data is None or mm_data.is_empty():
            return None, mm_data
        # First attempt (``force_full_pixels=False``): send ``mm_data`` as-is.
        # New-turn images carry ``pixel_values`` (full payload); prior-turn
        # images are descriptor-only and ``_build_mm_features`` serializes them
        # hash-only, assuming the engine still has them cached.
        # Cache-miss fallback (``force_full_pixels=True``): re-attach pixels for
        # every image via ``materialize_pixels`` (reprocessed from the message
        # base64) so the whole prompt is sent in full. ``materialize_pixels``
        # lives on multimodal renderers + the pool, not the base ``Renderer``
        # protocol; reached only when ``mm_data`` is non-empty, which implies a
        # multimodal renderer.
        build_mm = cast(Any, renderer).materialize_pixels(mm_data, messages) if force_full_pixels else mm_data
        return _build_mm_features(renderer, build_mm), _strip_pixels(mm_data)

    async with _limit_mm_payloads(mm_data):
        features, out_mm_data = await _maybe_offload(renderer, _features_and_descriptor_mm)
    # ``prompt_attr.multi_modal_data`` aliases the original pixel-bearing
    # ``mm_data``; rebind it to the stripped copy so the attribution surfaced
    # to the trajectory is also descriptor-only.
    if prompt_attr is not None and getattr(prompt_attr, "multi_modal_data", None) is not None:
        prompt_attr = replace(prompt_attr, multi_modal_data=out_mm_data)
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
    completion_ids = choice.get("token_ids") or []

    parsed = await _maybe_offload(renderer, lambda: renderer.parse_response(completion_ids, tools=tools))

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
    ok_tool_calls = [tc for tc in parsed.tool_calls if tc.status == ToolCallParseStatus.OK]
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
        # multi-turn bridging and training-sample construction. Descriptor
        # only (``pixel_values`` stripped) — the env worker keeps no decoded
        # tensors; pixels are re-derived for the next-turn POST and for
        # training-sample construction.
        "multi_modal_data": out_mm_data,
        # The renderer's per-token attribution for the prompt — either
        # the RenderedTokens computed here via renderer.render(...) or
        # the one threaded in by the caller alongside prompt_ids (the
        # bridge path). Lets downstream consumers (verifiers
        # RendererClient → prime-rl) build SFT-on-tool-body and other
        # selective loss masks without a second render pass. ``None``
        # when the caller passed prompt_ids without attribution.
        "prompt_attribution": prompt_attr,
    }


def _strip_pixels(mm_data: MultiModalData) -> MultiModalData:
    """Return ``mm_data`` with ``pixel_values`` dropped from every item.

    Keeps the descriptor (``image_grid_thw`` etc.), ``mm_hashes`` and
    ``mm_placeholders`` — everything needed for token alignment and for
    re-deriving pixels later (POST via ``materialize_pixels``; training via
    the orchestrator). The decoded pixel tensors are never retained on the
    trajectory, which is what keeps env-worker memory flat across a rollout.
    """
    if not mm_data.mm_items:
        return mm_data
    new_items = {
        modality: [{k: v for k, v in item.items() if k != "pixel_values"} for item in items]
        for modality, items in mm_data.mm_items.items()
    }
    return replace(mm_data, mm_items=new_items)


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
    renderer_cls = renderer.renderer_cls if isinstance(renderer, RendererPool) else type(renderer)

    # Qwen3-VL and Qwen3.5 both ship ``pixel_values`` + ``image_grid_thw``
    # via the shared Qwen2-VL field factory. ``spatial_merge_size=2`` is
    # the family default and matches every Qwen-VL processor in tree.
    if issubclass(renderer_cls, (Qwen3VLRenderer, Qwen35Renderer)):
        return _build_qwen_vl_features(mm_data, spatial_merge_size=2)

    raise NotImplementedError(
        f"Multimodal serialization not implemented for {renderer_cls.__name__}. "
        "Add a dispatch branch in renderers.client._build_mm_features."
    )


def _build_qwen_vl_features(mm_data: MultiModalData, *, spatial_merge_size: int) -> dict[str, Any]:
    """vLLM features payload for the Qwen-VL family (Qwen2-VL / Qwen3-VL).

    Stacks per-image processor outputs back into a batched ``BatchFeature``,
    runs the Qwen2-VL field factory (shared across the family), wraps as
    ``MultiModalKwargsItems``, base64-encodes each item, and assembles a
    JSON-serializable dict matching vLLM's ``MultiModalFeatures`` schema.

    Returns ``None`` semantics live one level up — this helper assumes the
    caller already verified ``mm_data`` is non-empty.
    """
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
        from vllm.multimodal.inputs import MultiModalKwargsItems
        from vllm.v1.serial_utils import MsgpackEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Multimodal generate via /inference/v1/generate requires `vllm` "
            "and `torch` to encode the features payload. Install vLLM in this "
            "environment, or pre-build features upstream."
        ) from exc

    mode = _mm_feature_store_mode()
    run_id = run_id_from_env() if mode != "off" else ""
    encoder = MsgpackEncoder(size_threshold=2**62)
    fingerprint = mm_feature_fingerprint(family="qwen_vl", spatial_merge_size=spatial_merge_size)

    out: dict[str, Any] = {
        "mm_hashes": {},
        "mm_placeholders": {},
        "kwargs_data": {},
    }

    image_items = mm_data.mm_items.get("image") or []
    if image_items:
        # An item carrying ``pixel_values`` is sent as a full payload; an item
        # without (descriptor-only) is sent hash-only, on the assumption that
        # the engine already has it cached from an earlier turn. ``kwargs_data``
        # stays aligned with ``mm_items``: ``None`` marks a hash-only slot.
        # mm_items ship numpy arrays (the renderer is torch-free); convert at
        # this vLLM-glue boundary where torch is already a hard dependency.
        encoded: list[Any] = [None] * len(image_items)
        mmfile_count = 0
        inline_count = 0
        full_indices = [i for i, it in enumerate(image_items) if it.get("pixel_values") is not None]
        if full_indices:
            full_items = [image_items[i] for i in full_indices]
            pixel_values = torch.cat([torch.as_tensor(it["pixel_values"]) for it in full_items], dim=0)
            image_grid_thw = torch.cat([torch.as_tensor(it["image_grid_thw"]) for it in full_items], dim=0)
            hf_inputs = BatchFeature(data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})
            config = _create_qwen2vl_field_factory(spatial_merge_size)(hf_inputs)
            kwargs_items = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config)
            for idx, item in zip(full_indices, kwargs_items["image"]):
                bufs = encoder.encode(item)
                assert len(bufs) == 1, "All tensors should be inline"
                raw_payload = bufs[0]
                if mode == "off":
                    encoded[idx] = base64.b64encode(raw_payload).decode("ascii")
                    inline_count += 1
                    continue

                mm_hash = (mm_data.mm_hashes.get("image") or [])[idx]
                placeholder = (mm_data.mm_placeholders.get("image") or [])[idx]
                ref = _write_mm_feature_artifact(
                    run_id=run_id,
                    fingerprint=fingerprint,
                    modality="image",
                    mm_hash=mm_hash,
                    payload=raw_payload,
                    placeholder_length=placeholder.length,
                )
                encoded[idx] = ref
                mmfile_count += 1
        if image_items:
            _request_logger.debug(
                "built qwen-vl mm features mode=%s none=%d inline=%d mmfile=%d",
                mode,
                len(image_items) - len(full_indices),
                inline_count,
                mmfile_count,
            )
        out["kwargs_data"]["image"] = encoded
        out["mm_hashes"]["image"] = list(mm_data.mm_hashes.get("image") or [])
        out["mm_placeholders"]["image"] = [
            {"offset": p.offset, "length": p.length} for p in mm_data.mm_placeholders.get("image") or []
        ]

    # If no full payload was built across any modality, drop ``kwargs_data`` so
    # vLLM takes the hash-only (cache-hit) path. Otherwise hand it the payload
    # (with ``None`` slots for the hash-only images).
    if not any(any(item is not None for item in items) for items in out["kwargs_data"].values()):
        out["kwargs_data"] = None

    return out
