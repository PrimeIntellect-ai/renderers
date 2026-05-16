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
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, cast

import numpy as np
from openai import AsyncOpenAI, BadRequestError

from renderers.base import Message, Renderer, RendererPool, ToolSpec

_request_logger = logging.getLogger("renderers.client")


@dataclass(frozen=True)
class _FallbackPlaceholderRange:
    offset: int
    length: int
    is_embed: Any = None


@dataclass
class _FallbackMultiModalFieldElem:
    data: Any
    field: Any = None


@dataclass
class _FallbackMultiModalFeatureSpec:
    data: dict[str, _FallbackMultiModalFieldElem] | None
    modality: str
    identifier: str
    mm_position: _FallbackPlaceholderRange
    mm_hash: str | None = None


async def _run_pooled(pool: RendererPool, fn):
    def _work():
        with pool.checkout() as r:
            return fn(r)

    return await asyncio.to_thread(_work)


def _build_mm_features(renderer_cls: type, mm_data: Any) -> list[Any] | None:
    """Build vLLM multimodal feature specs for renderer-native payloads."""
    from renderers.qwen3_vl import Qwen3VLRenderer
    from renderers.qwen35 import Qwen35Renderer

    if issubclass(renderer_cls, (Qwen3VLRenderer, Qwen35Renderer)):
        # Qwen3-VL and Qwen3.5 both emit Qwen2-VL-family image payloads:
        # pixel_values plus image_grid_thw. All seven current Qwen3.5 sizes
        # use merge_size=2; move this to renderer metadata when that API lands.
        return _build_qwen_vl_features(mm_data, spatial_merge_size=2)

    raise NotImplementedError(f"No multimodal feature builder for {renderer_cls!r}")


def _build_qwen_vl_features(
    mm_data: Any, *, spatial_merge_size: int
) -> list[Any] | None:
    image_payloads = _image_payloads(mm_data)
    if not image_payloads:
        return None

    try:
        from vllm.multimodal.inputs import (
            MultiModalFeatureSpec,
            MultiModalFieldConfig,
            PlaceholderRange,
        )
    except Exception:
        return _build_fallback_qwen_vl_features(
            image_payloads, spatial_merge_size=spatial_merge_size
        )

    features: list[Any] = []
    next_offset = 0
    for payload_idx, payload in enumerate(image_payloads):
        pixel_values = _tensor_data(payload["pixel_values"])
        image_grid_thw = _image_grid_tensor(payload["image_grid_thw"])
        grid_rows = _grid_rows(image_grid_thw)
        sizes = [_grid_prod(row) for row in grid_rows]

        field_elems = MultiModalFieldConfig.flat_from_sizes(
            "image", _tensor(sizes, like=image_grid_thw)
        ).field.build_elems("image", "pixel_values", pixel_values)
        grid_elems = MultiModalFieldConfig.batched("image").field.build_elems(
            "image", "image_grid_thw", image_grid_thw
        )

        for image_idx, (pixel_elem, grid_elem, grid_row) in enumerate(
            zip(field_elems, grid_elems, grid_rows, strict=True)
        ):
            length = _grid_prod(grid_row) // (spatial_merge_size**2)
            mm_position = _placeholder_range(
                payload,
                image_idx,
                default_offset=next_offset,
                default_length=length,
                placeholder_cls=PlaceholderRange,
            )
            next_offset = mm_position.offset + mm_position.length
            feature_kwargs = {
                "data": {
                    "pixel_values": pixel_elem,
                    "image_grid_thw": grid_elem,
                },
                "modality": "image",
                "identifier": _identifier(payload, payload_idx, image_idx),
                "mm_position": mm_position,
                "mm_hash": _mm_hash(payload),
            }
            try:
                features.append(MultiModalFeatureSpec(**feature_kwargs))
            except TypeError:
                feature_kwargs.pop("mm_hash")
                features.append(MultiModalFeatureSpec(**feature_kwargs))

    return features


def _build_fallback_qwen_vl_features(
    image_payloads: list[dict[str, Any]], *, spatial_merge_size: int
) -> list[_FallbackMultiModalFeatureSpec]:
    features: list[_FallbackMultiModalFeatureSpec] = []
    next_offset = 0
    for payload_idx, payload in enumerate(image_payloads):
        for image_idx, grid_row in enumerate(_grid_rows(payload["image_grid_thw"])):
            length = _grid_prod(grid_row) // (spatial_merge_size**2)
            mm_position = _placeholder_range(
                payload,
                image_idx,
                default_offset=next_offset,
                default_length=length,
                placeholder_cls=_FallbackPlaceholderRange,
            )
            next_offset = mm_position.offset + mm_position.length
            features.append(
                _FallbackMultiModalFeatureSpec(
                    data={
                        "pixel_values": _FallbackMultiModalFieldElem(
                            payload["pixel_values"]
                        ),
                        "image_grid_thw": _FallbackMultiModalFieldElem(grid_row),
                    },
                    modality="image",
                    identifier=_identifier(payload, payload_idx, image_idx),
                    mm_position=mm_position,
                    mm_hash=_mm_hash(payload),
                )
            )
    return features


def _image_payloads(mm_data: Any) -> list[dict[str, Any]]:
    if mm_data is None:
        return []

    image_data = _get(mm_data, "image")
    if image_data is None and _get(mm_data, "pixel_values") is not None:
        image_data = mm_data
    if image_data is None:
        return []

    if _is_pixel_grid_pair(image_data):
        image_data = [image_data]
    elif isinstance(image_data, dict) and "pixel_values" in image_data:
        image_data = [image_data]

    payloads: list[dict[str, Any]] = []
    for item in image_data:
        if _is_pixel_grid_pair(item):
            pixel_values, image_grid_thw = item
            payloads.append(
                {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
            )
        else:
            payloads.append(
                {
                    "pixel_values": _get(item, "pixel_values"),
                    "image_grid_thw": _get(item, "image_grid_thw"),
                    "mm_position": _get(item, "mm_position"),
                    "mm_positions": _get(item, "mm_positions"),
                    "offset": _get(item, "offset"),
                    "identifier": _get(item, "identifier"),
                    "mm_hash": _get(item, "mm_hash"),
                }
            )

    return [
        payload
        for payload in payloads
        if payload["pixel_values"] is not None and payload["image_grid_thw"] is not None
    ]


def _is_pixel_grid_pair(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _grid_rows(image_grid_thw: Any) -> list[Any]:
    rows = _to_list(image_grid_thw)
    if not rows:
        return []
    if all(isinstance(x, int | float) for x in rows):
        return [rows]
    return rows


def _grid_prod(grid_row: Any) -> int:
    return int(reduce(mul, (int(x) for x in _to_list(grid_row)), 1))


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _tensor(value: list[int], *, like: Any) -> Any:
    try:
        import torch

        device = getattr(like, "device", None)
        return torch.as_tensor(value, device=device)
    except Exception:
        return value


def _tensor_data(value: Any) -> Any:
    try:
        import torch

        return torch.as_tensor(value)
    except Exception:
        return value


def _image_grid_tensor(value: Any) -> Any:
    tensor = _tensor_data(value)
    if hasattr(tensor, "ndim") and tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def _placeholder_range(
    payload: dict[str, Any],
    image_idx: int,
    *,
    default_offset: int,
    default_length: int,
    placeholder_cls: Any,
) -> Any:
    mm_position = _indexed(_get(payload, "mm_positions"), image_idx) or _get(
        payload, "mm_position"
    )
    if mm_position is not None:
        offset = _get(mm_position, "offset")
        length = _get(mm_position, "length")
        if offset is not None and length is not None:
            return placeholder_cls(offset=int(offset), length=int(length))

    offset = _indexed(_get(payload, "offset"), image_idx)
    if offset is None:
        offset = default_offset
    return placeholder_cls(offset=int(offset), length=default_length)


def _indexed(value: Any, idx: int) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return value[idx] if idx < len(value) else None
    return value


def _identifier(payload: dict[str, Any], payload_idx: int, image_idx: int) -> str:
    identifier = _indexed(_get(payload, "identifier"), image_idx)
    if identifier is not None:
        return str(identifier)
    return f"image-{payload_idx}-{image_idx}"


def _mm_hash(payload: dict[str, Any]) -> str | None:
    value = _get(payload, "mm_hash")
    return str(value) if value is not None else None


async def generate(
    *,
    client: AsyncOpenAI,
    renderer: Renderer | RendererPool,
    messages: list[Message],
    model: str,
    prompt_ids: list[int] | None = None,
    tools: list[ToolSpec] | None = None,
    sampling_params: dict[str, Any] | None = None,
    cache_salt: str | None = None,
    priority: int | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Tokenize messages, call vLLM /inference/v1/generate, parse the response.

    ``sampling_params`` is forwarded to vLLM verbatim. Two fields are always
    set by us and override caller values: ``stop_token_ids`` (from the
    renderer) and ``logprobs=1`` (we always emit completion_logprobs). Pass
    ``prompt_ids`` to skip rendering and use a prebuilt token sequence.

    Returns a dict with: request_id, prompt_ids, completion_ids,
    completion_logprobs, content, reasoning_content, tool_calls,
    finish_reason, routed_experts.
    """
    if tools and not getattr(renderer, "supports_tools", True):
        raise ValueError(
            f"{type(renderer).__name__} does not support tools. "
            "Choose a model-specific renderer instead of the default fallback."
        )

    pool = renderer if isinstance(renderer, RendererPool) else None

    def _prepare(r: Renderer):
        ids = (
            list(prompt_ids)
            if prompt_ids is not None
            else r.render_ids(messages, tools=tools, add_generation_prompt=True)
        )
        return ids, r.get_stop_token_ids()

    if pool is not None:
        prompt_ids, stop_token_ids = await _run_pooled(pool, _prepare)
    else:
        prompt_ids, stop_token_ids = _prepare(renderer)

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
    post_kwargs: dict[str, Any] = {
        "cast_to": cast(Any, dict[str, Any]),
        "body": body,
    }
    if extra_headers:
        post_kwargs["options"] = cast(Any, {"headers": extra_headers})
    try:
        data = await client.post(endpoint, **post_kwargs)
    except BadRequestError as exc:
        _log_overlong_prompt_diagnostic(
            prompt_ids=prompt_ids,
            messages=messages,
            max_tokens=sp.get("max_tokens"),
            exc=exc,
        )
        raise

    choice = (data.get("choices") or [{}])[0]
    completion_ids = choice.get("token_ids") or []

    if pool is not None:
        parsed = await _run_pooled(pool, lambda r: r.parse_response(completion_ids))
    else:
        parsed = renderer.parse_response(completion_ids)

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
    # when we extracted tool calls client-side, so OpenAI-compatible agent
    # loops continue past the tool turn instead of treating the response as
    # final.
    finish_reason = choice.get("finish_reason")
    if parsed.tool_calls and finish_reason == "stop":
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
    }


def _log_overlong_prompt_diagnostic(
    *,
    prompt_ids: list[int],
    messages: list[Message],
    max_tokens: int | None,
    exc: BadRequestError,
) -> None:
    """Log a structured snapshot when vLLM rejects with 4xx — usually overlong.

    Captures total prompt length, per-message role + character count, and
    the first chunk of the response body.
    """
    body_text = ""
    response = getattr(exc, "response", None)
    if response is not None:
        body_text = (response.text or "")[:500].replace("\n", " ")
    msg_summary = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str):
            content_len = len(content)
        elif isinstance(content, list):
            content_len = sum(
                len(p.get("text", "")) if isinstance(p, dict) else 0 for p in content
            )
        else:
            content_len = 0
        tool_calls = m.get("tool_calls")
        tc_count = len(tool_calls) if tool_calls else 0
        msg_summary.append(f"[{i}]{role}(c={content_len},tc={tc_count})")
    _request_logger.warning(
        "vllm 4xx prompt_len=%d messages=%d max_tokens=%s per_msg=%s response_body=%s",
        len(prompt_ids),
        len(messages),
        max_tokens,
        " ".join(msg_summary),
        body_text,
    )
