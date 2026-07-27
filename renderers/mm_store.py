"""Inline image helpers for multimodal rendering.

The default renderer multimodal mode does not ship processed image features.
Messages carry ``data:image/...;base64`` URLs inline, and renderers emit
lightweight image refs that embed the inline source, so the inference engine
(and later the trainer) can materialize pixels wherever they are consumed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass

IMAGE_REF_PREFIX = "mmraw"
RAW_MM_ITEM_KIND = "prime_raw_mm_item"

_SAFE = {
    "multimodal family": re.compile(r"^[A-Za-z0-9_.-]+$"),
    "raw multimodal modality": re.compile(r"^[A-Za-z0-9_.-]+$"),
    "image layout fingerprint": re.compile(r"^[a-f0-9]{16,64}$"),
    "image hash": re.compile(r"^[a-f0-9]{16,128}$"),
}


def _ensure_safe(label: str, value: str) -> str:
    if not _SAFE[label].fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def decode_data_image_url(url: object) -> bytes:
    """Decode a ``data:image/...;base64`` URL to raw image bytes.

    Raises for anything that isn't a base64 data-image URL — inline raw
    multimodal rendering has no other supported image source.
    """
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError(
            "inline raw multimodal rendering requires data:image/...;base64 sources, "
            f"got {type(url).__name__ if not isinstance(url, str) else url[:64]!r}"
        )
    marker = ";base64,"
    if marker not in url:
        raise ValueError("data image URL must be base64-encoded (missing ';base64,')")
    header, b64 = url.split(marker, 1)
    try:
        return base64.b64decode(b64)
    except Exception as exc:
        raise ValueError(f"Undecodable base64 data in {header!r} image URL") from exc


def _json_fingerprint_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _encode_ref_payload(payload: dict[str, object] | None) -> str:
    """Serialize a ref payload as compact JSON.

    Deliberately not base64-wrapped: the ref travels as a string inside a JSON
    request body, so the only encoding cost is escaping the payload's own quotes
    (tens of bytes). A base64 wrapper would instead inflate the whole payload by
    a third — including an inline image source, where that is ~34 KiB per image.
    """
    return json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))


def _decode_ref_payload(encoded: str) -> dict[str, object]:
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("Raw multimodal ref payload must decode to a dict")
    return payload


def image_layout_fingerprint(*, family: str, **values: object) -> str:
    """Stable adapter-owned fingerprint for raw multimodal layout contracts."""
    _ensure_safe("multimodal family", family)
    encoded_values = ":".join(
        f"{key}={_json_fingerprint_value(values[key])}" for key in sorted(values)
    )
    raw = f"image-layout:{family}:{encoded_values}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def raw_mm_item(
    *,
    modality: str,
    family: str,
    layout_fingerprint: str,
    payload: dict[str, object],
    raw_image_data: str,
    vllm_modality: str | None = None,
) -> dict[str, object]:
    """Build the JSON-safe raw multimodal descriptor envelope.

    ``payload`` is intentionally adapter-owned. Shared consumers may route by
    ``family`` and validate the common envelope, but must not inspect adapter
    payload keys.
    """
    _ensure_safe("multimodal family", family)
    _ensure_safe("raw multimodal modality", modality)
    _ensure_safe("image layout fingerprint", layout_fingerprint)
    out: dict[str, object] = {
        "kind": RAW_MM_ITEM_KIND,
        "modality": modality,
        "family": family,
        "layout_fingerprint": layout_fingerprint,
        "payload": payload,
    }
    if vllm_modality is not None:
        out["vllm_modality"] = vllm_modality
    out["raw_image_data"] = raw_image_data
    return out


@dataclass(frozen=True)
class RawMMRef:
    family: str
    fingerprint: str
    modality: str
    mm_hash: str
    payload: dict[str, object]
    raw_image_data: str


def raw_mm_ref(
    *,
    family: str,
    fingerprint: str,
    modality: str,
    mm_hash: str,
    raw_image_data: str,
    payload: dict[str, object] | None = None,
) -> str:
    """Generic raw multimodal asset ref.

    Adapter-owned details stay in the descriptor payload so refs can serve
    future families without baking shape names into the wire id.
    """
    _ensure_safe("multimodal family", family)
    _ensure_safe("image layout fingerprint", fingerprint)
    _ensure_safe("raw multimodal modality", modality)
    _ensure_safe("image hash", mm_hash)

    ref_payload: dict[str, object] = {
        "family": family,
        "fingerprint": fingerprint,
        "modality": modality,
        "mm_hash": mm_hash,
        "payload": payload or {},
        "raw_image_data": raw_image_data,
    }

    return f"{IMAGE_REF_PREFIX}:{_encode_ref_payload(ref_payload)}"


def split_raw_mm_ref(ref: str) -> RawMMRef:
    prefix, _, encoded = ref.partition(":")
    if prefix != IMAGE_REF_PREFIX or not encoded:
        raise ValueError(f"Invalid raw multimodal ref shape: {ref!r}")

    payload = _decode_ref_payload(encoded)
    family = payload.get("family")
    fingerprint = payload.get("fingerprint")
    modality = payload.get("modality")
    mm_hash = payload.get("mm_hash")
    raw_image_data = payload.get("raw_image_data")
    item_payload = payload.get("payload")

    if not isinstance(family, str):
        raise ValueError("Raw multimodal ref is missing family")
    if not isinstance(fingerprint, str):
        raise ValueError("Raw multimodal ref is missing fingerprint")
    if not isinstance(modality, str):
        raise ValueError("Raw multimodal ref is missing modality")
    if not isinstance(mm_hash, str):
        raise ValueError("Raw multimodal ref is missing mm_hash")
    if not isinstance(raw_image_data, str):
        raise ValueError("Raw multimodal ref is missing raw_image_data")
    if not isinstance(item_payload, dict):
        raise ValueError("Raw multimodal ref payload must be a dict")

    return RawMMRef(
        family=_ensure_safe("multimodal family", family),
        fingerprint=_ensure_safe("image layout fingerprint", fingerprint),
        modality=_ensure_safe("raw multimodal modality", modality),
        mm_hash=_ensure_safe("image hash", mm_hash),
        payload=item_payload,
        raw_image_data=raw_image_data,
    )


def is_raw_mm_ref(ref: object) -> bool:
    return isinstance(ref, str) and ref.startswith(f"{IMAGE_REF_PREFIX}:")
