"""Bridge helpers for the native Qwen-VL image processor.

The Rust pipeline in ``renderers_native.Qwen3VlImageProcessor`` produces
``{pixel_values, image_grid_thw, num_tokens, hash}`` dicts that match
what HF's ``Qwen3VLImageProcessor.preprocess(...)`` emits — same shapes,
same OpenAI CLIP normalisation, same patch layout. Pixel-byte parity
is approximate (CatmullRom vs PIL bicubic) but grid dims and token
counts are exact.

These helpers convert the dict shape into numpy arrays so the result
plugs into vLLM's ``MultiModalKwargsItem`` / SGLang's payload without
extra glue:

    from renderers._native_vision import process_image_for_qwen_vl
    media_item = process_image_for_qwen_vl(pil_or_bytes, message_idx=2)
    # media_item is the dict shape Renderer.render_with_media expects.
"""

from __future__ import annotations

import io
from typing import Any

try:
    import renderers_native  # type: ignore[import-not-found]
    _NATIVE = renderers_native
except ImportError:
    _NATIVE = None


_PROCESSOR_CACHE: dict[tuple[int, int, int, int, int], Any] = {}


def get_qwen_vl_processor(
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
):
    """Return a cached ``Qwen3VlImageProcessor`` with the given config.

    Raises ``RuntimeError`` if the native extension isn't built. The
    processor itself is cheap to construct (no model weights) so the
    cache here is just a courtesy — repeated calls with the same kwargs
    return the same handle.
    """
    if _NATIVE is None:
        raise RuntimeError(
            "renderers_native is not installed; build it with "
            "`maturin develop --manifest-path crates/renderers-py/Cargo.toml --release`"
        )
    key = (
        min_pixels if min_pixels is not None else 56 * 56,
        max_pixels if max_pixels is not None else 28 * 28 * 1280,
        patch_size,
        temporal_patch_size,
        merge_size,
    )
    cached = _PROCESSOR_CACHE.get(key)
    if cached is None:
        cached = _NATIVE.Qwen3VlImageProcessor(
            min_pixels=key[0],
            max_pixels=key[1],
            patch_size=key[2],
            temporal_patch_size=key[3],
            merge_size=key[4],
        )
        _PROCESSOR_CACHE[key] = cached
    return cached


def process_image_for_qwen_vl(
    image: Any,
    *,
    message_idx: int,
    return_numpy: bool = True,
    **processor_kwargs,
) -> dict[str, Any]:
    """Process a single image into the dict shape
    ``Renderer.render_with_media`` expects.

    Args:
        image: Either ``bytes`` (raw image data), a filesystem path, or
            a PIL ``Image.Image`` instance.
        message_idx: Index of the user message this image is attached
            to. Threaded into the returned dict so the caller can
            ``[*items]`` straight into ``render_with_media``.
        return_numpy: When True (default), unpack ``pixel_values`` and
            ``image_grid_thw`` into numpy arrays before returning. Set
            False to keep the lossless list-of-floats shape (useful for
            JSON serialisation).
        **processor_kwargs: Forwarded to
            ``get_qwen_vl_processor`` (``min_pixels`` / ``max_pixels`` /
            ``patch_size`` / ``temporal_patch_size`` / ``merge_size``).

    Returns:
        A dict shaped as
        ``{"message_idx", "modality", "num_tokens", "hash", "hf_payload"}``.
    """
    proc = get_qwen_vl_processor(**processor_kwargs)

    if isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
        out = proc.process_bytes(raw)
    elif isinstance(image, str):
        out = proc.process_path(image)
    else:
        # Treat as PIL Image — re-encode to PNG bytes.
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        out = proc.process_bytes(buf.getvalue())

    import numpy as np  # local to keep import cost off the hot path

    pv = out["hf_payload"]["pixel_values"]
    gt = out["hf_payload"]["image_grid_thw"]

    def _as_array(value, dtype):
        if isinstance(value, dict):
            return np.asarray(value["data"], dtype=dtype).reshape(tuple(value["shape"]))
        return np.asarray(value, dtype=dtype)

    pixel_values = _as_array(pv, np.float32)
    image_grid_thw = _as_array(gt, np.int64)

    if return_numpy:
        out["hf_payload"] = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
    else:
        out["hf_payload"] = {
            "pixel_values": {
                "shape": list(pixel_values.shape),
                "data": pixel_values.reshape(-1).tolist(),
            },
            "image_grid_thw": {
                "shape": list(image_grid_thw.shape),
                "data": image_grid_thw.reshape(-1).tolist(),
            },
        }

    out["message_idx"] = message_idx
    return out
