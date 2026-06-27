"""Static multimodal image layout contracts mirrored from model processors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenVLImageLayoutSpec:
    """Qwen-VL image processor values needed for raw descriptor layout math."""

    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    min_pixels: int = 65536
    max_pixels: int = 16777216


@dataclass(frozen=True)
class KimiK25ImageLayoutSpec:
    """Kimi K2.5 image processor values needed for raw descriptor layout math."""

    patch_size: int = 14
    merge_kernel_size: int = 2
    in_patch_limit: int = 16384
    patch_limit_on_one_side: int = 512
    fixed_output_tokens: int | None = None
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


QWEN_VL_IMAGE_LAYOUT = QwenVLImageLayoutSpec()
KIMI_K25_IMAGE_LAYOUT = KimiK25ImageLayoutSpec()
