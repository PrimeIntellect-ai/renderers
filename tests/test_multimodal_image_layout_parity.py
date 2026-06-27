"""Image-layout descriptor parity against real HF processors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from renderers.image_layout_specs import KIMI_K25_IMAGE_LAYOUT, QWEN_VL_IMAGE_LAYOUT
from renderers.kimi_k25 import describe_kimi_image_layout
from renderers.qwen3_vl import describe_qwen_image_layout

pytest.importorskip("PIL", reason="Pillow required for image layout parity tests")
pytest.importorskip("torch", reason="torch required for image layout parity tests")
pytest.importorskip(
    "transformers", reason="transformers required for image layout parity tests"
)

from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402


QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
KIMI_MODEL = "moonshotai/Kimi-K2.5"
KIMI_REVISION = "4d01dfe0332d63057c186e0b262165819efb6611"

IMAGE_SIZES = [(32, 32), (64, 256), (512, 512)]


def _hf_snapshot_cached(model_name: str) -> bool:
    cache = (
        Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
        / "hub"
    )
    snapshots = cache / ("models--" + model_name.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(p.is_dir() for p in snapshots.iterdir())


def _load_processor(model_name: str, **kwargs: Any):
    if not _hf_snapshot_cached(model_name):
        pytest.skip(f"{model_name}: HF snapshot not cached locally")
    return AutoProcessor.from_pretrained(model_name, **kwargs)


def _images():
    return [
        Image.new("RGB", size, color=(64 + idx * 32, 128, 192))
        for idx, size in enumerate(IMAGE_SIZES)
    ]


def _tensor_rows(value: Any) -> list[list[int]]:
    return [[int(cell) for cell in row] for row in value.tolist()]


def test_qwen_image_layout_descriptor_matches_processor():
    processor = _load_processor(QWEN_MODEL)
    images = _images()
    messages = [
        {
            "role": "user",
            "content": [
                item
                for idx, image in enumerate(images)
                for item in (
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"image {idx}"},
                )
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processor_grids = _tensor_rows(
        processor(images=images, text=text, return_tensors="pt")["image_grid_thw"]
    )
    descriptors = [
        describe_qwen_image_layout({"type": "image", "image": image})
        for image in images
    ]

    assert [desc.image_grid_thw[0] for desc in descriptors] == processor_grids
    merge_area = QWEN_VL_IMAGE_LAYOUT.merge_size**2
    assert [desc.num_image_tokens for desc in descriptors] == [
        grid_t * grid_h * grid_w // merge_area
        for grid_t, grid_h, grid_w in processor_grids
    ]


def test_kimi_image_layout_descriptor_matches_processor():
    processor = _load_processor(
        KIMI_MODEL, trust_remote_code=True, revision=KIMI_REVISION
    )
    images = _images()
    messages = [
        {
            "role": "user",
            "content": [
                item
                for idx, image in enumerate(images)
                for item in (
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": f"image {idx}"},
                )
            ],
        }
    ]

    out = processor(messages=messages, add_generation_prompt=True, return_tensors="pt")
    processor_grids = _tensor_rows(out["grid_thws"])
    descriptors = [
        describe_kimi_image_layout({"type": "image_url", "image_url": image})
        for image in images
    ]

    assert [desc.grid_thws[0] for desc in descriptors] == processor_grids
    merge_area = KIMI_K25_IMAGE_LAYOUT.merge_kernel_size**2
    assert [desc.num_media_tokens for desc in descriptors] == [
        grid_t * grid_h * grid_w // merge_area
        for grid_t, grid_h, grid_w in processor_grids
    ]
