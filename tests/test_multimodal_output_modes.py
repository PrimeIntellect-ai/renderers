import numpy as np
import pytest

from renderers.kimi_k25 import kimi_processed_image_item_for_render
from renderers.qwen3_vl import qwen_processed_image_item_for_render


def _tiny_image_path(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "tiny.png"
    Image.new("RGB", (16, 16), color=(120, 80, 40)).save(path)
    return path


def test_qwen_processed_image_item_emits_processor_payload(tmp_path):
    class _ImageProcessor:
        merge_size = 2

        def __call__(self, images, return_tensors):
            assert len(images) == 1
            assert return_tensors == "np"
            return {
                "pixel_values": np.ones((4, 3), dtype=np.float32),
                "image_grid_thw": np.array([[1, 4, 4]], dtype=np.int64),
            }

    class _Processor:
        image_processor = _ImageProcessor()

    num_tokens, image_hash, item = qwen_processed_image_item_for_render(
        {"type": "image", "image": str(_tiny_image_path(tmp_path))},
        processor=_Processor(),
        image_cache={},
    )

    assert num_tokens == 4
    assert len(image_hash) == 32
    assert set(item) == {"pixel_values", "image_grid_thw"}
    assert item["pixel_values"].shape == (4, 3)
    assert item["image_grid_thw"].tolist() == [[1, 4, 4]]


def test_kimi_processed_image_item_emits_processor_payload(tmp_path):
    class _ImageProcessor:
        def preprocess(self, media, return_tensors):
            assert len(media) == 1
            assert media[0]["type"] == "image"
            assert return_tensors == "np"
            return {
                "pixel_values": np.ones((2, 3), dtype=np.float32),
                "grid_thws": np.array([[1, 2, 2]], dtype=np.int64),
            }

        def media_tokens_calculator(self, media):
            assert media["type"] == "image"
            return 2

    class _Processor:
        image_processor = _ImageProcessor()

    placeholder_len, image_hash, item = kimi_processed_image_item_for_render(
        {"type": "image", "image": str(_tiny_image_path(tmp_path))},
        processor=_Processor(),
        image_cache={},
    )

    assert placeholder_len == 1
    assert len(image_hash) == 32
    assert set(item) == {"pixel_values", "grid_thws"}
    assert item["pixel_values"].shape == (2, 3)
    assert item["grid_thws"].tolist() == [[1, 2, 2]]
