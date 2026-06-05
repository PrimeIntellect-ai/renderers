from __future__ import annotations

import numpy as np

import renderers._native_vision as _native_vision


class _FakeProcessor:
    def process_bytes(self, _raw: bytes):
        return {
            "modality": "image",
            "num_tokens": 2,
            "hash": "abc",
            "hf_payload": {
                "pixel_values": np.arange(6, dtype=np.float32).reshape(2, 3),
                "image_grid_thw": np.array([[1, 2, 4]], dtype=np.int64),
            },
        }


def test_process_image_for_qwen_vl_accepts_native_numpy_payload(monkeypatch):
    monkeypatch.setattr(
        _native_vision, "get_qwen_vl_processor", lambda **_kwargs: _FakeProcessor()
    )

    out = _native_vision.process_image_for_qwen_vl(b"image", message_idx=3)

    assert out["message_idx"] == 3
    assert out["hf_payload"]["pixel_values"].shape == (2, 3)
    assert out["hf_payload"]["image_grid_thw"].shape == (1, 3)


def test_process_image_for_qwen_vl_return_numpy_false_converts_to_dict(monkeypatch):
    monkeypatch.setattr(
        _native_vision, "get_qwen_vl_processor", lambda **_kwargs: _FakeProcessor()
    )

    out = _native_vision.process_image_for_qwen_vl(
        b"image", message_idx=3, return_numpy=False
    )

    assert out["hf_payload"]["pixel_values"] == {
        "shape": [2, 3],
        "data": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }
    assert out["hf_payload"]["image_grid_thw"] == {
        "shape": [1, 3],
        "data": [1, 2, 4],
    }
