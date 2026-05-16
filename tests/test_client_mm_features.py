from renderers.client import _build_mm_features
from renderers.qwen35 import Qwen35Renderer


def test_build_mm_features_dispatches_qwen35_renderer():
    features = _build_mm_features(
        Qwen35Renderer,
        {
            "image": {
                "pixel_values": [[1.0], [2.0], [3.0], [4.0]],
                "image_grid_thw": [1, 4, 4],
                "offset": 7,
                "identifier": "image-0",
            }
        },
    )

    assert features is not None
    assert len(features) == 1
    feature = features[0]
    assert feature.modality == "image"
    assert feature.identifier == "image-0"
    assert feature.mm_position.offset == 7
    assert feature.mm_position.length == 4
    assert set(feature.data) == {"pixel_values", "image_grid_thw"}
