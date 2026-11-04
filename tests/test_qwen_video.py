"""Focused Qwen3-VL-family video coverage against vLLM 0.32's prompt layout.

``test_multimodal.py`` checks videos against HF's processor. These tests pin
what vLLM's Qwen3-VL processor builds (``Qwen3VLMultiModalProcessor``), which
is what the RL path trains on when vLLM expands a ``video_url`` itself:

    <|vision_start|> { <t seconds> <|vision_start|> N x <|video_pad|> <|vision_end|> } <|vision_end|>

one frame group per frame pair, with one placeholder per video spanning the
frame groups. They also cover the deferred (``process_multimodal=False``)
path and the OpenCV decoding that mirrors vLLM's video loader.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from renderers import (
    Qwen38RendererConfig,
    Qwen3VLRendererConfig,
    build_training_sample,
    create_renderer,
)
from renderers.qwen3_vl import _load_video, _video_timestamps

# A 4 s, 512x288, 10 fps clip as vLLM's Qwen3-VL loader samples it: 2 fps,
# so 8 of its 40 frames.
CLIP_METADATA = {
    "total_num_frames": 40,
    "fps": 10.0,
    "duration": 4.0,
    "video_backend": "opencv",
    "frames_indices": [0, 6, 11, 17, 22, 28, 33, 39],
    "do_sample_frames": False,
}

# vLLM 0.32.0's ``prompt_token_ids`` for that clip with the Qwen3.8-Flash-Next
# tokenizer: ``<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>
# Describe.<|im_end|>\n`` with the video expanded to 4 frame groups of 144
# pads (grid ``[4, 18, 32]``), and its video placeholder ``(4, 608)``.
_VLLM_TIMESTAMPS = [  # <0.3 seconds> <1.4 seconds> <2.5 seconds> <3.6 seconds>
    [27, 15, 13, 18, 6283, 29],
    [27, 16, 13, 19, 6283, 29],
    [27, 17, 13, 20, 6283, 29],
    [27, 18, 13, 21, 6283, 29],
]
_VLLM_PROMPT = [
    248045,
    846,
    198,
    248053,
    *[
        token
        for timestamp in _VLLM_TIMESTAMPS
        for token in (*timestamp, 248053, *[248057] * 144, 248054)
    ],
    248054,
    72240,
    13,
    248046,
    198,
]
_VLLM_PLACEHOLDER = (4, 608)

_MODEL_CONFIGS = {
    # Qwen35Renderer family, without the reasoning-effort system prompt.
    "Qwen/Qwen3.8-Flash-Next": Qwen38RendererConfig(enable_thinking=False),
    "Qwen/Qwen3-VL-4B-Instruct": Qwen3VLRendererConfig(),
}

_loaded: dict[str, tuple] = {}


def _load(model_name: str, **config_updates):
    from tests.model_assets import load_test_processor, load_test_tokenizer

    if model_name not in _loaded:
        _loaded[model_name] = (
            load_test_tokenizer(model_name),
            load_test_processor(model_name),
        )
    tokenizer, processor = _loaded[model_name]
    config = _MODEL_CONFIGS[model_name].model_copy(update=config_updates)
    renderer = create_renderer(tokenizer, config)
    renderer._processor = processor
    return tokenizer, renderer


def _clip(height: int = 288, width: int = 512):
    frames = np.zeros((8, height, width, 3), dtype=np.uint8)
    frames[..., 0] = np.arange(8, dtype=np.uint8)[:, None, None] * 30
    return frames, dict(CLIP_METADATA)


def _video_message(video, text="Describe."):
    return {
        "role": "user",
        "content": [{"type": "video", "video": video}, {"type": "text", "text": text}],
    }


@pytest.mark.model
@pytest.mark.multimodal
def test_video_prompt_matches_vllm():
    """The measured clip renders to vLLM's prompt and placeholder exactly."""
    tokenizer, renderer = _load("Qwen/Qwen3.8-Flash-Next")
    rendered = renderer.render([_video_message(_clip())])

    assert rendered.token_ids == _VLLM_PROMPT
    mm = rendered.multi_modal_data
    assert mm is not None
    (placeholder,) = mm.mm_placeholders["video"]
    assert (placeholder.offset, placeholder.length) == _VLLM_PLACEHOLDER
    assert placeholder.is_embed is not None and sum(placeholder.is_embed) == 576
    (item,) = mm.mm_items["video"]
    assert item["video_grid_thw"].tolist() == [[4, 18, 32]]
    assert item["pixel_values_videos"].shape[0] == 4 * 18 * 32
    assert len(mm.mm_hashes["video"]) == 1


@pytest.mark.model
@pytest.mark.multimodal
@pytest.mark.parametrize("video_model", sorted(_MODEL_CONFIGS))
def test_video_expansion_follows_vllm_get_video_repl(video_model):
    """Per frame pair: ``<t seconds>`` tokenized on its own, then
    ``<|vision_start|>``, the pads and ``<|vision_end|>``, inside the
    template's own vision markers. Only the pads are body content."""
    tokenizer, renderer = _load(video_model)
    rendered = renderer.render(
        [_video_message(_clip(64, 96), "What happens?")], add_generation_prompt=True
    )
    ids = rendered.token_ids
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    video_pad = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    expected: list[int] = []
    for seconds in ("0.3", "1.4", "2.5", "3.6"):
        expected += tokenizer.encode(f"<{seconds} seconds>", add_special_tokens=False)
        expected += [vision_start, *[video_pad] * 6, vision_end]  # grid [4, 4, 6]
    (placeholder,) = rendered.multi_modal_data.mm_placeholders["video"]
    start, end = placeholder.offset, placeholder.offset + placeholder.length
    assert ids[start - 1] == vision_start and ids[end] == vision_end
    assert ids[start:end] == expected
    assert placeholder.is_embed == [token == video_pad for token in expected]
    assert ids.count(video_pad) == 24

    content = rendered.is_content[start:end]
    assert content == [token == video_pad for token in expected]


@pytest.mark.model
@pytest.mark.multimodal
@pytest.mark.parametrize("video_model", sorted(_MODEL_CONFIGS))
def test_video_mm_token_type_ids_mark_only_pads(video_model):
    """``build_training_sample`` flags the pads as video (2), not the frame
    timestamps or vision markers inside the video's placeholder."""
    tokenizer, renderer = _load(video_model)
    video_pad = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    sample = build_training_sample(
        renderer,
        [
            _video_message(_clip(64, 96)),
            {"role": "assistant", "content": "A red fade."},
        ],
        role_to_mask=lambda message: message["role"] == "assistant",
    )
    assert sample.mm_token_type_ids == [
        2 if token == video_pad else 0 for token in sample.token_ids
    ]


@pytest.mark.model
@pytest.mark.multimodal
@pytest.mark.parametrize("video_model", sorted(_MODEL_CONFIGS))
def test_deferred_video_leaves_one_pad_for_vllm(video_model):
    """With ``process_multimodal=False`` the template's placeholder goes out
    as is, for vLLM to expand from the raw ``video_url``, and the bridge
    numbers videos from the logical prompt's pads."""
    tokenizer, renderer = _load(video_model, add_vision_id=True)
    url = "data:video/mp4;base64,AAAA"
    first = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": url}},
                {"type": "text", "text": "Turn one."},
            ],
        }
    ]
    rendered = renderer.render(
        first, add_generation_prompt=True, process_multimodal=False
    )
    assert rendered.multi_modal_data is None
    text = tokenizer.decode(rendered.token_ids)
    assert "Video 1: <|vision_start|><|video_pad|><|vision_end|>Turn one." in text

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    response = "</think>Saw it." if text.rstrip().endswith("<think>") else "Saw it."
    completion = tokenizer.encode(response, add_special_tokens=False) + [im_end]
    second = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": url}},
                {"type": "text", "text": "Turn two."},
            ],
        }
    ]
    bridged = renderer.bridge_to_next_turn(
        rendered.token_ids, completion, second, process_multimodal=False
    )
    assert bridged is not None and bridged.multi_modal_data is None
    extension = tokenizer.decode(bridged.token_ids[len(rendered.token_ids) :])
    assert "Video 2: <|vision_start|><|video_pad|><|vision_end|>Turn two." in extension


def test_video_timestamps_follow_vllm():
    """``Qwen3VLProcessingInfo._get_video_second_idx``: mean seconds of each
    frame pair; when the HF processor samples frames itself, its indices."""
    video_processor = SimpleNamespace(
        fps=2, min_frames=4, max_frames=768, temporal_patch_size=2
    )
    timestamps = _video_timestamps(CLIP_METADATA, video_processor)
    assert [f"{t:.1f}" for t in timestamps] == ["0.3", "1.4", "2.5", "3.6"]

    # A 3-frame clip keeps every frame, so HF samples them: [0, 1, 2], padded
    # to a whole frame pair by repeating the last one.
    short = {
        "total_num_frames": 3,
        "fps": 10.0,
        "frames_indices": [0, 1, 2],
        "do_sample_frames": True,
    }
    assert _video_timestamps(short, video_processor) == pytest.approx([0.05, 0.2])


def test_video_url_decodes_like_vllm(tmp_path):
    """Raw sources decode as vLLM's Qwen3-VL loader decodes them: OpenCV,
    2 frames per second, ``linspace(...).round()`` indices."""
    cv2 = pytest.importorskip("cv2", reason="OpenCV decodes raw video sources")
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot encode mp4v")
    for i in range(40):
        writer.write(np.full((64, 96, 3), i * 6, dtype=np.uint8))
    writer.release()
    data = path.read_bytes()

    url = "data:video/mp4;base64," + base64.b64encode(data).decode()
    frames, metadata = _load_video({"type": "video_url", "video_url": {"url": url}})
    assert frames.shape == (8, 64, 96, 3) and frames.dtype == np.uint8
    assert metadata == {**CLIP_METADATA}

    from_path, path_metadata = _load_video({"type": "video", "video": str(path)})
    assert np.array_equal(from_path, frames) and path_metadata == metadata


def teardown_module():
    """Release model resources when this module finishes, not at process exit."""
    _loaded.clear()
