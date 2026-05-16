import pytest

from renderers.qwen3_vl import (
    Qwen3VLRenderer,
    _is_image_part,
    _is_video_part,
    _load_pil_image,
)


class _FakeTokenizer:
    unk_token_id = 0

    _special_tokens = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|endoftext|>": 3,
        "<tool_call>": 4,
        "</tool_call>": 5,
        "<tool_response>": 6,
        "</tool_response>": 7,
    }

    def convert_tokens_to_ids(self, token):
        return self._special_tokens.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_qwen3_vl_rejects_arrow_unified_none_media_keys():
    content = [
        {
            "type": "text",
            "text": "hello",
            "image": None,
            "image_url": None,
            "video": None,
            "video_url": None,
        },
        {
            "type": "image_url",
            "text": None,
            "image": None,
            "image_url": {
                "url": "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
                "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
            },
            "video": None,
            "video_url": None,
        },
    ]

    renderer = Qwen3VLRenderer(_FakeTokenizer())

    assert _is_image_part(content[0]) is False
    assert _is_image_part(content[1]) is True
    assert _is_video_part(content[0]) is False
    renderer.render([{"role": "user", "content": content}])


def test_qwen3_vl_untyped_media_fallback_requires_truthy_payload():
    assert _is_image_part({"type": "text", "image_url": {"url": "x"}}) is False
    assert _is_image_part({"image_url": None}) is False
    assert _is_image_part({"image_url": {"url": "data:image/png;base64,abc"}}) is True
    assert _is_video_part({"type": "text", "video_url": "file.mp4"}) is False
    assert _is_video_part({"video_url": None}) is False
    assert _is_video_part({"video_url": "file.mp4"}) is True


def test_qwen3_vl_load_pil_image_reports_non_image_part():
    with pytest.raises(TypeError, match="Expected image content part"):
        _load_pil_image({"type": "text", "text": "hello", "image_url": None})
