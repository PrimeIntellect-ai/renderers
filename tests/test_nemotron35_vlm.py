"""Focused multimodal tests for the private Nemotron 3.5 Super checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from renderers.base import MODEL_RENDERER_MAP, MULTIMODAL_MODELS, PlaceholderRange
from renderers.nemotron3 import Nemotron35Renderer


MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026"


class _Tokenizer:
    name_or_path = MODEL
    unk_token_id = -1
    eos_token_id = 2

    _specials = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|endoftext|>": 3,
        "<think>": 4,
        "</think>": 5,
        "<tool_call>": 6,
        "</tool_call>": 7,
        "<tool_response>": 8,
        "</tool_response>": 9,
        "<img>": 10,
        "</img>": 11,
        "<image>": 18,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._specials.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        token_ids: list[int] = []
        index = 0
        specials = sorted(self._specials, key=len, reverse=True)
        while index < len(text):
            special = next(
                (value for value in specials if text.startswith(value, index)), None
            )
            if special is not None:
                token_ids.append(self._specials[special])
                index += len(special)
            else:
                token_ids.append(1000 + ord(text[index]))
                index += 1
        return token_ids

    def decode(self, token_ids: list[int], **_: Any) -> str:
        reverse_specials = {value: key for key, value in self._specials.items()}
        return "".join(
            reverse_specials[token_id]
            if token_id in reverse_specials
            else chr(token_id - 1000)
            for token_id in token_ids
        )


class _ImageProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, images: list[Any], return_tensors: str):
        assert len(images) == 1
        assert return_tensors == "np"
        self.calls += 1
        return {
            "pixel_values": [[[[0.0]]]],
            "num_tokens": [3],
            "num_patches": [1],
            "imgs_sizes": [(1, 1)],
        }


def test_private_super_is_registered_as_image_model():
    assert MODEL_RENDERER_MAP[MODEL] == "nemotron-3.5"
    assert MULTIMODAL_MODELS[MODEL] == {"image"}


def test_image_content_expands_and_attaches_only_model_inputs():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (4, 3), color=(10, 20, 30))
    image_processor = _ImageProcessor()
    processor = SimpleNamespace(image_processor=image_processor)
    renderer = Nemotron35Renderer(_Tokenizer(), processor=processor)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown? "},
                {"type": "image", "image": image},
                {"type": "text", "text": " Be concise."},
            ],
        }
    ]

    rendered = renderer.render(messages, add_generation_prompt=True)

    assert messages[0]["content"][1]["image"] is image
    assert image_processor.calls == 1
    assert renderer.mm_token_type_id_map == {18: 1}
    assert rendered.multi_modal_data is not None
    assert rendered.multi_modal_data.mm_hashes.keys() == {"image"}
    assert len(rendered.multi_modal_data.mm_hashes["image"]) == 1
    placeholder = rendered.multi_modal_data.mm_placeholders["image"]
    assert placeholder == [PlaceholderRange(offset=placeholder[0].offset, length=3)]
    assert rendered.token_ids[placeholder[0].offset : placeholder[0].offset + 3] == [
        18,
        18,
        18,
    ]
    assert rendered.multi_modal_data.mm_items == {
        "image": [
            {
                "pixel_values": [[[[0.0]]]],
                "num_tokens": [3],
                "num_patches": [1],
                "imgs_sizes": [(1, 1)],
            }
        ]
    }


def test_deferred_multimodal_processing_keeps_one_logical_image_token():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (2, 2))
    renderer = Nemotron35Renderer(
        _Tokenizer(), processor=SimpleNamespace(image_processor=_ImageProcessor())
    )

    rendered = renderer.render(
        [{"role": "user", "content": [{"type": "image", "image": image}]}],
        process_multimodal=False,
    )

    assert rendered.token_ids.count(18) == 1
    assert rendered.multi_modal_data is None


def test_two_images_produce_two_distinct_placeholder_ranges():
    Image = pytest.importorskip("PIL.Image")
    renderer = Nemotron35Renderer(
        _Tokenizer(), processor=SimpleNamespace(image_processor=_ImageProcessor())
    )

    rendered = renderer.render(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (2, 2))},
                    {"type": "text", "text": "compare"},
                    {"type": "image", "image": Image.new("RGB", (3, 2))},
                ],
            }
        ]
    )

    assert rendered.multi_modal_data is not None
    placeholders = rendered.multi_modal_data.mm_placeholders["image"]
    assert [placeholder.length for placeholder in placeholders] == [3, 3]
    assert placeholders[0].offset + placeholders[0].length < placeholders[1].offset
    assert len(rendered.multi_modal_data.mm_items["image"]) == 2


def test_multimodal_bridge_uses_full_rerender_fallback():
    renderer = Nemotron35Renderer(_Tokenizer())

    assert not renderer.supports_multimodal_bridge
    assert (
        renderer.bridge_to_next_turn(
            [1],
            [2],
            [{"role": "user", "content": [{"type": "image", "image": object()}]}],
        )
        is None
    )


def test_text_only_nemotron35_tokenizer_does_not_need_image_token():
    class _TextTokenizer(_Tokenizer):
        _specials = {
            key: value
            for key, value in _Tokenizer._specials.items()
            if key != "<image>"
        }

    renderer = Nemotron35Renderer(_TextTokenizer())

    assert renderer.mm_token_type_id_map == {}
    assert renderer.render_ids([{"role": "user", "content": "Hello"}])


@pytest.mark.parametrize(
    "source",
    [
        "https://example.test/image.png",
        "file:///tmp/image.png",
        "/tmp/image.png",
    ],
)
def test_nemotron35_rejects_network_and_filesystem_image_sources(source):
    renderer = Nemotron35Renderer(
        _Tokenizer(), processor=SimpleNamespace(image_processor=_ImageProcessor())
    )

    with pytest.raises(ValueError, match="in-memory images or data:image"):
        renderer.render(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": source}}],
                }
            ]
        )


def test_nemotron35_rejects_unpinned_remote_processor(monkeypatch):
    class _RemoteTokenizer(_Tokenizer):
        name_or_path = "example/unreviewed-model"

    monkeypatch.setattr(
        "renderers.nemotron3._require_transformers",
        lambda _: pytest.fail(
            "unknown remote model must fail before transformers load"
        ),
    )
    renderer = Nemotron35Renderer(_RemoteTokenizer())

    with pytest.raises(RuntimeError, match="not approved for remote-code loading"):
        renderer._get_processor()


def test_nemotron35_rejects_unapproved_local_processor(tmp_path, monkeypatch):
    class _LocalTokenizer(_Tokenizer):
        name_or_path = str(tmp_path)

    monkeypatch.setattr(
        "renderers.nemotron3._require_transformers",
        lambda _: pytest.fail("unapproved local model must fail before load"),
    )

    with pytest.raises(RuntimeError, match="explicitly injected reviewed processor"):
        Nemotron35Renderer(_LocalTokenizer())._get_processor()


def test_nemotron35_loads_approved_remote_processor_without_revision(monkeypatch):
    calls = []
    expected = object()
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(
            from_pretrained=lambda name, **kwargs: (
                calls.append((name, kwargs)) or expected
            )
        )
    )
    monkeypatch.setattr(
        "renderers.nemotron3._require_transformers", lambda _: transformers
    )

    assert Nemotron35Renderer(_Tokenizer())._get_processor() is expected
    assert calls == [
        (
            MODEL,
            {"trust_remote_code": True},
        )
    ]


def test_real_processor_token_parity_when_metadata_is_available():
    metadata = os.environ.get("NEMOTRON35_VLM_METADATA")
    if metadata is None:
        pytest.skip("set NEMOTRON35_VLM_METADATA to the downloaded model metadata")
    metadata_path = Path(metadata)
    if not metadata_path.is_dir():
        pytest.skip(f"Nemotron metadata directory does not exist: {metadata_path}")

    Image = pytest.importorskip("PIL.Image")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(metadata_path)
    processor = transformers.AutoProcessor.from_pretrained(
        metadata_path, trust_remote_code=True
    )
    renderer = Nemotron35Renderer(tokenizer, processor=processor)
    images = [
        Image.new("RGB", (2048, 256), color=(10, 20, 30)),
        Image.new("RGB", (256, 2048), color=(30, 20, 10)),
    ]
    for selected_images in (images[:1], images):
        content = [{"type": "text", "text": "Describe: "}]
        content.extend({"type": "image", "image": image} for image in selected_images)
        messages = [{"role": "user", "content": content}]

        rendered = renderer.render(messages, add_generation_prompt=True)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        expected = processor(
            images=selected_images,
            text=prompt,
            return_tensors=None,
        )["input_ids"][0]

        assert rendered.token_ids == expected
        assert rendered.multi_modal_data is not None
        assert len(rendered.multi_modal_data.mm_items["image"]) == len(selected_images)
        if len(selected_images) == 2:
            shapes = [
                tuple(item["pixel_values"].shape)
                for item in rendered.multi_modal_data.mm_items["image"]
            ]
            assert shapes[0] != shapes[1]
