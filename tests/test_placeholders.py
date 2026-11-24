"""Logical media placeholders sit where the renderer placed the media.

A placeholder token id elsewhere in the prompt — sampled by the model, or in
text — is not a placeholder: ``mm_placeholders`` records only the renderer's
own placements, and ``generate`` sends those positions to the engine.
"""

from functools import lru_cache

import pytest

from renderers import create_renderer
from tests.model_assets import load_test_tokenizer

pytestmark = pytest.mark.model

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


@lru_cache(maxsize=None)
def _load(model):
    tokenizer = load_test_tokenizer(model)
    return tokenizer, create_renderer(tokenizer)


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-VL-4B-Instruct"])
def test_placeholders_ignore_sampled_and_literal_pad_tokens(model):
    tokenizer, renderer = _load(model)
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    first = [{"role": "user", "content": [IMAGE, {"type": "text", "text": "a"}]}]
    rendered = renderer.render(
        first, add_generation_prompt=True, process_multimodal=False
    )
    [first_image] = rendered.multi_modal_data.mm_placeholders["image"]
    assert rendered.token_ids[first_image.offset] == pad

    # The policy samples the placeholder token; the next turn's text contains it.
    completion = tokenizer.encode("x<|image_pad|>y", add_special_tokens=False)
    completion.append(renderer.get_stop_token_ids()[0])
    new = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "see <|image_pad|> and"}, IMAGE],
        }
    ]
    bridged = renderer.bridge_to_next_turn(
        rendered.token_ids,
        completion,
        new,
        previous_multi_modal_data=rendered.multi_modal_data,
        process_multimodal=False,
    )
    assert bridged is not None
    placeholders = bridged.multi_modal_data.mm_placeholders["image"]
    assert placeholders[0] == first_image
    assert len(placeholders) == 2
    assert bridged.token_ids.count(pad) > 2
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    assert all(bridged.token_ids[p.offset - 1] == vision_start for p in placeholders)

    full = renderer.render(
        [
            *first,
            {"role": "assistant", "content": tokenizer.decode(completion[:-1])},
            *new,
        ],
        add_generation_prompt=True,
        process_multimodal=False,
    )
    full_placeholders = full.multi_modal_data.mm_placeholders["image"]
    assert len(full_placeholders) == 2
    assert all(full.token_ids[p.offset - 1] == vision_start for p in full_placeholders)
