"""NumPy fast-path coverage for the native PyO3 module."""

from __future__ import annotations

import os

import numpy as np
import pytest

from renderers import _native_router as router


@pytest.fixture(scope="module")
def qwen3_native():
    native = router.load_native()
    if native is None:
        pytest.skip("renderers_native not built; run `maturin develop`")

    try:
        from renderers.base import load_tokenizer

        tokenizer = load_tokenizer("Qwen/Qwen3-8B")
        tok_path = router.resolve_tokenizer_path(tokenizer)
    except Exception as exc:
        pytest.skip(f"could not resolve Qwen3 tokenizer: {exc}")
    if not os.path.exists(tok_path):
        pytest.skip(f"tokenizer.json missing on disk at {tok_path}")

    return native.Renderer.qwen3(tok_path)


def test_render_ids_np_matches_list_api(qwen3_native):
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hi."},
    ]

    ids = qwen3_native.render_ids_np(messages, add_generation_prompt=True)

    assert ids.dtype == np.uint32
    assert ids.tolist() == qwen3_native.render_ids(
        messages,
        add_generation_prompt=True,
    )


def test_parse_response_np_borrows_uint32_completion(qwen3_native):
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    assistant = {"role": "assistant", "content": "4"}
    prompt_ids = qwen3_native.render_ids_np(prompt, add_generation_prompt=True)
    full_ids = qwen3_native.render_ids_np(prompt + [assistant])
    completion_ids = full_ids[len(prompt_ids) :]

    parsed = qwen3_native.parse_response_np(completion_ids)

    assert parsed.content == "4"


def test_bridge_to_next_turn_np_matches_list_api(qwen3_native):
    prompt = [{"role": "user", "content": "Plan Saturday."}]
    assistant = {"role": "assistant", "content": "Start with breakfast."}
    new_messages = [{"role": "user", "content": "Add one museum."}]

    prompt_ids = qwen3_native.render_ids_np(prompt, add_generation_prompt=True)
    full_ids = qwen3_native.render_ids_np(prompt + [assistant])
    completion_ids = full_ids[len(prompt_ids) :]

    bridged_np = qwen3_native.bridge_to_next_turn_np(
        prompt_ids,
        completion_ids,
        new_messages,
    )
    bridged_list = qwen3_native.bridge_to_next_turn(
        prompt_ids.tolist(),
        completion_ids.tolist(),
        new_messages,
    )

    assert bridged_np is not None
    assert bridged_list is not None
    assert bridged_np.dtype == np.uint32
    assert bridged_np.tolist() == bridged_list.token_ids
