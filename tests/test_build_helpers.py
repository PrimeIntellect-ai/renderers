"""Barrage test: build_training_sample and build_trajectory_step.

Runs against every (model, renderer) pair.
"""

import numpy as np
import pytest

from renderers import build_training_sample, build_trajectory_step
from renderers.base import _build_mm_token_type_ids
from renderers.token_arrays import FixedWidthRangeBuilder
from tests.reference_rendering import render_reference


def test_build_mm_token_type_ids_marks_ranges():
    """Image runs → 1, video runs → 2, everything else → 0; clips at length."""
    image_ranges = FixedWidthRangeBuilder()
    image_ranges.append(2, 3)
    video_ranges = FixedWidthRangeBuilder()
    video_ranges.append(7, 2)
    placeholders = {"image": image_ranges.finish(), "video": video_ranges.finish()}
    ids = _build_mm_token_type_ids(placeholders, length=10)
    assert np.array_equal(
        ids, np.fromiter((0, 0, 1, 1, 1, 0, 0, 2, 2, 0), dtype=ids.dtype, count=10)
    )


def _expected(tokenizer, messages, **kwargs):
    return render_reference(tokenizer, messages, **kwargs)


def test_build_training_sample_ids_match(model_name, tokenizer, renderer):
    """Token IDs must match the model-aware reference renderer."""
    if (
        model_name in {"google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it"}
        and not renderer.config.enable_thinking
    ):
        pytest.skip(
            "Gemma 4 26B/31B deliberately keeps the disabled-thinking prefill "
            "on assistant history; stability is covered separately"
        )
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    sample = build_training_sample(
        renderer, msgs, role_to_mask=lambda m: m["role"] == "assistant"
    )
    ids = sample.token_ids
    assert np.array_equal(ids, _expected(tokenizer, msgs))
    # text-only sample carries no multimodal payload
    assert sample.multi_modal_data is None
    assert sample.mm_token_type_ids is None


def test_build_training_sample_has_trainable_tokens(model_name, tokenizer, renderer):
    """At least some tokens should be marked for training."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    sample = build_training_sample(
        renderer, msgs, role_to_mask=lambda m: m["role"] == "assistant"
    )
    ids, mask = sample.token_ids, sample.loss_mask
    assert np.count_nonzero(mask) > 0
    assert len(mask) == len(ids)


def test_build_training_sample_ensures_final_stop(model_name, tokenizer, renderer):
    """The final assistant turn ends at a trainable renderer stop token.

    Templates whose assistant close is part of the message (ChatML, Llama)
    already satisfy this, so the sample stays byte-identical; templates
    that terminate turns with the next message's role marker (GLM) get the
    canonical stop appended as the training target.
    """
    if renderer.render([{"role": "user", "content": "x"}]).sampled_mask.size == 0:
        return  # DefaultRenderer: no sampled_mask, role-only masking
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    sample = build_training_sample(renderer, msgs, ensure_final_stop=True)
    stop_ids = set(renderer.get_stop_token_ids())
    trainable_positions = np.flatnonzero(sample.loss_mask)
    last_trainable = int(trainable_positions[-1])
    assert sample.token_ids[last_trainable] in stop_ids

    baseline = build_training_sample(renderer, msgs)
    baseline_trainable_positions = np.flatnonzero(baseline.loss_mask)
    if baseline.token_ids[int(baseline_trainable_positions[-1])] in stop_ids:
        # In-message close: ensure_final_stop must not change the bytes.
        assert np.array_equal(sample.token_ids, baseline.token_ids)


def test_build_trajectory_step_reconstructs_full(model_name, tokenizer, renderer):
    """prompt_ids + completion_ids must equal the full rendered sequence."""
    prompt = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    completion = [{"role": "assistant", "content": "Hello!"}]
    step = build_trajectory_step(renderer, prompt, completion)
    full_ids = renderer.render_ids(prompt + completion)
    assert np.array_equal(
        np.concatenate((step["prompt_ids"], step["completion_ids"])), full_ids
    )


def test_build_trajectory_step_masks(model_name, tokenizer, renderer):
    """Prompt mask all False, completion mask all True."""
    prompt = [{"role": "user", "content": "Hi"}]
    completion = [{"role": "assistant", "content": "Hello!"}]
    step = build_trajectory_step(renderer, prompt, completion)
    assert not np.any(step["prompt_mask"])
    assert np.all(step["completion_mask"])
    assert len(step["completion_logprobs"]) == len(step["completion_ids"])
