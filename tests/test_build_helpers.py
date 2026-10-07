"""Barrage test: build_training_sample and build_trajectory_step.

Runs against every (model, renderer) pair.
"""

import pytest

from renderers import build_training_sample, build_trajectory_step
from renderers.base import PlaceholderRange, _build_mm_token_type_ids
from tests.reference_rendering import render_reference


def test_build_mm_token_type_ids_marks_ranges():
    """Image runs → 1, video runs → 2, everything else → 0; clips at length."""
    placeholders = {
        "image": [PlaceholderRange(offset=2, length=3)],  # tokens 2,3,4
        "video": [PlaceholderRange(offset=7, length=2)],  # tokens 7,8
    }
    ids = _build_mm_token_type_ids(placeholders, length=10)
    assert ids == [0, 0, 1, 1, 1, 0, 0, 2, 2, 0]


def _expected(tokenizer, messages, **kwargs):
    return render_reference(tokenizer, messages, **kwargs)


def test_build_training_sample_ids_match(model_name, tokenizer, renderer):
    """Token IDs must match the model-aware reference renderer."""
    if (
        model_name
        in {
            "google/gemma-4-26B-A4B-it",
            "google/gemma-4-31B-it",
        }
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
    assert ids == _expected(tokenizer, msgs)
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
    assert sum(mask) > 0
    assert len(mask) == len(ids)


def test_build_training_sample_ensures_final_stop(model_name, tokenizer, renderer):
    """The final assistant turn ends at a trainable renderer stop token.

    Templates whose assistant close is part of the message (ChatML, Llama)
    already satisfy this, so the sample stays byte-identical; templates
    that terminate turns with the next message's role marker (GLM) get the
    canonical stop appended as the training target.
    """
    if not renderer.render([{"role": "user", "content": "x"}]).sampled_mask:
        return  # DefaultRenderer: no sampled_mask, role-only masking
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    sample = build_training_sample(renderer, msgs, ensure_final_stop=True)
    stop_ids = set(renderer.get_stop_token_ids())
    last_trainable = max(k for k, m in enumerate(sample.loss_mask) if m)
    assert sample.token_ids[last_trainable] in stop_ids

    baseline = build_training_sample(renderer, msgs)
    if (
        baseline.token_ids[max(k for k, m in enumerate(baseline.loss_mask) if m)]
        in stop_ids
    ):
        # In-message close: ensure_final_stop must not change the bytes.
        assert sample.token_ids == baseline.token_ids


def test_build_trajectory_step_reconstructs_full(model_name, tokenizer, renderer):
    """prompt_ids + completion_ids must equal the full rendered sequence."""
    prompt = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    completion = [{"role": "assistant", "content": "Hello!"}]
    step = build_trajectory_step(renderer, prompt, completion)
    full_ids = renderer.render_ids(prompt + completion)
    assert step["prompt_ids"] + step["completion_ids"] == full_ids


def test_build_trajectory_step_masks(model_name, tokenizer, renderer):
    """Prompt mask all False, completion mask all True."""
    prompt = [{"role": "user", "content": "Hi"}]
    completion = [{"role": "assistant", "content": "Hello!"}]
    step = build_trajectory_step(renderer, prompt, completion)
    assert all(m is False for m in step["prompt_mask"])
    assert all(m is True for m in step["completion_mask"])
    assert len(step["completion_logprobs"]) == len(step["completion_ids"])


def _trainable_mask_messages(flags):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Alpha answer."},
        {"role": "user", "content": "Next question"},
        {"role": "assistant", "content": "Bravo answer."},
    ]
    for message, flag in zip(msgs, flags):
        if flag is not None:
            message["trainable_mask"] = flag
    return msgs


def _default_kwargs(renderer, msgs):
    """Default build_training_sample kwargs; renderers without sampled_mask need a role filter."""
    rendered = renderer.render(msgs)
    if len(rendered.sampled_mask) == len(rendered.token_ids):
        return {}
    return {"role_to_mask": lambda m: m["role"] == "assistant"}


@pytest.mark.parametrize("masked_turn", [1, 3])
def test_build_training_sample_trainable_mask_masks_one_assistant_turn(
    model_name, tokenizer, renderer, masked_turn
):
    """0 masks every token attributed to the turn; the other turn trains as by default; ids are unchanged."""
    plain = _trainable_mask_messages([None] * 4)
    flags = [None, 1, None, 1]
    flags[masked_turn] = 0
    msgs = _trainable_mask_messages(flags)
    kwargs = _default_kwargs(renderer, plain)
    baseline = build_training_sample(renderer, plain, **kwargs)
    sample = build_training_sample(renderer, msgs, **kwargs)

    assert sample.token_ids == baseline.token_ids
    attributed = renderer.render(plain).message_indices
    expected = [
        trainable and index != masked_turn
        for trainable, index in zip(baseline.loss_mask, attributed)
    ]
    assert sample.loss_mask == expected
    assert 0 < sum(sample.loss_mask) < sum(baseline.loss_mask)


def test_build_training_sample_trainable_mask_unset_is_default(
    model_name, tokenizer, renderer
):
    """Absent, None and empty values leave the sample byte-identical."""
    plain = _trainable_mask_messages([None] * 4)
    msgs = _trainable_mask_messages([None] * 4)
    msgs[0]["trainable_mask"] = None
    msgs[2]["trainable_mask"] = ""
    kwargs = _default_kwargs(renderer, plain)
    baseline = build_training_sample(renderer, plain, ensure_final_stop=True, **kwargs)
    sample = build_training_sample(renderer, msgs, ensure_final_stop=True, **kwargs)
    assert sample.token_ids == baseline.token_ids
    assert sample.loss_mask == baseline.loss_mask


def test_build_training_sample_trainable_mask_trains_user_body(
    model_name, tokenizer, renderer
):
    """1 on a user message trains its body tokens, never its scaffolding."""
    plain = _trainable_mask_messages([None] * 4)
    msgs = _trainable_mask_messages([None, None, 1, None])
    kwargs = _default_kwargs(renderer, plain)
    baseline = build_training_sample(renderer, plain, **kwargs)
    sample = build_training_sample(renderer, msgs, **kwargs)
    rendered = renderer.render(plain)
    has_sampled = len(rendered.sampled_mask) == len(rendered.token_ids)
    has_content = len(rendered.is_content) == len(rendered.token_ids)
    if has_sampled and not has_content:
        pytest.skip("renderer does not mark message bodies (is_content)")
    expected = []
    for k, index in enumerate(rendered.message_indices):
        if index != 2:
            expected.append(baseline.loss_mask[k])
        elif not has_sampled or rendered.sampled_mask[k]:
            expected.append(True)
        else:
            expected.append(rendered.is_content[k])
    assert sample.loss_mask == expected
    assert sum(sample.loss_mask) > sum(baseline.loss_mask)


def test_build_training_sample_trainable_mask_final_turn_skips_final_stop(
    model_name, tokenizer, renderer
):
    """ensure_final_stop does not append a stop for a final assistant turn marked 0."""
    msgs = _trainable_mask_messages([None, None, None, 0])
    kwargs = _default_kwargs(renderer, msgs)
    sample = build_training_sample(renderer, msgs, ensure_final_stop=True, **kwargs)
    plain = _trainable_mask_messages([None] * 4)
    assert sample.token_ids == renderer.render_ids(plain)
    attributed = renderer.render(plain).message_indices
    assert not any(
        trainable
        for trainable, index in zip(sample.loss_mask, attributed)
        if index == 3
    )


@pytest.mark.parametrize("value", [2, -1, 0.5, "yes", [1]])
def test_build_training_sample_trainable_mask_rejects_invalid_values(
    model_name, tokenizer, renderer, value
):
    msgs = _trainable_mask_messages([None, value, None, None])
    with pytest.raises(ValueError, match="trainable_mask"):
        build_training_sample(
            renderer,
            msgs,
            **_default_kwargs(renderer, _trainable_mask_messages([None] * 4)),
        )
