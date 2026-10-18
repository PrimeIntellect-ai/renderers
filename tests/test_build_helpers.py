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


@pytest.mark.parametrize(
    "selection",
    [[0] * 6, [1] * 6, [0, 1, 0, 0, 1, 0], [False, False, True, True, False, True]],
)
@pytest.mark.parametrize("ensure_final_stop", [False, True])
@pytest.mark.parametrize("body_roles", [None, {"user", "tool"}])
@pytest.mark.parametrize("tool_turn", [False, True])
def test_build_training_sample_message_loss_mask(
    model_name, tokenizer, renderer, selection, ensure_final_stop, body_roles, tool_turn
):
    """Select arbitrary messages without changing tokens or enabling new targets."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Next question"},
        {"role": "assistant", "content": "New answer."},
        {"role": "user", "content": "One more question"},
        {"role": "assistant", "content": "Final answer."},
    ]
    if tool_turn:
        messages[1]["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"x"}'},
            }
        ]
        messages[2] = {
            "role": "tool",
            "content": "Lookup result",
            "tool_call_id": "call_1",
            "name": "lookup",
        }
    rendered = renderer.render(messages)
    kwargs = {
        "ensure_final_stop": ensure_final_stop,
        "content_sft_roles": body_roles,
    }
    if not rendered.sampled_mask:
        kwargs["role_to_mask"] = lambda m: m["role"] == "assistant"
    baseline = build_training_sample(renderer, messages, **kwargs)
    sample = build_training_sample(
        renderer, messages, message_loss_mask=selection, **kwargs
    )
    assert sample.token_ids == baseline.token_ids
    assert len(sample.loss_mask) == len(sample.token_ids)
    for k, index in enumerate(rendered.message_indices):
        if index < 0 or not selection[index]:
            assert not sample.loss_mask[k]
        else:
            assert sample.loss_mask[k] == baseline.loss_mask[k]
    # A synthesized stop is controlled by the final assistant's flag.
    assert sample.loss_mask[len(rendered.token_ids) :] == [
        keep and bool(selection[-1])
        for keep in baseline.loss_mask[len(rendered.token_ids) :]
    ]
    disabled = build_training_sample(
        renderer,
        messages,
        message_loss_mask=selection,
        role_to_mask=lambda _: False,
        ensure_final_stop=ensure_final_stop,
    )
    assert not any(disabled.loss_mask)
    trained = tokenizer.decode(
        [t for t, keep in zip(sample.token_ids[1:], sample.loss_mask[1:]) if keep]
    )
    for index, text in [
        (1, "Earlier answer."),
        (3, "New answer."),
        (5, "Final answer."),
    ]:
        assert (text in trained) == bool(selection[index])
    # Closing tokens are assistant output, including markers emitted while
    # rendering the next user/tool message. Its body keeps its own owner.
    for index, sampled in zip(rendered.message_indices, rendered.sampled_mask):
        if sampled:
            assert index >= 0 and messages[index]["role"] == "assistant"


@pytest.mark.parametrize(
    "selection",
    [
        [],
        [1],
        [1] * 5,
        [1, 0, 2, 1],
        [1, -1, 0, 1],
        [1, "0", 0, 1],
        [1, 1.0, 0, 1],
        [1, None, 0, 1],
    ],
)
def test_build_training_sample_rejects_invalid_message_loss_mask(selection):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Answer"},
        {"role": "tool", "content": "Result"},
        {"role": "assistant", "content": "Final"},
    ]
    with pytest.raises(ValueError, match="message_loss_mask"):
        build_training_sample(None, messages, message_loss_mask=selection)


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
