"""Barrage test: build_training_sample and build_trajectory_step.

Runs against every (model, renderer) pair.
"""

from renderers import build_training_sample, build_trajectory_step
from renderers.base import PlaceholderRange, _build_mm_token_type_ids


def test_build_mm_token_type_ids_marks_ranges():
    """Image runs → 1, video runs → 2, everything else → 0; clips at length."""
    placeholders = {
        "image": [PlaceholderRange(offset=2, length=3)],  # tokens 2,3,4
        "video": [PlaceholderRange(offset=7, length=2)],  # tokens 7,8
    }
    ids = _build_mm_token_type_ids(placeholders, length=10)
    assert ids == [0, 0, 1, 1, 1, 0, 0, 2, 2, 0]


def _expected(tokenizer, messages, **kwargs):
    # Match the Renderer Protocol's default for add_generation_prompt
    # (False); some tokenizers default it to True in their config
    # (e.g. Kimi) which would otherwise flip the parity check on the flag
    # alone. Callers wanting the gen prompt still pass it through.
    kwargs.setdefault("add_generation_prompt", False)
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, **kwargs
    )
    if isinstance(result, dict):
        return list(result["input_ids"])
    if isinstance(result, str):
        return list(tokenizer.encode(result, add_special_tokens=False))
    return list(result)


def test_build_training_sample_ids_match(model_name, tokenizer, renderer):
    """Token IDs must match apply_chat_template."""
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
