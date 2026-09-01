from __future__ import annotations

import numpy as np
import pytest

from renderers.base import (
    MultiModalData,
    PlaceholderRange,
    RenderedConversation,
    RenderedTokens,
    RenderedTrainingSample,
    build_training_sample,
)
from renderers.token_arrays import (
    FixedWidthArrayBuilder,
    FixedWidthRangeBuilder,
    MASK_DTYPE,
    LOGPROBS_DTYPE,
    RenderedTokenBuilder,
    TOKEN_IDS_DTYPE,
    TRAINING_TOKEN_IDS_DTYPE,
    encode_token_ids,
    require_1d_array,
)


class _NoIterationArray(np.ndarray):
    def __iter__(self):
        raise AssertionError("numeric payload iteration is forbidden")

    def tolist(self):
        raise AssertionError("numeric payload tolist is forbidden")


def _hostile(values: np.ndarray) -> np.ndarray:
    return values.view(_NoIterationArray)


def _readonly_hostile(values: np.ndarray) -> np.ndarray:
    hostile = _hostile(values)
    hostile.flags.writeable = False
    return hostile


def test_builder_grows_and_seals_without_iterating_or_copying_at_finish():
    builder = FixedWidthArrayBuilder(TOKEN_IDS_DTYPE, initial_capacity=1)
    values = _hostile(np.asarray([11, 13, 17], dtype=TOKEN_IDS_DTYPE))

    builder.append(7)
    builder.extend(values)
    builder.extend_constant(19, 2)
    result = builder.finish()

    assert np.array_equal(result, np.asarray([7, 11, 13, 17, 19, 19], dtype=TOKEN_IDS_DTYPE))
    assert result.dtype == TOKEN_IDS_DTYPE
    assert not result.flags.writeable
    with pytest.raises(RuntimeError, match="already sealed"):
        builder.append(23)


def test_builder_and_validator_reject_legacy_lists():
    builder = FixedWidthArrayBuilder(TOKEN_IDS_DTYPE)
    with pytest.raises(TypeError, match="must be a NumPy array"):
        builder.extend([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a NumPy array"):
        require_1d_array("tokens", [1, 2, 3], dtype=TOKEN_IDS_DTYPE)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_builder_rejects_non_integer_capacity_and_count(value):
    with pytest.raises(TypeError, match="non-negative integer"):
        FixedWidthArrayBuilder(TOKEN_IDS_DTYPE, initial_capacity=value)  # type: ignore[arg-type]
    builder = FixedWidthArrayBuilder(TOKEN_IDS_DTYPE)
    with pytest.raises(TypeError, match="non-negative integer"):
        builder.extend_constant(1, value)  # type: ignore[arg-type]


def test_builder_rejects_scalar_dtype_compatibility():
    tokens = FixedWidthArrayBuilder(TOKEN_IDS_DTYPE)
    mask = FixedWidthArrayBuilder(MASK_DTYPE)
    with pytest.raises(TypeError, match="must be int"):
        tokens.append(True)
    with pytest.raises(TypeError, match="must be bool"):
        mask.append(1)


def test_range_builder_grows_without_object_rows_or_final_copy():
    builder = FixedWidthRangeBuilder(initial_capacity=1)
    builder.append(3, 5)
    values = _hostile(np.asarray([[11, 2], [17, 7]], dtype="<i8"))
    builder.extend(values)

    result = builder.finish()

    assert np.array_equal(result, np.asarray([[3, 5], [11, 2], [17, 7]], dtype="<i8"))
    assert not result.flags.writeable
    with pytest.raises(TypeError, match="must be a NumPy array"):
        FixedWidthRangeBuilder().extend([(1, 2)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-negative integer"):
        FixedWidthRangeBuilder().append(np.iinfo(np.int64).max + 1, 1)


def test_hostile_render_span_training_and_multimodal_seams_stay_vectorized():
    ranges = _readonly_hostile(np.asarray([[1, 2], [np.iinfo(np.int64).max, np.iinfo(np.int64).max]], dtype="<i8"))
    multimodal = MultiModalData(mm_placeholders={"image": ranges})
    rendered = RenderedTokens(
        token_ids=_readonly_hostile(np.asarray([11, 13, 17, 19], dtype=TOKEN_IDS_DTYPE)),
        message_indices=_readonly_hostile(np.asarray([0, 0, 1, 1], dtype="<i4")),
        sampled_mask=_readonly_hostile(np.asarray([False, True, False, True], dtype=MASK_DTYPE)),
        is_content=_readonly_hostile(np.asarray([False, True, True, False], dtype=MASK_DTYPE)),
        message_roles=["user", "assistant"],
        multi_modal_data=multimodal,
    )

    assert np.array_equal(rendered.tokens_per_message(), np.asarray([2, 2], dtype="<i8"))
    assert np.array_equal(rendered.message_token_spans(), np.asarray([[0, 2], [2, 4]], dtype="<i8"))
    assert np.array_equal(rendered.role_token_spans()["assistant"], np.asarray([[2, 4]], dtype="<i8"))
    assert np.array_equal(rendered.content_token_spans_by_role()["assistant"], np.asarray([[2, 3]], dtype="<i8"))

    class _Renderer:
        def render(self, messages, *, tools=None):
            return rendered

        def get_stop_token_ids(self):
            return [19]

    training = build_training_sample(
        _Renderer(),
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        role_to_mask=lambda message: message["role"] == "assistant",
    )
    assert np.array_equal(training.token_ids, np.asarray([11, 13, 17, 19], dtype="<i8"))
    assert np.array_equal(training.loss_mask, np.asarray([False, False, False, True]))
    assert np.array_equal(training.mm_token_type_ids, np.asarray([0, 1, 1, 0], dtype="<i8"))
    assert all(
        not values.flags.writeable for values in (training.token_ids, training.loss_mask, training.mm_token_type_ids)
    )

    with pytest.raises(TypeError, match="non-negative integer"):
        PlaceholderRange(np.iinfo(np.int64).max + 1, 1)


def test_rendered_token_builder_keeps_all_signals_aligned_and_fixed_width():
    builder = RenderedTokenBuilder(initial_capacity=1)
    tokens = _hostile(np.asarray([11, 13], dtype=TOKEN_IDS_DTYPE))
    content = _hostile(np.asarray([False, True], dtype=MASK_DTYPE))

    builder.emit_special(7, -1, is_sampled=False, is_content=False)
    builder.emit_tokens(tokens, 0, is_sampled=True, is_content=content)
    rendered = builder.finish(message_roles=["assistant"])

    assert np.array_equal(rendered.token_ids, np.asarray([7, 11, 13], dtype="<i4"))
    assert np.array_equal(rendered.message_indices, np.asarray([-1, 0, 0], dtype="<i4"))
    assert np.array_equal(rendered.sampled_mask, np.asarray([False, True, True], dtype=np.bool_))
    assert np.array_equal(rendered.is_content, np.asarray([False, False, True], dtype=np.bool_))
    assert all(
        not values.flags.writeable
        for values in (rendered.token_ids, rendered.message_indices, rendered.sampled_mask, rendered.is_content)
    )


def test_rendered_token_builder_rejects_list_and_misaligned_mask_custody():
    builder = RenderedTokenBuilder()
    with pytest.raises(TypeError, match="must be a NumPy array"):
        builder.emit_tokens([1, 2], 0, is_sampled=False, is_content=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        builder.emit_tokens(
            np.asarray([1, 2], dtype=TOKEN_IDS_DTYPE),
            0,
            is_sampled=False,
            is_content=np.asarray([True], dtype=MASK_DTYPE),
        )
    assert len(builder) == 0
    with pytest.raises(TypeError, match="is_content must be bool"):
        builder.emit_tokens(
            np.asarray([1, 2], dtype=TOKEN_IDS_DTYPE),
            0,
            is_sampled=False,
            is_content=[True, False],  # type: ignore[arg-type]
        )
    assert len(builder) == 0


def test_training_sample_rejects_mutable_aliases_without_mutating_caller():
    token_ids = np.asarray([2, 3], dtype=TRAINING_TOKEN_IDS_DTYPE)
    loss_mask = np.asarray([False, True], dtype=MASK_DTYPE)

    with pytest.raises(ValueError, match="must already be read-only"):
        RenderedTrainingSample(token_ids=token_ids, loss_mask=loss_mask)

    assert token_ids.flags.writeable
    assert loss_mask.flags.writeable


def test_encode_token_ids_uses_numpy_tokenizer_contract_without_iteration():
    expected = _hostile(np.asarray([[2, 3, 5]], dtype="<i8"))

    class _Tokenizer:
        def __call__(self, text, *, add_special_tokens, return_tensors):
            assert text == "payload"
            assert add_special_tokens is False
            assert return_tensors == "np"
            return {"input_ids": expected}

        def encode(self, *args, **kwargs):
            raise AssertionError("NumPy tokenizer path must bypass list-returning encode")

    actual = encode_token_ids(_Tokenizer(), "payload")

    assert np.array_equal(actual, np.asarray([2, 3, 5], dtype=TOKEN_IDS_DTYPE))
    assert actual.dtype == TOKEN_IDS_DTYPE
    assert not actual.flags.writeable
    expected[0, 0] = 101
    assert actual[0] == 2


def test_encode_token_ids_rejects_legacy_encode_fallback():
    class _Tokenizer:
        def encode(self, text, *, add_special_tokens):
            raise AssertionError("legacy list-returning encode must never be invoked")

    with pytest.raises(TypeError, match="callable NumPy tokenization"):
        encode_token_ids(_Tokenizer(), "payload")


def test_rendered_conversation_validates_and_takes_readonly_completion_ownership():
    prompt = _readonly_hostile(np.asarray([2, 3], dtype=TOKEN_IDS_DTYPE))
    conversation = RenderedConversation(prompt_ids=prompt)
    completion = _hostile(np.asarray([5, 7], dtype=TOKEN_IDS_DTYPE))
    logprobs = _hostile(np.asarray([-0.5, -0.25], dtype=LOGPROBS_DTYPE))

    completed = conversation.with_completion(completion, completion_logprobs=logprobs)

    assert np.array_equal(completed.prompt_ids, prompt)
    assert np.array_equal(completed.completion_ids, completion)
    assert np.array_equal(completed.completion_logprobs, logprobs)
    assert all(
        not values.flags.writeable
        for values in (completed.prompt_ids, completed.completion_ids, completed.completion_logprobs)
    )
    completion[0] = 101
    assert completed.completion_ids[0] == 5

    with pytest.raises(TypeError, match="must be a NumPy array"):
        RenderedConversation(prompt_ids=[2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must already be read-only"):
        RenderedConversation(prompt_ids=np.asarray([2, 3], dtype=TOKEN_IDS_DTYPE))
    with pytest.raises(ValueError, match="zero or match"):
        RenderedConversation(
            prompt_ids=prompt,
            completion_ids=_readonly_hostile(np.asarray([5, 7], dtype=TOKEN_IDS_DTYPE)),
            completion_logprobs=_readonly_hostile(np.asarray([-0.5], dtype=LOGPROBS_DTYPE)),
        )
