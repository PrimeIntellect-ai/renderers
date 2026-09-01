from __future__ import annotations

import numpy as np
import pytest

from renderers.token_arrays import (
    FixedWidthArrayBuilder,
    MASK_DTYPE,
    RenderedTokenBuilder,
    TOKEN_IDS_DTYPE,
    encode_token_ids,
    require_1d_array,
)


class _NoIterationArray(np.ndarray):
    def __iter__(self):
        raise AssertionError("numeric payload iteration is forbidden")


def _hostile(values: np.ndarray) -> np.ndarray:
    return values.view(_NoIterationArray)


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


def test_rendered_token_builder_keeps_all_signals_aligned_and_fixed_width():
    builder = RenderedTokenBuilder(initial_capacity=1)
    tokens = _hostile(np.asarray([11, 13], dtype=TOKEN_IDS_DTYPE))
    content = _hostile(np.asarray([False, True], dtype=MASK_DTYPE))

    builder.emit_special(7, -1, sampled=False, content=False)
    builder.emit_tokens(tokens, 0, sampled=True, content=content)
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
        builder.emit_tokens([1, 2], 0, sampled=False, content=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        builder.emit_tokens(
            np.asarray([1, 2], dtype=TOKEN_IDS_DTYPE), 0, sampled=False, content=np.asarray([True], dtype=MASK_DTYPE)
        )
    assert len(builder) == 0
    with pytest.raises(TypeError, match="content must be bool"):
        builder.emit_tokens(
            np.asarray([1, 2], dtype=TOKEN_IDS_DTYPE),
            0,
            sampled=False,
            content=[True, False],  # type: ignore[arg-type]
        )
    assert len(builder) == 0


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
            return [2, 3, 5]

    with pytest.raises(TypeError, match="legacy list token custody is unsupported"):
        encode_token_ids(_Tokenizer(), "payload")
