from __future__ import annotations

import numpy as np
import pytest

from renderers.base import (
    AttributedTextSegments,
    MultiModalData,
    PlaceholderRange,
    RenderedConversation,
    RenderedTokens,
    RenderedTrainingSample,
    build_training_sample,
    attribute_text_segments,
)
from renderers.configs import DefaultRendererConfig, Hy3RendererConfig, LagunaXS21RendererConfig
from renderers.default import DefaultRenderer
from renderers.hy3 import Hy3Renderer
from renderers.laguna_xs2 import LagunaXS21Renderer
from renderers.token_arrays import (
    FixedWidthArrayBuilder,
    FixedWidthRangeBuilder,
    MASK_DTYPE,
    LOGPROBS_DTYPE,
    RenderedTokenBuilder,
    TOKEN_IDS_DTYPE,
    TRAINING_TOKEN_IDS_DTYPE,
    TextSegmentBuilder,
    encode_token_ids,
    owned_offsets_from_array,
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


def test_offset_ownership_rejects_uint64_values_before_narrowing():
    offsets = _hostile(np.asarray([[0, np.iinfo(np.uint64).max]], dtype=np.uint64))
    with pytest.raises(ValueError, match="outside the non-negative int64 range"):
        owned_offsets_from_array("hostile", offsets, token_count=1)


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


def test_attributed_segments_require_readonly_arrays_and_seal_offsetless_masks():
    mutable_tokens = np.asarray([1], dtype=TOKEN_IDS_DTYPE)
    readonly_mask = np.asarray([False], dtype=MASK_DTYPE)
    readonly_mask.flags.writeable = False
    with pytest.raises(ValueError, match="must already be read-only"):
        AttributedTextSegments(mutable_tokens, readonly_mask, True)

    class _OffsetlessTokenizer:
        def __call__(self, text, *, add_special_tokens, return_tensors):
            return {"input_ids": _hostile(np.asarray([[2, 3]], dtype="<i8"))}

    attributed = attribute_text_segments(_OffsetlessTokenizer(), [("ab", True)])

    assert np.array_equal(attributed.token_ids, np.asarray([2, 3], dtype=TOKEN_IDS_DTYPE))
    assert np.array_equal(attributed.is_content, np.asarray([False, False], dtype=MASK_DTYPE))
    assert not attributed.token_ids.flags.writeable
    assert not attributed.is_content.flags.writeable
    assert attributed.has_content_attribution is False


def test_dynamic_text_segments_keep_data_scaled_content_flags_fixed_width():
    class _OffsetTokenizer:
        def __call__(self, text, *, add_special_tokens, return_tensors, return_offsets_mapping=False):
            token_ids = _hostile(np.ones((1, len(text)), dtype="<i8"))
            result = {"input_ids": token_ids}
            if return_offsets_mapping:
                offsets = np.empty((1, len(text), 2), dtype="<i8")
                offsets[0, :, 0] = np.arange(len(text), dtype="<i8")
                offsets[0, :, 1] = np.arange(1, len(text) + 1, dtype="<i8")
                result["offset_mapping"] = _hostile(offsets)
            return result

    segments = TextSegmentBuilder(initial_capacity=1)
    for index in range(128):
        segments.append("x", is_content=index % 2 == 0)

    attributed = attribute_text_segments(_OffsetTokenizer(), segments)
    expected_content = np.arange(128, dtype="<i8") % 2 == 0

    assert attributed.token_ids.size == 128
    assert np.array_equal(attributed.is_content, expected_content)
    assert not attributed.token_ids.flags.writeable
    assert not attributed.is_content.flags.writeable


def test_hy3_render_and_bridge_keep_hostile_arrays_typed_across_joined_segments():
    class _Tokenizer:
        unk_token_id = -1
        eos_token_id = None

        def __init__(self):
            self._specials: dict[str, int] = {}
            self.offset_probe_calls = 0

        def convert_tokens_to_ids(self, token):
            return self._specials.setdefault(token, 1000 + len(self._specials))

        def __call__(self, text, *, add_special_tokens, return_tensors, return_offsets_mapping=False):
            assert add_special_tokens is False
            assert return_tensors == "np"
            if text == "a" and return_offsets_mapping:
                self.offset_probe_calls += 1
            token_ids = _hostile(np.fromiter((ord(char) for char in text), dtype="<i8", count=len(text)).reshape(1, -1))
            result = {"input_ids": token_ids}
            if return_offsets_mapping:
                offsets = np.empty((1, len(text), 2), dtype="<i8")
                offsets[0, :, 0] = np.arange(len(text), dtype="<i8")
                offsets[0, :, 1] = np.arange(1, len(text) + 1, dtype="<i8")
                result["offset_mapping"] = _hostile(offsets)
            return result

        def encode(self, *args, **kwargs):
            raise AssertionError("legacy tokenizer encode must never be called")

        def decode(self, *args, **kwargs):
            raise AssertionError("native offsets make decode fallback unreachable")

    tokenizer = _Tokenizer()
    renderer = Hy3Renderer(tokenizer, Hy3RendererConfig())
    assert tokenizer.offset_probe_calls == 1
    rendered = renderer.render(
        [{"role": "system", "content": "policy"}, {"role": "user", "content": "question"}], add_generation_prompt=True
    )
    completion = _readonly_hostile(np.full(1, renderer._eos, dtype=TOKEN_IDS_DTYPE))
    bridged = renderer.bridge_to_next_turn(
        _readonly_hostile(rendered.token_ids), completion, [{"role": "tool", "content": "answer"}]
    )

    assert bridged is not None
    expected_prefix = np.empty(rendered.token_ids.size + completion.size, dtype=TOKEN_IDS_DTYPE)
    expected_prefix[: rendered.token_ids.size] = rendered.token_ids
    expected_prefix[rendered.token_ids.size :] = completion
    assert np.array_equal(bridged.token_ids[: expected_prefix.size], expected_prefix)

    def assert_body_run(result, text, *, message_index, sampled):
        expected = np.fromiter((ord(char) for char in text), dtype=TOKEN_IDS_DTYPE, count=len(text))
        windows = np.lib.stride_tricks.sliding_window_view(result.token_ids, expected.size)
        matches = np.flatnonzero(np.all(windows == expected, axis=1))
        assert matches.size == 1
        start = int(matches[0])
        end = start + expected.size
        assert np.all(result.message_indices[start:end] == message_index)
        assert np.all(result.sampled_mask[start:end] == sampled)
        assert np.all(result.is_content[start:end])

    assert_body_run(rendered, "policy", message_index=0, sampled=False)
    assert_body_run(rendered, "question", message_index=1, sampled=False)
    assert_body_run(bridged, "answer", message_index=0, sampled=False)
    expected_render_tail = np.fromiter(
        (renderer._assistant, renderer._think, renderer._think_end), dtype=TOKEN_IDS_DTYPE, count=3
    )
    assert rendered.token_ids[0] == renderer._bos
    assert np.array_equal(rendered.token_ids[-3:], expected_render_tail)
    assert bridged.message_indices[rendered.token_ids.size] == -1
    assert not bridged.sampled_mask[rendered.token_ids.size]
    assert not bridged.is_content[rendered.token_ids.size]
    for values in (
        rendered.token_ids,
        rendered.message_indices,
        rendered.sampled_mask,
        rendered.is_content,
        bridged.token_ids,
        bridged.message_indices,
        bridged.sampled_mask,
        bridged.is_content,
    ):
        assert isinstance(values, np.ndarray)
        assert not values.flags.writeable
    assert tokenizer.offset_probe_calls == 1


def test_laguna_offsetless_header_keeps_default_scaffold_out_of_message_zero():
    class _Tokenizer:
        unk_token_id = -1
        eos_token_id = None

        def __init__(self):
            self._specials: dict[str, int] = {}

        def convert_tokens_to_ids(self, token):
            return self._specials.setdefault(token, 1000 + len(self._specials))

        def __call__(self, text, *, add_special_tokens, return_tensors):
            return {
                "input_ids": _hostile(
                    np.fromiter((ord(char) for char in text), dtype="<i8", count=len(text)).reshape(1, -1)
                )
            }

        def decode(self, token_ids, **kwargs):
            return np.asarray(token_ids, dtype=np.uint8).tobytes().decode()

        def encode(self, *args, **kwargs):
            raise AssertionError("legacy tokenizer encode must never be called")

    def find_run(result, text):
        expected = np.fromiter((ord(char) for char in text), dtype=TOKEN_IDS_DTYPE, count=len(text))
        windows = np.lib.stride_tricks.sliding_window_view(result.token_ids, expected.size)
        matches = np.flatnonzero(np.all(windows == expected, axis=1))
        assert matches.size == 1
        start = int(matches[0])
        return slice(start, start + expected.size)

    tokenizer = _Tokenizer()
    renderer = LagunaXS21Renderer(tokenizer, LagunaXS21RendererConfig(enable_thinking=True))
    default = renderer.render([{"role": "user", "content": "question"}])
    explicit = renderer.render([{"role": "system", "content": "owned-policy"}, {"role": "user", "content": "question"}])

    default_run = find_run(default, "You are a helpful")
    explicit_run = find_run(explicit, "owned-policy")
    assert np.all(default.message_indices[default_run] == -1)
    assert np.all(explicit.message_indices[explicit_run] == 0)
    assert default.is_content.size == 0
    assert explicit.is_content.size == 0


def test_default_renderer_requests_numpy_and_builds_attribution_without_lists():
    class _Tokenizer:
        eos_token_id = None
        all_special_tokens = []

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_tensors"] == "np"
            assert kwargs["return_dict"] is False
            count = len(messages) + int(kwargs["add_generation_prompt"])
            return _hostile(np.arange(1, count + 1, dtype="<i8").reshape(1, -1))

        def decode(self, token_ids, **kwargs):
            assert isinstance(token_ids, np.ndarray)
            return "decoded"

    renderer = DefaultRenderer(_Tokenizer(), DefaultRendererConfig())
    rendered = renderer.render(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], add_generation_prompt=True
    )

    assert np.array_equal(rendered.token_ids, np.arange(1, 4, dtype=TOKEN_IDS_DTYPE))
    assert np.array_equal(rendered.message_indices, np.fromiter((0, 1, -1), dtype="<i4", count=3))
    assert not rendered.token_ids.flags.writeable
    assert not rendered.message_indices.flags.writeable


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
