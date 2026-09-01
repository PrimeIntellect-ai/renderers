"""Strict fixed-width arrays and grow-as-you-go renderer storage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

TOKEN_IDS_DTYPE = np.dtype("<i4")
MESSAGE_INDICES_DTYPE = np.dtype("<i4")
MASK_DTYPE = np.dtype(np.bool_)
LOGPROBS_DTYPE = np.dtype("<f8")
OFFSETS_DTYPE = np.dtype("<i8")
COUNTS_DTYPE = np.dtype("<i8")
TRAINING_TOKEN_IDS_DTYPE = np.dtype("<i8")
MM_TOKEN_TYPE_IDS_DTYPE = np.dtype("<i8")


def require_1d_array(name: str, value: object, *, dtype: np.dtype, minimum: int | None = None) -> np.ndarray:
    """Validate an ndarray without accepting or materializing list payloads."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(value).__name__}")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got shape {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.str}, got {value.dtype.str}")
    if minimum is not None and value.size and np.any(value < minimum):
        raise ValueError(f"{name} values must be >= {minimum}")
    return value


def readonly_view(value: np.ndarray) -> np.ndarray:
    """Return a read-only view; callers must already own the source buffer."""
    view = value.view()
    view.flags.writeable = False
    return view


def require_readonly(name: str, value: np.ndarray) -> np.ndarray:
    """Reject externally mutable custody rather than mutating caller ownership."""
    if value.flags.writeable:
        raise ValueError(f"{name} must already be read-only")
    return value


def require_range_array(name: str, value: object) -> np.ndarray:
    """Validate fixed-width ``[offset, length]`` rows without object custody."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(value).__name__}")
    if value.ndim != 2 or value.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape [items, 2], got {value.shape}")
    if value.dtype != OFFSETS_DTYPE:
        raise TypeError(f"{name} must have dtype {OFFSETS_DTYPE.str}, got {value.dtype.str}")
    if value.size and np.any(value < 0):
        raise ValueError(f"{name} offsets and lengths must be non-negative")
    return value


def empty_array(dtype: np.dtype) -> np.ndarray:
    value = np.empty(0, dtype=dtype)
    value.flags.writeable = False
    return value


def owned_readonly_copy(name: str, value: object, *, dtype: np.dtype, minimum: int | None = None) -> np.ndarray:
    """Validate then take explicit immutable ownership at a public boundary."""
    source = require_1d_array(name, value, dtype=dtype, minimum=minimum)
    owned = np.array(source, dtype=dtype, copy=True, order="C")
    owned.flags.writeable = False
    return owned


class FixedWidthArrayBuilder:
    """A geometrically growing NumPy buffer with no list-backed phase."""

    __slots__ = ("_buffer", "_dtype", "_sealed", "_size")

    def __init__(self, dtype: np.dtype, *, initial_capacity: int = 64) -> None:
        if type(initial_capacity) is not int or initial_capacity < 0:
            raise TypeError("initial_capacity must be a non-negative integer")
        self._dtype = np.dtype(dtype)
        self._buffer = np.empty(initial_capacity, dtype=self._dtype)
        self._size = 0
        self._sealed = False

    def __len__(self) -> int:
        return self._size

    def _reserve(self, additional: int) -> None:
        if type(additional) is not int or additional < 0:
            raise TypeError("additional capacity must be a non-negative integer")
        if self._sealed:
            raise RuntimeError("fixed-width array builder is already sealed")
        required = self._size + additional
        if required <= self._buffer.size:
            return
        capacity = max(required, 1, self._buffer.size * 2)
        grown = np.empty(capacity, dtype=self._dtype)
        grown[: self._size] = self._buffer[: self._size]
        self._buffer = grown

    def append(self, value: int | bool) -> None:
        self._validate_scalar(value)
        self._reserve(1)
        self._buffer[self._size] = value
        self._size += 1

    def extend(self, values: np.ndarray) -> None:
        require_1d_array("builder values", values, dtype=self._dtype)
        count = values.size
        self._reserve(count)
        self._buffer[self._size : self._size + count] = values
        self._size += count

    def extend_constant(self, value: int | bool, count: int) -> None:
        self._validate_scalar(value)
        if type(count) is not int or count < 0:
            raise TypeError("count must be a non-negative integer")
        self._reserve(count)
        self._buffer[self._size : self._size + count] = value
        self._size += count

    def _validate_scalar(self, value: int | bool) -> None:
        if self._dtype == MASK_DTYPE:
            if type(value) is not bool:
                raise TypeError(f"builder value must be bool, got {type(value).__name__}")
            return
        if type(value) is not int:
            raise TypeError(f"builder value must be int, got {type(value).__name__}")
        bounds = np.iinfo(self._dtype)
        if value < bounds.min or value > bounds.max:
            raise ValueError(f"builder value {value} is outside {self._dtype.str}")

    def finish(self) -> np.ndarray:
        """Seal and return the populated prefix without an O(n) final copy."""
        self._sealed = True
        return readonly_view(self._buffer[: self._size])


class TextSegments:
    """Structural text chunks aligned with one fixed-width content mask."""

    __slots__ = ("is_content", "texts")

    def __init__(self, texts: tuple[str, ...], is_content: np.ndarray) -> None:
        require_1d_array("segment is_content", is_content, dtype=MASK_DTYPE)
        require_readonly("segment is_content", is_content)
        if len(texts) != is_content.size:
            raise ValueError("segment texts and content mask must have equal lengths")
        self.texts = texts
        self.is_content = is_content


class TextSegmentBuilder:
    """Accumulate structural text while numeric attribution stays fixed-width."""

    __slots__ = ("_content", "_sealed", "_texts")

    def __init__(self, *, initial_capacity: int = 4) -> None:
        self._texts: list[str] = []
        self._content = FixedWidthArrayBuilder(MASK_DTYPE, initial_capacity=initial_capacity)
        self._sealed = False

    def __len__(self) -> int:
        return len(self._texts)

    def append(self, text: str, *, is_content: bool) -> None:
        if self._sealed:
            raise RuntimeError("text segment builder is already sealed")
        if type(text) is not str:
            raise TypeError(f"segment text must be str, got {type(text).__name__}")
        if type(is_content) is not bool:
            raise TypeError(f"segment is_content must be bool, got {type(is_content).__name__}")
        self._texts.append(text)
        self._content.append(is_content)

    def finish(self) -> TextSegments:
        self._sealed = True
        return TextSegments(tuple(self._texts), self._content.finish())


class FixedWidthRangeBuilder:
    """Grow-as-you-go fixed-width storage for offset/length rows."""

    __slots__ = ("_buffer", "_sealed", "_size")

    def __init__(self, *, initial_capacity: int = 4) -> None:
        if type(initial_capacity) is not int or initial_capacity < 0:
            raise TypeError("initial_capacity must be a non-negative integer")
        self._buffer = np.empty((initial_capacity, 2), dtype=OFFSETS_DTYPE)
        self._size = 0
        self._sealed = False

    def __len__(self) -> int:
        return self._size

    def _reserve(self, additional: int) -> None:
        if type(additional) is not int or additional < 0:
            raise TypeError("additional capacity must be a non-negative integer")
        if self._sealed:
            raise RuntimeError("fixed-width range builder is already sealed")
        required = self._size + additional
        if required <= self._buffer.shape[0]:
            return
        capacity = max(required, 1, self._buffer.shape[0] * 2)
        grown = np.empty((capacity, 2), dtype=OFFSETS_DTYPE)
        grown[: self._size] = self._buffer[: self._size]
        self._buffer = grown

    def append(self, offset: int, length: int) -> None:
        upper_bound = np.iinfo(OFFSETS_DTYPE).max
        for name, value in (("offset", offset), ("length", length)):
            if type(value) is not int or value < 0 or value > upper_bound:
                raise TypeError(f"{name} must be a non-negative integer")
        self._reserve(1)
        self._buffer[self._size, 0] = offset
        self._buffer[self._size, 1] = length
        self._size += 1

    def extend(self, values: np.ndarray) -> None:
        require_range_array("range builder values", values)
        count = values.shape[0]
        self._reserve(count)
        self._buffer[self._size : self._size + count] = values
        self._size += count

    def finish(self) -> np.ndarray:
        self._sealed = True
        return readonly_view(self._buffer[: self._size])


def finish_range_builders(builders: Mapping[str, FixedWidthRangeBuilder]) -> dict[str, np.ndarray]:
    """Seal a structural modality map without numeric list intermediates."""
    return {modality: builder.finish() for modality, builder in builders.items()}


def merge_range_maps(*maps: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Merge bridge-owned range maps into new fixed-width custody."""
    builders: dict[str, FixedWidthRangeBuilder] = {}
    for values_by_modality in maps:
        for modality, values in values_by_modality.items():
            require_range_array(f"mm_placeholders[{modality!r}]", values)
            builders.setdefault(modality, FixedWidthRangeBuilder()).extend(values)
    return finish_range_builders(builders)


_OFFSET_CAPABILITY_UNRESOLVED = object()


class RenderedTokenBuilder:
    """Aligned grow-as-you-go storage for every per-token renderer signal."""

    __slots__ = ("_is_content", "_message_indices", "_offset_tokenizer", "_sampled_mask", "_token_ids", "_tokenizer")

    def __init__(
        self,
        tokenizer: Any = None,
        *,
        offset_tokenizer: Any = _OFFSET_CAPABILITY_UNRESOLVED,
        initial_capacity: int = 64,
    ) -> None:
        self._tokenizer = tokenizer
        self._offset_tokenizer = offset_tokenizer
        self._token_ids = FixedWidthArrayBuilder(TOKEN_IDS_DTYPE, initial_capacity=initial_capacity)
        self._message_indices = FixedWidthArrayBuilder(MESSAGE_INDICES_DTYPE, initial_capacity=initial_capacity)
        self._sampled_mask = FixedWidthArrayBuilder(MASK_DTYPE, initial_capacity=initial_capacity)
        self._is_content = FixedWidthArrayBuilder(MASK_DTYPE, initial_capacity=initial_capacity)

    def __len__(self) -> int:
        return len(self._token_ids)

    def emit_special(
        self, token_id: int, message_index: int = -1, *, is_sampled: bool = False, is_content: bool = False
    ) -> None:
        if type(token_id) is not int:
            raise TypeError(f"token_id must be int, got {type(token_id).__name__}")
        if type(message_index) is not int:
            raise TypeError(f"message_index must be int, got {type(message_index).__name__}")
        if type(is_sampled) is not bool:
            raise TypeError(f"is_sampled must be bool, got {type(is_sampled).__name__}")
        if type(is_content) is not bool:
            raise TypeError(f"is_content must be bool, got {type(is_content).__name__}")
        if token_id < 0 or token_id > np.iinfo(TOKEN_IDS_DTYPE).max:
            raise ValueError(f"token_id is outside the int32 token range: {token_id}")
        if message_index < -1 or message_index > np.iinfo(MESSAGE_INDICES_DTYPE).max:
            raise ValueError(f"message_index is outside the int32 attribution range: {message_index}")
        self._token_ids.append(token_id)
        self._message_indices.append(message_index)
        self._sampled_mask.append(is_sampled)
        self._is_content.append(is_content)

    def emit_tokens(
        self, token_ids: np.ndarray, message_index: int, *, is_sampled: bool | np.ndarray, is_content: bool | np.ndarray
    ) -> None:
        require_1d_array("token_ids", token_ids, dtype=TOKEN_IDS_DTYPE, minimum=0)
        if type(message_index) is not int:
            raise TypeError(f"message_index must be int, got {type(message_index).__name__}")
        if message_index < -1 or message_index > np.iinfo(MESSAGE_INDICES_DTYPE).max:
            raise ValueError(f"message_index is outside the int32 attribution range: {message_index}")
        if isinstance(is_sampled, np.ndarray):
            require_1d_array("is_sampled", is_sampled, dtype=MASK_DTYPE)
            if is_sampled.size != token_ids.size:
                raise ValueError(
                    f"is_sampled length {is_sampled.size} does not match token_ids length {token_ids.size}"
                )
        elif type(is_sampled) is not bool:
            raise TypeError(f"is_sampled must be bool or a NumPy bool array, got {type(is_sampled).__name__}")
        if isinstance(is_content, np.ndarray):
            require_1d_array("is_content", is_content, dtype=MASK_DTYPE)
            if is_content.size != token_ids.size:
                raise ValueError(
                    f"is_content length {is_content.size} does not match token_ids length {token_ids.size}"
                )
        elif type(is_content) is not bool:
            raise TypeError(f"is_content must be bool or a NumPy bool array, got {type(is_content).__name__}")

        self._token_ids.extend(token_ids)
        self._message_indices.extend_constant(message_index, token_ids.size)
        if isinstance(is_sampled, np.ndarray):
            self._sampled_mask.extend(is_sampled)
        else:
            self._sampled_mask.extend_constant(is_sampled, token_ids.size)
        if isinstance(is_content, np.ndarray):
            self._is_content.extend(is_content)
        else:
            self._is_content.extend_constant(is_content, token_ids.size)

    def emit_aligned(
        self, token_ids: np.ndarray, message_indices: np.ndarray, sampled_mask: np.ndarray, is_content: np.ndarray
    ) -> None:
        """Append already-aligned fixed-width signals without scalar iteration."""
        require_1d_array("token_ids", token_ids, dtype=TOKEN_IDS_DTYPE, minimum=0)
        require_1d_array("message_indices", message_indices, dtype=MESSAGE_INDICES_DTYPE, minimum=-1)
        require_1d_array("sampled_mask", sampled_mask, dtype=MASK_DTYPE)
        require_1d_array("is_content", is_content, dtype=MASK_DTYPE)
        sizes = {token_ids.size, message_indices.size, sampled_mask.size, is_content.size}
        if len(sizes) != 1:
            raise ValueError("aligned token signals must have equal lengths")
        self._token_ids.extend(token_ids)
        self._message_indices.extend(message_indices)
        self._sampled_mask.extend(sampled_mask)
        self._is_content.extend(is_content)

    def prepend_prior(self, token_ids: np.ndarray) -> None:
        self.emit_tokens(token_ids, -1, is_sampled=False, is_content=False)

    def emit_text(
        self, text: str, message_index: int = -1, *, is_sampled: bool = False, is_content: bool = False
    ) -> None:
        if text:
            if self._tokenizer is None:
                raise RuntimeError("emit_text requires a tokenizer-bound RenderedTokenBuilder")
            self.emit_tokens(
                encode_token_ids(self._tokenizer, text), message_index, is_sampled=is_sampled, is_content=is_content
            )

    def emit_text_segments(
        self,
        segments: TextSegmentBuilder | list[tuple[str, bool]],
        message_index: int = -1,
        *,
        is_sampled: bool = False,
        overlap_is_content: bool = False,
    ) -> bool:
        from renderers.base import attribute_text_segments

        if self._tokenizer is None:
            raise RuntimeError("emit_text_segments requires a tokenizer-bound RenderedTokenBuilder")
        attributed = attribute_text_segments(
            self._tokenizer,
            segments,
            overlap_is_content=overlap_is_content,
            _offset_tokenizer=self._resolved_offset_tokenizer(),
        )
        self.emit_tokens(attributed.token_ids, message_index, is_sampled=is_sampled, is_content=attributed.is_content)
        return attributed.has_content_attribution

    def emit_assistant_segments(
        self,
        segments: TextSegmentBuilder | list[tuple[str, bool]],
        message_index: int = -1,
        *,
        overlap_is_content: bool = False,
    ) -> bool:
        """Emit assistant text, sampling exactly the attributable content tokens."""
        from renderers.base import attribute_text_segments

        if self._tokenizer is None:
            raise RuntimeError("emit_assistant_segments requires a tokenizer-bound builder")
        attributed = attribute_text_segments(
            self._tokenizer,
            segments,
            overlap_is_content=overlap_is_content,
            _offset_tokenizer=self._resolved_offset_tokenizer(),
        )
        self.emit_tokens(
            attributed.token_ids, message_index, is_sampled=attributed.is_content, is_content=attributed.is_content
        )
        return attributed.has_content_attribution

    def _resolved_offset_tokenizer(self) -> Any:
        if self._offset_tokenizer is _OFFSET_CAPABILITY_UNRESOLVED:
            from renderers.base import _get_offset_tokenizer

            self._offset_tokenizer = _get_offset_tokenizer(self._tokenizer)
        return self._offset_tokenizer

    def finish(
        self,
        *,
        message_roles: list[str] | None = None,
        message_tool_names: list[str | None] | None = None,
        multi_modal_data: Any = None,
        sampled_available: bool = True,
        content_available: bool = True,
    ) -> Any:
        """Seal aligned arrays and construct the public RenderedTokens value."""
        from renderers.base import RenderedTokens

        sizes = {len(self._token_ids), len(self._message_indices), len(self._sampled_mask), len(self._is_content)}
        if len(sizes) != 1:
            raise RuntimeError("rendered token builder signals are misaligned")
        token_ids = self._token_ids.finish()
        message_indices = self._message_indices.finish()
        sampled_mask = self._sampled_mask.finish() if sampled_available else empty_array(MASK_DTYPE)
        is_content = self._is_content.finish() if content_available else empty_array(MASK_DTYPE)
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            sampled_mask=sampled_mask,
            is_content=is_content,
            message_roles=message_roles or [],
            message_tool_names=message_tool_names or [],
            multi_modal_data=multi_modal_data,
        )


def owned_token_ids_from_array(name: str, value: object) -> np.ndarray:
    """Validate fixed-width tokenizer output and take readonly int32 ownership."""
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"{name} must return NumPy input_ids; legacy {type(value).__name__} token custody is unsupported"
        )
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError(f"{name} input_ids must have shape [tokens] or [1, tokens], got {value.shape}")
    if value.dtype.kind not in "iu" or value.dtype.itemsize > 8:
        raise TypeError(f"{name} input_ids must use a fixed-width integer dtype, got {value.dtype}")
    if value.size and (np.any(value < 0) or np.any(value > np.iinfo(TOKEN_IDS_DTYPE).max)):
        raise ValueError(f"{name} input_ids are outside the int32 token range")
    owned = np.array(value, dtype=TOKEN_IDS_DTYPE, copy=True, order="C")
    owned.flags.writeable = False
    return owned


def owned_offsets_from_array(name: str, value: object, *, token_count: int) -> np.ndarray:
    """Validate tokenizer offsets and take readonly little-endian int64 ownership."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must return NumPy offsets, got {type(value).__name__}")
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.shape != (token_count, 2):
        raise ValueError(f"{name} offsets must have shape [{token_count}, 2], got {value.shape}")
    if value.dtype.kind not in "iu" or value.dtype.itemsize > 8:
        raise TypeError(f"{name} offsets must use a fixed-width integer dtype, got {value.dtype}")
    if value.size and (np.any(value < 0) or np.any(value > np.iinfo(OFFSETS_DTYPE).max)):
        raise ValueError(f"{name} offsets are outside the non-negative int64 range")
    owned = np.array(value, dtype=OFFSETS_DTYPE, copy=True, order="C")
    owned.flags.writeable = False
    return owned


def encode_token_ids(tokenizer: Any, text: str) -> np.ndarray:
    """Encode only through the NumPy tokenizer ABI; never invoke list APIs."""
    if not callable(tokenizer):
        raise TypeError(f"{type(tokenizer).__name__} must support callable NumPy tokenization")
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="np")
    except TypeError as exc:
        raise TypeError(f"{type(tokenizer).__name__} must support callable NumPy tokenization") from exc
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError(f"{type(tokenizer).__name__} must return a mapping with NumPy input_ids")
    return owned_token_ids_from_array(type(tokenizer).__name__, encoded["input_ids"])
