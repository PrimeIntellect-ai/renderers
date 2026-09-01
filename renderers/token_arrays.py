"""Strict fixed-width arrays and grow-as-you-go renderer storage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

TOKEN_IDS_DTYPE = np.dtype("<i4")
MESSAGE_INDICES_DTYPE = np.dtype("<i4")
MASK_DTYPE = np.dtype(np.bool_)
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
    """Return a read-only zero-copy view of a caller-owned buffer."""
    view = value.view()
    view.flags.writeable = False
    return view


def empty_array(dtype: np.dtype) -> np.ndarray:
    value = np.empty(0, dtype=dtype)
    value.flags.writeable = False
    return value


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


class RenderedTokenBuilder:
    """Aligned grow-as-you-go storage for every per-token renderer signal."""

    __slots__ = ("_is_content", "_message_indices", "_sampled_mask", "_token_ids")

    def __init__(self, *, initial_capacity: int = 64) -> None:
        self._token_ids = FixedWidthArrayBuilder(TOKEN_IDS_DTYPE, initial_capacity=initial_capacity)
        self._message_indices = FixedWidthArrayBuilder(MESSAGE_INDICES_DTYPE, initial_capacity=initial_capacity)
        self._sampled_mask = FixedWidthArrayBuilder(MASK_DTYPE, initial_capacity=initial_capacity)
        self._is_content = FixedWidthArrayBuilder(MASK_DTYPE, initial_capacity=initial_capacity)

    def __len__(self) -> int:
        return len(self._token_ids)

    def emit_special(self, token_id: int, message_index: int, *, sampled: bool, content: bool) -> None:
        if type(token_id) is not int:
            raise TypeError(f"token_id must be int, got {type(token_id).__name__}")
        if type(message_index) is not int:
            raise TypeError(f"message_index must be int, got {type(message_index).__name__}")
        if type(sampled) is not bool:
            raise TypeError(f"sampled must be bool, got {type(sampled).__name__}")
        if type(content) is not bool:
            raise TypeError(f"content must be bool, got {type(content).__name__}")
        if token_id < 0 or token_id > np.iinfo(TOKEN_IDS_DTYPE).max:
            raise ValueError(f"token_id is outside the int32 token range: {token_id}")
        if message_index < -1 or message_index > np.iinfo(MESSAGE_INDICES_DTYPE).max:
            raise ValueError(f"message_index is outside the int32 attribution range: {message_index}")
        self._token_ids.append(token_id)
        self._message_indices.append(message_index)
        self._sampled_mask.append(sampled)
        self._is_content.append(content)

    def emit_tokens(
        self, token_ids: np.ndarray, message_index: int, *, sampled: bool, content: bool | np.ndarray
    ) -> None:
        require_1d_array("token_ids", token_ids, dtype=TOKEN_IDS_DTYPE, minimum=0)
        if type(message_index) is not int:
            raise TypeError(f"message_index must be int, got {type(message_index).__name__}")
        if type(sampled) is not bool:
            raise TypeError(f"sampled must be bool, got {type(sampled).__name__}")
        if message_index < -1 or message_index > np.iinfo(MESSAGE_INDICES_DTYPE).max:
            raise ValueError(f"message_index is outside the int32 attribution range: {message_index}")
        if isinstance(content, np.ndarray):
            require_1d_array("is_content", content, dtype=MASK_DTYPE)
            if content.size != token_ids.size:
                raise ValueError(f"is_content length {content.size} does not match token_ids length {token_ids.size}")
        elif type(content) is not bool:
            raise TypeError(f"content must be bool or a NumPy bool array, got {type(content).__name__}")

        self._token_ids.extend(token_ids)
        self._message_indices.extend_constant(message_index, token_ids.size)
        self._sampled_mask.extend_constant(sampled, token_ids.size)
        if isinstance(content, np.ndarray):
            self._is_content.extend(content)
        else:
            self._is_content.extend_constant(content, token_ids.size)

    def prepend_prior(self, token_ids: np.ndarray) -> None:
        self.emit_tokens(token_ids, -1, sampled=False, content=False)

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


def _single_token_sequence(name: str, value: object) -> np.ndarray:
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


def encode_token_ids(tokenizer: Any, text: str) -> np.ndarray:
    """Encode through a NumPy-capable tokenizer contract, rejecting lists."""
    if callable(tokenizer):
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_tensors="np")
        except (KeyError, NotImplementedError, TypeError, ValueError):
            encoded = None
        if isinstance(encoded, Mapping) and "input_ids" in encoded:
            return _single_token_sequence(type(tokenizer).__name__, encoded["input_ids"])

    encoded = tokenizer.encode(text, add_special_tokens=False)
    return _single_token_sequence(type(tokenizer).__name__, encoded)
