"""Tokenizer contracts and the lightweight ``tokenizers`` adapter.

The renderer core only needs four operations: encode, decode, special-token
lookup, and encoding with character offsets. Keeping that surface structural
lets callers pass Hugging Face tokenizers, engine-owned tokenizers, or the
Rust-backed :class:`tokenizers.Tokenizer` without importing ``transformers``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict

from tokenizers import Tokenizer


class TokenizerEncoding(TypedDict, total=False):
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]


class TokenizerLike(Protocol):
    """Minimal tokenizer surface used by model-specific renderers."""

    name_or_path: str
    unk_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str: ...

    def convert_tokens_to_ids(self, token: str) -> int | None: ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs: Any,
    ) -> TokenizerEncoding: ...


class ChatTemplateTokenizerLike(TokenizerLike, Protocol):
    """Extra Hugging Face surface required by ``DefaultRenderer``."""

    eos_token_id: int | None
    all_special_tokens: list[str]

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any: ...


class TokenizersTokenizer:
    """HF-shaped adapter around the standalone Rust ``tokenizers`` package.

    ``tokenizers.Tokenizer.encode`` returns an ``Encoding`` object, while the
    established renderer contract expects a list of IDs and a mapping-shaped
    offset result. This adapter performs only that shape conversion; it does
    not implement chat templates or multimodal processing.
    """

    def __init__(self, tokenizer: Tokenizer, *, name_or_path: str):
        self._tokenizer = tokenizer
        self.name_or_path = name_or_path
        unk_token = getattr(tokenizer.model, "unk_token", None)
        self.unk_token_id = (
            tokenizer.token_to_id(unk_token) if isinstance(unk_token, str) else None
        )

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, *, revision: str | None = None
    ) -> "TokenizersTokenizer":
        path = Path(model_name_or_path)
        if path.is_dir():
            tokenizer_path = path / "tokenizer.json"
            if not tokenizer_path.is_file():
                raise FileNotFoundError(
                    f"No tokenizer.json found in local tokenizer directory {path}."
                )
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        elif path.is_file():
            tokenizer = Tokenizer.from_file(str(path))
        else:
            tokenizer = Tokenizer.from_pretrained(
                model_name_or_path,
                revision=revision or "main",
            )
        return cls(tokenizer, name_or_path=model_name_or_path)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(
            self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            ).ids
        )

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(
            ids,
            skip_special_tokens=skip_special_tokens,
        )

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs: Any,
    ) -> TokenizerEncoding:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported tokenizer arguments: {unexpected}")
        encoding = self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
        result: TokenizerEncoding = {"input_ids": list(encoding.ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = list(encoding.offsets)
        return result
