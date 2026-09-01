"""GPT-OSS renderer availability boundary."""

from __future__ import annotations

from renderers.base import Tokenizer
from renderers.configs import GptOssRendererConfig


_TYPED_ARRAY_ERROR = (
    "GptOssRenderer requires an openai-harmony fixed-width NumPy token ABI; "
    "the installed Harmony API materializes Python token lists"
)


class GptOssRenderer:
    """Fail closed until Harmony can preserve typed token custody."""

    def __init__(self, tokenizer: Tokenizer, config: GptOssRendererConfig | None = None) -> None:
        del tokenizer, config
        raise RuntimeError(_TYPED_ARRAY_ERROR)


__all__ = ["GptOssRenderer"]
