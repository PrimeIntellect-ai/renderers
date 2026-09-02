"""Regression coverage for generation prompts that pre-open ``<think>``."""

from renderers import (
    DeepSeekR1Renderer,
    DeepSeekV3Renderer,
    Qwen35Renderer,
    Qwen35RendererConfig,
)


class _TokenizerStub:
    """Small reversible tokenizer with single-ID special tokens."""

    name_or_path = "Qwen/Qwen3.5-4B"
    unk_token_id = -1

    def __init__(self):
        self._special_ids: dict[str, int] = {}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special_ids.setdefault(token, 1_000 + len(self._special_ids))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text.startswith("<｜") and text.endswith("｜>"):
            return [self.convert_tokens_to_ids(text)]
        return list(text.encode())

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return bytes(token for token in ids if token < 256).decode()


def test_qwen35_truncated_prefilled_think_is_reasoning():
    tokenizer = _TokenizerStub()
    renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=True))

    parsed = renderer.parse_response(tokenizer.encode("Let me work through this."))

    assert parsed.reasoning_content == "Let me work through this."
    assert parsed.content == ""
    assert parsed.tool_calls == []


def test_qwen35_without_think_prefill_keeps_plain_content():
    tokenizer = _TokenizerStub()
    renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))

    parsed = renderer.parse_response(tokenizer.encode("The answer is 4."))

    assert parsed.reasoning_content is None
    assert parsed.content == "The answer is 4."


def test_deepseek_r1_truncated_prefilled_think_is_reasoning():
    tokenizer = _TokenizerStub()
    renderer = DeepSeekR1Renderer(tokenizer)

    parsed = renderer.parse_response(tokenizer.encode("Let me work through this."))

    assert parsed.reasoning_content == "Let me work through this."
    assert parsed.content == ""
    assert parsed.tool_calls == []


def test_deepseek_r1_does_not_parse_tool_calls_inside_truncated_reasoning():
    tokenizer = _TokenizerStub()
    renderer = DeepSeekR1Renderer(tokenizer)
    tool_calls_begin = tokenizer.encode("<｜tool▁calls▁begin｜>")

    parsed = renderer.parse_response(
        tokenizer.encode("I might call a tool next.") + tool_calls_begin
    )

    assert parsed.reasoning_content == "I might call a tool next."
    assert parsed.content == ""
    assert parsed.tool_calls == []


def test_deepseek_v3_without_think_prefill_keeps_plain_content():
    tokenizer = _TokenizerStub()
    renderer = DeepSeekV3Renderer(tokenizer)

    parsed = renderer.parse_response(tokenizer.encode("The answer is 4."))

    assert parsed.reasoning_content is None
    assert parsed.content == "The answer is 4."
