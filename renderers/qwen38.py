"""Qwen3.8 Renderer — mirrors the Qwen3.8 Jinja chat template.

Qwen3.8 retains Qwen3.6's multimodal message grammar and JSON-safe tool
argument serialization, with three template changes:

- ``reasoning_effort`` can be ``xhigh`` (default), ``medium``, or ``low``;
  xhigh/low inject a matching instruction into the system prompt.
- ``preserve_thinking`` defaults to true, so historical reasoning blocks are
  retained unless explicitly disabled.
- Legacy inline ``<think>...</think>`` text is no longer split out of
  assistant ``content``; only ``reasoning_content`` populates the reasoning
  block.
"""

from __future__ import annotations

from renderers.base import Message
from renderers.configs import Qwen38RendererConfig
from renderers.qwen35 import Qwen35Renderer
from renderers.qwen36 import Qwen36Renderer


_REASONING_INSTRUCTIONS = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the "
        "task, validate key assumptions, consider plausible alternatives, and "
        "prioritize correctness, consistency, and clarity in the final answer."
    ),
    "medium": "",
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, "
        "moving directly to the conclusion without unnecessary elaboration."
    ),
}


class Qwen38Renderer(Qwen36Renderer):
    """Deterministic message-to-token renderer for Qwen3.8 models."""

    config: Qwen38RendererConfig
    _config_cls = Qwen38RendererConfig

    def _reasoning_instructions(self) -> str:
        if not self.config.enable_thinking:
            return ""
        return _REASONING_INSTRUCTIONS[self.config.reasoning_effort]

    def _omit_empty_system_message(self) -> bool:
        return True

    @staticmethod
    def _last_query_index(messages: list[Message]) -> int:
        last_query_index = Qwen35Renderer._last_query_index(messages)
        if last_query_index == len(messages):
            raise ValueError("No user query found in messages.")
        return last_query_index

    @staticmethod
    def _extract_assistant_parts(msg: Message, content: str) -> tuple[str, str]:
        reasoning_content = msg.get("reasoning_content")
        if not isinstance(reasoning_content, str):
            reasoning_content = ""
        return reasoning_content.strip(), content


__all__ = ["Qwen38Renderer"]
