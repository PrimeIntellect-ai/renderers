"""Qwen3.8 Renderer — mirrors the Qwen3.8 Jinja chat template.

Qwen3.8 retains Qwen3.6's multimodal message grammar and JSON-safe tool
argument serialization, with two template changes:

- ``reasoning_effort`` can be ``xhigh`` (default), ``medium``, or ``low``;
  xhigh/low inject a matching instruction into the system prompt.
- ``preserve_thinking`` defaults to true, so historical reasoning blocks are
  retained unless explicitly disabled.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from renderers.base import Message
from renderers.configs import Qwen38RendererConfig
from renderers.qwen36 import Qwen36Renderer
from renderers.qwen3_vl import _is_image_part, _is_video_part


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
    def _iter_content_parts(content: list[Any]) -> Iterable[Any]:
        """Keep media and string-valued text in every rendering path."""
        for item in content:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict) and (
                _is_image_part(item)
                or _is_video_part(item)
                or isinstance(item.get("text"), str)
            ):
                yield item

    @staticmethod
    def _is_user_query_message(msg: Message) -> bool:
        if msg.get("role") != "user":
            return False
        content = Qwen38Renderer._render_content(msg.get("content")).strip()
        return not (
            content.startswith("<tool_response>")
            and content.endswith("</tool_response>")
        )

    @staticmethod
    def _last_query_index(messages: list[Message]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            if Qwen38Renderer._is_user_query_message(messages[i]):
                return i
        raise ValueError("No user query found in messages.")


__all__ = ["Qwen38Renderer"]
