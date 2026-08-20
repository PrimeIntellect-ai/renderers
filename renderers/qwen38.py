"""Qwen3.8 Renderer — mirrors the Qwen3.8 Jinja chat template.

Qwen3.8 extends the Qwen3.5 template with three deltas (see the unified
diff against ``Qwen/Qwen3.5-9B``'s ``chat_template.jinja``):

- **Reasoning-effort instructions.** When ``enable_thinking`` is truthy, the
  template resolves ``reasoning_effort`` (default ``xhigh``) into a
  ``reasoning_instructions`` string that is injected into the leading system
  message — both the tools path (right after ``system\n``, before ``# Tools``)
  and the no-tools path (prefixed to caller system content, or as a standalone
  system message when the caller supplied none). ``medium`` produces no
  instructions; ``xhigh`` / ``low`` produce the pinned strings below.
- **Thinking preservation defaults on.** ``preserve_thinking`` is undefined
  in the template and treated as ``True``, so historical `` thinking`` blocks
  are kept on every assistant turn (not just after the last real user query).
  ``Qwen38RendererConfig.preserve_thinking`` therefore defaults to ``True``.
- **No `` response`` fallback.** The template no longer derives reasoning from
  ``content`` when ``reasoning_content`` is absent; it relies solely on the
  ``reasoning_content`` field.

Tool-call argument serialization (``str`` verbatim, everything else compact
JSON) and the XML tool-call structure are identical to Qwen3.6, so
``_render_arg_value`` matches ``Qwen36Renderer``.
"""

from __future__ import annotations

import json
from typing import Any

from renderers.configs import Qwen38RendererConfig
from renderers.qwen35 import (
    Qwen35Renderer,
    _TOOLS_FOOTER,
    _TOOLS_HEADER,
    _TOOLS_INSTRUCTIONS,
)

# Pinned reasoning-effort instruction strings (must match the Jinja template
# exactly). ``medium`` yields no instructions.
_REASONING_INSTRUCTIONS: dict[str, str] = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the "
        "task, validate key assumptions, consider plausible alternatives, and "
        "prioritize correctness, consistency, and clarity in the final answer."
    ),
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, "
        "moving directly to the conclusion without unnecessary elaboration."
    ),
}


class Qwen38Renderer(Qwen35Renderer):
    """Deterministic message → token renderer for Qwen3.8 models."""

    _config_cls = Qwen38RendererConfig

    def __init__(self, tokenizer, config=None, *, processor=None):
        super().__init__(tokenizer, config, processor=processor)
        # ``enable_thinking`` is resolved to a concrete bool by the parent
        # ``__init__``; compute the template's reasoning-instruction string once.
        self.reasoning_instructions = self._reasoning_instructions()

    def _reasoning_instructions(self) -> str:
        """The template's ``reasoning_instructions`` for the resolved config.

        Empty when thinking is disabled or ``reasoning_effort`` is ``medium``.
        ``None`` on the config means the template default (``xhigh``).
        """
        if not self.config.enable_thinking:
            return ""
        effort = self.config.reasoning_effort or "xhigh"
        return _REASONING_INSTRUCTIONS.get(effort, "")

    @staticmethod
    def _render_arg_value(arg_value: Any) -> str:
        # Qwen3.8: ``args_value | string if args_value is string else
        # args_value | tojson | safe`` — same effective behavior as Qwen3.6.
        if isinstance(arg_value, str):
            return arg_value
        return json.dumps(arg_value, ensure_ascii=False)

    def _extract_reasoning(self, msg, content):
        # Qwen3.8 dropped the `` response``-in-content fallback; reasoning
        # comes solely from the ``reasoning_content`` field.
        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        return reasoning_content.strip(), content

    def _emit_system_and_tools(
        self,
        messages,
        tools,
        *,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        first_is_system = messages[0].get("role") == "system"
        ri = self.reasoning_instructions

        if tools:
            # System message index for attribution
            sys_idx = 0 if first_is_system else -1

            emit_special(self._im_start, sys_idx, is_sampled=False, is_content=False)
            # Reasoning instructions + tools header / footer / instructions and
            # the JSON tool specs are template-injected scaffold; only caller
            # system content is body (``is_content=True``).
            segments: list[tuple[str, bool]] = [("system\n", False)]
            if ri:
                segments.append((ri + "\n\n", False))
            segments.append((_TOOLS_HEADER, False))
            for tool in tools:
                segments.append(("\n" + json.dumps(tool, ensure_ascii=False), False))
            segments.append((_TOOLS_FOOTER, False))
            segments.append((_TOOLS_INSTRUCTIONS, False))
            if first_is_system:
                sys_content = self._render_content(messages[0].get("content")).strip()
                if sys_content:
                    segments.append(("\n\n", False))
                    segments.append((sys_content, True))
            emit_text_segments(segments, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False, is_content=False)
            emit_text("\n", sys_idx, is_sampled=False, is_content=False)
        elif first_is_system:
            sys_content = self._render_content(messages[0].get("content")).strip()
            if sys_content or ri:
                emit_special(self._im_start, 0, is_sampled=False, is_content=False)
                segments = [("system\n", False)]
                if ri:
                    segments.append((ri + "\n\n", False))
                if sys_content:
                    segments.append((sys_content, True))
                emit_text_segments(segments, 0, is_sampled=False)
                emit_special(self._im_end, 0, is_sampled=False, is_content=False)
                emit_text("\n", 0, is_sampled=False, is_content=False)
        elif ri:
            # Template emits a standalone system message carrying only the
            # reasoning instructions when the caller supplied no system message.
            emit_special(self._im_start, -1, is_sampled=False, is_content=False)
            emit_text_segments(
                [("system\n", False), (ri, False)],
                -1,
                is_sampled=False,
            )
            emit_special(self._im_end, -1, is_sampled=False, is_content=False)
            emit_text("\n", -1, is_sampled=False, is_content=False)


__all__ = ["Qwen38Renderer"]
