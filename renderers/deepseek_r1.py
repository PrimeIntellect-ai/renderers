"""DeepSeek-R1 Renderer — the reasoning variant of the DeepSeek format.

R1 shares DeepSeek-V3's special tokens, message structure, and tool-call
wire format, so it subclasses :class:`renderers.deepseek_v3.DeepSeekV3Renderer`
and overrides only the two places its chat template diverges:

1. Generation prompt — R1 unconditionally prefills ``<think>\\n``
   (``<｜Assistant｜><think>\\n``) to trigger reasoning, where V3 emits a bare
   ``<｜Assistant｜>``. Handled by ``_GEN_THINK_PREFILL``.
2. Historical assistant turns — R1 strips the reasoning trace, keeping only
   the text after ``</think>`` (``content.split('</think>')[-1]``), where V3
   emits content verbatim. Handled by ``_prepare_assistant_content``.

Everything else — system handling, tool-call / tool-output rendering,
special-token resolution, and ``parse_response`` (``parse_deepseek_v3``,
shared) — is inherited unchanged.

Scope: full ``deepseek-ai/DeepSeek-R1`` and ``-R1-0528``. The R1-Distill
models (``DeepSeek-R1-Distill-Qwen/Llama``) use their base models'
tokenizers and route to the Qwen3 / Llama-3 renderers, not this one.
"""

from __future__ import annotations

from renderers.base import Message
from renderers.configs import DeepSeekR1RendererConfig
from renderers.deepseek_v3 import DeepSeekV3Renderer


class DeepSeekR1Renderer(DeepSeekV3Renderer):
    """Deterministic message → token renderer for DeepSeek-R1 models."""

    _config_cls: type = DeepSeekR1RendererConfig
    _implied_thinking_retention = "template"
    _GEN_THINK_PREFILL: str = "<think>\n"

    def _prepare_assistant_content(
        self, msg: Message, *, preserve_thinking: bool = False
    ) -> str:
        """Assistant content with R1's default history-stripping behaviour.

        Structured ``thinking``/``text`` parts are reconstructed inline first
        so the same ``</think>`` split applies. When ``preserve_thinking`` is
        true, this renderer-level override preserves inline thinking or
        reconstructs it from the separate ``reasoning_content`` field.
        """
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "thinking":
                    parts.append(f"<think>{p.get('thinking', '')}</think>")
                elif p.get("type") == "text":
                    parts.append(p.get("text", ""))
            content = "".join(parts)
        reasoning = msg.get("reasoning_content")
        if preserve_thinking:
            if "</think>" in content:
                return content
            if isinstance(reasoning, str) and reasoning:
                if content:
                    return f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
                return f"<think>\n{reasoning.strip()}\n</think>"
            return content
        if "</think>" in content:
            content = content.split("</think>")[-1]
        return content
