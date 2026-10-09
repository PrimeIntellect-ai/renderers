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

    supports_reasoning_content = True
    _config_cls: type = DeepSeekR1RendererConfig
    _implied_thinking_retention = "template"
    _GEN_THINK_PREFILL: str = "<think>\n"

    def _prepare_assistant_content(self, msg: Message) -> str:
        """Assistant content with the reasoning trace stripped, mirroring the
        R1 template's ``content.split('</think>')[-1]`` on historical turns.

        Canonical ``reasoning_content`` is accepted by this reasoning-capable
        renderer, but the R1 template intentionally drops historical reasoning.
        The visible ``content`` is therefore emitted unchanged.
        """
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
            content = "".join(parts)
        return content
