"""Laguna S-2.1 Renderer — a larger sibling of Laguna XS-2.1.

S-2.1 shares XS-2.1's tokenizer and token format, so this is a thin subclass of
:class:`renderers.laguna_xs2.LagunaXS21Renderer`; see that module for the shared
format. The template delta is two thinking kwargs: ``enable_thinking`` defaults
to ``True`` (XS-2.1 defaults ``False``), and ``preserve_thinking`` widens the
reasoning-display gate to ``enable_thinking or preserve_thinking``.
"""

from __future__ import annotations

from renderers.base import Tokenizer
from renderers.configs import LagunaS21RendererConfig
from renderers.laguna_xs2 import LagunaXS21Renderer


class LagunaS21Renderer(LagunaXS21Renderer):
    """Mirrors the ``poolside/Laguna-S-2.1`` chat template."""

    # Narrows the inherited attribute so the S-2.1-only ``preserve_thinking``
    # resolves; ``__init__`` always stores an S-2.1 config.
    config: LagunaS21RendererConfig

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: LagunaS21RendererConfig | None = None,
    ):
        super().__init__(tokenizer, config or LagunaS21RendererConfig())

    def _render_history_reasoning(self) -> bool:
        return self.config.enable_thinking or self.config.preserve_thinking
