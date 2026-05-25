"""Qwen3.6 Renderer — mirrors the Qwen3.6 Jinja chat template.

Delta vs Qwen3.5 (template line 122):

- Tool-call argument serialization changed to ``tojson`` for every non-string
  value. Bools now render as ``true``/``false`` (not ``True``/``False``) and
  ``None`` as ``null`` (not ``None``), fixing the single-turn extension-break
  mode where a boolean parameter's case drifted across a re-render.

Historical-thinking retention follows Qwen3.5's default (drop past
``<think>`` blocks). The upstream template carries a ``preserve_thinking``
Jinja toggle for the opposite polarity; on the renderer side that intent
maps to the renderer-agnostic ``preserve_all_thinking`` /
``preserve_thinking_between_tool_calls`` flags on
:class:`renderers.Qwen36RendererConfig`.

Everything else — tool system prompt, tool-call XML structure, thinking
markers, bridge logic, parser — is identical to Qwen3.5.
"""

from __future__ import annotations

import json
from typing import Any

from renderers.configs import Qwen36RendererConfig
from renderers.qwen35 import Qwen35Renderer


class Qwen36Renderer(Qwen35Renderer):
    """Deterministic message → token renderer for Qwen3.6 models."""

    _config_cls = Qwen36RendererConfig

    @staticmethod
    def _render_arg_value(arg_value: Any) -> str:
        if isinstance(arg_value, str):
            return arg_value
        return json.dumps(arg_value, ensure_ascii=False)
