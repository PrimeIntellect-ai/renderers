"""DeepSeek-V4.1-Flash renderer.

The checkpoint's source of truth is ``encoding/encoding.py`` in the model
repository.  It is V4's encoder with these changes, applied here on top of
:class:`DeepSeekV4Renderer`:

* a ``<｜System｜>`` token leads the conversation whenever it opens with a
  reasoning-effort prefix or a system message (including the empty one that
  carries tools when the conversation has no leading system message);
* the reasoning effort is a numeric budget,
  ``Reasoning Effort: {budget} (range 1-100, ...)``, rendered in thinking mode
  only and defaulting to ``high`` (75);
* thinking mode is the default;
* DSML tags have space-prefixed names: ``<｜DSML｜ calls>``,
  ``<｜DSML｜ invoke>`` and ``<｜DSML｜ parameter>``;
* tools attach only to a leading system message;
* a user turn holding only tool results does not start a new turn, so
  dropped historical reasoning stops at the last real user message;
* JSON-string tool arguments may be double-encoded, and arguments that do not
  decode to an object render as a single ``arguments`` parameter.

Mid-conversation system messages are not supported yet.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from renderers.base import Message, RenderedTokens, ToolSpec
from renderers.configs import DeepSeekV41RendererConfig
from renderers.deepseek_v4 import _DSML, DeepSeekV4Renderer, _LogicalMessage
from renderers.tagged import DEEPSEEK_V41_GRAMMAR

_SYSTEM = "<｜System｜>"
_THINK_START = "<think>"
_THINK_END = "</think>"
_EFFORT_BUDGETS = {"low": 50, "high": 75, "max": 100}


def _split_inline_reasoning(message: Message) -> Message:
    """Move a leading ``<think>...</think>`` out of an assistant's content into ``reasoning_content``.

    Many SFT datasets keep the reasoning inline in the content; rendered as is,
    it would follow an empty think block instead of filling it.
    """
    content = message.get("content")
    if (
        message["role"] != "assistant"
        or message.get("reasoning_content") is not None
        or not isinstance(content, str)
    ):
        return message
    if not content.startswith(_THINK_START) or _THINK_END not in content:
        return message
    reasoning, content = content.removeprefix(_THINK_START).split(_THINK_END, 1)
    split = message.copy()
    split["reasoning_content"] = reasoning
    split["content"] = content
    return split


class DeepSeekV41Renderer(DeepSeekV4Renderer):
    """Renderer for ``deepseek-ai/DeepSeek-V4.1-Flash``."""

    _calls_tag = f"{_DSML} calls"
    _invoke_tag = f"{_DSML} invoke"
    _parameter_tag = f"{_DSML} parameter"
    _grammar = DEEPSEEK_V41_GRAMMAR

    def __init__(self, tokenizer, config: DeepSeekV41RendererConfig | None = None):
        super().__init__(tokenizer, config or DeepSeekV41RendererConfig())

    def _preamble(self, *, opens_with_system: bool) -> str:
        if not self.config.enable_thinking:
            return _SYSTEM if opens_with_system else ""
        effort = self.config.reasoning_effort
        budget = _EFFORT_BUDGETS[effort] if isinstance(effort, str) else effort
        return (
            f"{_SYSTEM}Reasoning Effort: {budget} (range 1-100, "
            "the higher the value, the more thorough the reasoning)\n\n"
        )

    def _is_turn_start(self, message: _LogicalMessage) -> bool:
        if message.role == "user":
            return any(block.kind != "tool_result" for block in message.blocks)
        return super()._is_turn_start(message)

    def _tool_target(self, messages: list[_LogicalMessage]) -> int | None:
        return 0 if messages[0].role == "system" else None

    @staticmethod
    def _tool_call_arguments(raw_arguments: Any) -> Any:
        arguments = raw_arguments
        # A JSON string, possibly double-encoded.
        for _ in range(2):
            if not isinstance(arguments, str):
                break
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                break
        if isinstance(arguments, Mapping):
            return dict(arguments)
        return {"arguments": raw_arguments}

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if any(message["role"] == "system" for message in messages[1:]):
            raise NotImplementedError(
                "the DeepSeek-V4.1 renderer does not render mid-conversation system messages"
            )
        messages = [_split_inline_reasoning(message) for message in messages]
        return super().render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if any(message["role"] == "system" for message in new_messages):
            return None
        return super().bridge_to_next_turn(
            previous_prompt_ids,
            previous_completion_ids,
            new_messages,
            tools=tools,
        )


__all__ = ["DeepSeekV41Renderer"]
