"""DeepSeek-V4.1-Flash renderer, for text conversations without tools.

The checkpoint's source of truth is ``encoding/encoding.py`` in the model
repository.  For plain chats it is V4's encoder with three changes, applied
here on top of :class:`DeepSeekV4Renderer`:

* a ``<｜System｜>`` token leads the conversation whenever it opens with a
  reasoning-effort prefix or a system message;
* the reasoning effort is a numeric budget,
  ``Reasoning Effort: {budget} (range 1-100, ...)``, rendered in thinking mode
  only and defaulting to ``high`` (75);
* thinking mode is the default.

Tool calls (V4.1 renamed the DSML tags) and mid-conversation system messages
are not supported yet.
"""

from __future__ import annotations

from renderers.base import Message, RenderedTokens, ToolSpec
from renderers.configs import DeepSeekV41RendererConfig
from renderers.deepseek_v4 import DeepSeekV4Renderer

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

    def __init__(self, tokenizer, config: DeepSeekV41RendererConfig | None = None):
        super().__init__(tokenizer, config or DeepSeekV41RendererConfig())

    def _reasoning_effort_prompt(self) -> str:
        effort = self.config.reasoning_effort
        budget = _EFFORT_BUDGETS[effort] if isinstance(effort, str) else effort
        return (
            f"{_SYSTEM}Reasoning Effort: {budget} (range 1-100, "
            "the higher the value, the more thorough the reasoning)\n\n"
        )

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if tools:
            raise NotImplementedError(
                "the DeepSeek-V4.1 renderer does not render tools"
            )
        if any(message["role"] == "system" for message in messages[1:]):
            raise NotImplementedError(
                "the DeepSeek-V4.1 renderer does not render mid-conversation system messages"
            )
        messages = [_split_inline_reasoning(message) for message in messages]
        if (
            messages
            and messages[0]["role"] == "system"
            and not self.config.enable_thinking
        ):
            # Without an effort prefix, the system message itself opens with the System token.
            first = messages[0].copy()
            if not isinstance(first["content"], str):
                raise NotImplementedError(
                    "the DeepSeek-V4.1 renderer renders text-only system messages"
                )
            first["content"] = _SYSTEM + first["content"]
            messages = [first, *messages[1:]]
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
        # V4's extension path renders the new turn with V4's tags; fall back to a full render.
        return None


__all__ = ["DeepSeekV41Renderer"]
