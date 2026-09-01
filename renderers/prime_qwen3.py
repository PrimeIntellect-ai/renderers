"""PrimeIntellect Qwen3 renderer with exact chat-template tokenization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    Tokenizer,
    _get_offset_tokenizer,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import PrimeQwen3RendererConfig
from renderers.parsing import parse_qwen35
from renderers.token_arrays import RenderedTokenBuilder, TextSegmentBuilder

_DEFAULT_TOOL_SYSTEM = "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."
_TOOLS_HEADER = "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_FOOTER = (
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>"
    "\n<parameter=example_parameter_1>\nvalue_1\n</parameter>"
    "\n<parameter=example_parameter_2>"
    "\nThis is the value for the second parameter"
    "\nthat can span\nmultiple lines\n</parameter>"
    "\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:"
    "\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags"
    "\n- Required parameters MUST be specified"
    "\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after"
    "\n- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function calls"
    "\n</IMPORTANT>"
)


def _render_extra_keys(value: Any, handled_keys: frozenset[str]) -> str:
    if not isinstance(value, Mapping):
        return ""

    rendered: list[str] = []
    for key, item in value.items():
        if key in handled_keys:
            continue
        item_text = (
            json.dumps(item, ensure_ascii=False)
            if isinstance(item, Mapping)
            or (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
            )
            else str(item)
        )
        rendered.append(f"\n<{key}>{item_text}</{key}>")
    return "".join(rendered)


def _tool_definition(tool: ToolSpec) -> str:
    raw_tool: Any = tool
    if isinstance(raw_tool, Mapping) and isinstance(raw_tool.get("function"), Mapping):
        raw_tool = raw_tool["function"]
    if not isinstance(raw_tool, Mapping):
        raise TypeError("Tool definitions must be mappings.")

    rendered = "\n<function>\n<name>" + str(raw_tool.get("name", "")) + "</name>"
    if "description" in raw_tool:
        rendered += (
            "\n<description>" + str(raw_tool["description"]).strip() + "</description>"
        )
    rendered += "\n<parameters>"

    parameters = raw_tool.get("parameters")
    if isinstance(parameters, Mapping):
        properties = parameters.get("properties")
        if isinstance(properties, Mapping):
            for param_name, param_fields in properties.items():
                rendered += "\n<parameter>\n<name>" + str(param_name) + "</name>"
                if isinstance(param_fields, Mapping):
                    if "type" in param_fields:
                        rendered += "\n<type>" + str(param_fields["type"]) + "</type>"
                    if "description" in param_fields:
                        rendered += (
                            "\n<description>"
                            + str(param_fields["description"]).strip()
                            + "</description>"
                        )
                    rendered += _render_extra_keys(
                        param_fields, frozenset({"name", "type", "description"})
                    )
                rendered += "\n</parameter>"
        rendered += _render_extra_keys(parameters, frozenset({"type", "properties"}))

    rendered += "\n</parameters>"
    rendered += _render_extra_keys(
        raw_tool, frozenset({"type", "name", "description", "parameters"})
    )
    rendered += "\n</function>"
    return rendered


class PrimeQwen3Renderer:
    """Renderer for PrimeIntellect/Qwen3-0.6B and Qwen3-1.7B."""

    def __init__(
        self, tokenizer: Tokenizer, config: PrimeQwen3RendererConfig | None = None
    ):
        self._tokenizer = tokenizer
        self._offset_tokenizer = _get_offset_tokenizer(tokenizer)
        self.config = config or PrimeQwen3RendererConfig()
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )

        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")

    def _token_id(self, token: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(token_id, int) and token_id != self._tokenizer.unk_token_id, (
            f"Token {token!r} not found in tokenizer vocabulary"
        )
        return token_id

    @staticmethod
    def _content(message: Message) -> str:
        content = message.get("content")
        if content is None:
            return ""
        if not isinstance(content, str):
            raise TypeError("PrimeQwen3Renderer only supports string message content.")
        return content

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        builder = RenderedTokenBuilder(
            self._tokenizer, offset_tokenizer=self._offset_tokenizer
        )
        first_is_system = messages[0].get("role") == "system"
        loop_start = 1 if first_is_system else 0

        if first_is_system or tools:
            system_index = 0 if first_is_system else -1
            builder.emit_special(
                self._im_start, system_index, is_sampled=False, is_content=False
            )
            system_segments = TextSegmentBuilder()
            system_segments.append("system\n", is_content=False)
            if first_is_system:
                system_segments.append(self._content(messages[0]), is_content=True)
            else:
                system_segments.append(_DEFAULT_TOOL_SYSTEM, is_content=False)

            if tools:
                tools_text = _TOOLS_HEADER
                for tool in tools:
                    tools_text += _tool_definition(tool)
                tools_text += _TOOLS_FOOTER
                system_segments.append(tools_text, is_content=False)

            builder.emit_text_segments(
                system_segments.finish(), system_index, is_sampled=False
            )
            builder.emit_special(
                self._im_end, system_index, is_sampled=False, is_content=False
            )
            builder.emit_text("\n", system_index, is_sampled=False, is_content=False)

        loop_messages = messages[loop_start:]
        for loop_index, message in enumerate(loop_messages):
            message_index = loop_start + loop_index
            role = message.get("role", "")
            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    self._render_assistant_tool_calls(message, message_index, builder)
                else:
                    self._render_assistant(message, message_index, builder)
            elif role == "tool":
                opens_group = (
                    loop_index > 0
                    and loop_messages[loop_index - 1].get("role") != "tool"
                )
                closes_group = (
                    loop_index == len(loop_messages) - 1
                    or loop_messages[loop_index + 1].get("role") != "tool"
                )
                self._render_tool(
                    message,
                    message_index,
                    builder,
                    opens_group=opens_group,
                    closes_group=closes_group,
                )
            else:
                self._render_history_message(message, message_index, builder)

        if add_generation_prompt:
            self._render_generation_prompt(builder)

        attribution_available = self._offset_tokenizer is not None
        return builder.finish(
            message_roles=[message.get("role") or "" for message in messages],
            message_tool_names=extract_message_tool_names(messages),
            sampled_available=attribution_available,
            content_available=attribution_available,
        )

    def _render_history_message(
        self, message: Message, message_index: int, builder: RenderedTokenBuilder
    ) -> None:
        role = message.get("role", "")
        builder.emit_special(
            self._im_start, message_index, is_sampled=False, is_content=False
        )
        segments = TextSegmentBuilder()
        segments.append(role + "\n", is_content=False)
        segments.append(self._content(message), is_content=True)
        builder.emit_text_segments(segments.finish(), message_index, is_sampled=False)
        builder.emit_special(
            self._im_end, message_index, is_sampled=False, is_content=False
        )
        builder.emit_text("\n", message_index, is_sampled=False, is_content=False)

    def _render_assistant(
        self, message: Message, message_index: int, builder: RenderedTokenBuilder
    ) -> None:
        content = self._content(message)
        builder.emit_special(
            self._im_start, message_index, is_sampled=False, is_content=False
        )

        if "reasoning_content" in message:
            builder.emit_text(
                "assistant\n", message_index, is_sampled=False, is_content=False
            )
            builder.emit_special(
                self._think, message_index, is_sampled=True, is_content=True
            )
            reasoning = message.get("reasoning_content")
            if reasoning:
                builder.emit_text(
                    str(reasoning).strip(),
                    message_index,
                    is_sampled=True,
                    is_content=True,
                )
            builder.emit_special(
                self._think_end, message_index, is_sampled=True, is_content=True
            )
            if content.strip():
                builder.emit_text(
                    "\n" + content.strip(),
                    message_index,
                    is_sampled=True,
                    is_content=True,
                )
        else:
            segments = TextSegmentBuilder()
            segments.append("assistant\n", is_content=False)
            segments.append(content, is_content=True)
            builder.emit_assistant_segments(segments.finish(), message_index)

        builder.emit_special(
            self._im_end, message_index, is_sampled=True, is_content=True
        )
        builder.emit_text("\n", message_index, is_sampled=False, is_content=False)

    def _render_assistant_tool_calls(
        self, message: Message, message_index: int, builder: RenderedTokenBuilder
    ) -> None:
        builder.emit_special(
            self._im_start, message_index, is_sampled=False, is_content=False
        )
        content = message.get("content")
        trimmed_content = content.strip() if isinstance(content, str) else ""
        opener_segments = TextSegmentBuilder()
        opener_segments.append("assistant\n", is_content=False)
        if trimmed_content:
            opener_segments.append(trimmed_content + "\n\n", is_content=True)
        builder.emit_assistant_segments(opener_segments.finish(), message_index)

        tool_calls = message.get("tool_calls") or []
        for tool_call_index, tool_call in enumerate(tool_calls):
            raw_call: Any = tool_call
            if isinstance(raw_call, Mapping) and isinstance(
                raw_call.get("function"), Mapping
            ):
                raw_call = raw_call["function"]
            if not isinstance(raw_call, Mapping):
                raise TypeError("Tool calls must be mappings.")

            builder.emit_special(
                self._tool_call, message_index, is_sampled=True, is_content=True
            )
            call_text = "\n<function=" + str(raw_call.get("name", "")) + ">\n"
            if "arguments" in raw_call:
                arguments = raw_call["arguments"]
                # OpenAI canonical form serializes arguments as a JSON string;
                # degrade malformed payloads to no parameters like Qwen3.5.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, Mapping):
                    arguments = {}
                for argument_name, argument_value in arguments.items():
                    value_text = (
                        json.dumps(argument_value, ensure_ascii=False)
                        if isinstance(argument_value, Mapping)
                        or (
                            isinstance(argument_value, Sequence)
                            and not isinstance(argument_value, (str, bytes, bytearray))
                        )
                        else str(argument_value)
                    )
                    call_text += (
                        "<parameter="
                        + str(argument_name)
                        + ">\n"
                        + value_text
                        + "\n</parameter>\n"
                    )
            call_text += "</function>\n"
            builder.emit_text(
                call_text, message_index, is_sampled=True, is_content=True
            )
            builder.emit_special(
                self._tool_call_end, message_index, is_sampled=True, is_content=True
            )
            if tool_call_index < len(tool_calls) - 1:
                builder.emit_text("\n", message_index, is_sampled=True, is_content=True)

        builder.emit_special(
            self._im_end, message_index, is_sampled=True, is_content=True
        )
        builder.emit_text("\n", message_index, is_sampled=False, is_content=False)

    def _render_tool(
        self,
        message: Message,
        message_index: int,
        builder: RenderedTokenBuilder,
        *,
        opens_group: bool,
        closes_group: bool,
    ) -> None:
        if opens_group:
            builder.emit_special(
                self._im_start, message_index, is_sampled=False, is_content=False
            )
            builder.emit_text(
                "user\n", message_index, is_sampled=False, is_content=False
            )

        builder.emit_special(
            self._tool_response, message_index, is_sampled=False, is_content=False
        )
        segments = TextSegmentBuilder()
        segments.append("\n", is_content=False)
        segments.append(self._content(message), is_content=True)
        segments.append("\n", is_content=False)
        builder.emit_text_segments(segments.finish(), message_index, is_sampled=False)
        builder.emit_special(
            self._tool_response_end, message_index, is_sampled=False, is_content=False
        )
        builder.emit_text("\n", message_index, is_sampled=False, is_content=False)

        if closes_group:
            builder.emit_special(
                self._im_end, message_index, is_sampled=False, is_content=False
            )
            builder.emit_text("\n", message_index, is_sampled=False, is_content=False)

    def _render_generation_prompt(self, builder: RenderedTokenBuilder) -> None:
        builder.emit_special(self._im_start, -1, is_sampled=False, is_content=False)
        builder.emit_text("assistant\n", -1, is_sampled=False, is_content=False)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> np.ndarray:
        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def parse_response(
        self, token_ids: np.ndarray, *, tools: list[ToolSpec] | None = None
    ) -> ParsedResponse:
        return parse_qwen35(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
            tools=tools,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end, self._endoftext]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: np.ndarray,
        previous_completion_ids: np.ndarray,
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002
    ) -> RenderedTokens | None:
        if (
            len(previous_prompt_ids) == 0
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention, new_messages
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._im_end, self._endoftext},
            synthesize_close=self._im_end,
        )
        if previous_ids is None:
            return None

        builder = RenderedTokenBuilder(
            self._tokenizer, offset_tokenizer=self._offset_tokenizer
        )
        builder.prepend_prior(previous_ids)
        builder.emit_text("\n", -1, is_sampled=False, is_content=False)

        for message_index, message in enumerate(new_messages):
            role = message.get("role", "")
            if role == "tool":
                opens_group = (
                    message_index == 0
                    or new_messages[message_index - 1].get("role") != "tool"
                )
                closes_group = (
                    message_index == len(new_messages) - 1
                    or new_messages[message_index + 1].get("role") != "tool"
                )
                self._render_tool(
                    message,
                    message_index,
                    builder,
                    opens_group=opens_group,
                    closes_group=closes_group,
                )
            else:
                self._render_history_message(message, message_index, builder)

        self._render_generation_prompt(builder)
        attribution_available = self._offset_tokenizer is not None
        return builder.finish(
            message_roles=[message.get("role") or "" for message in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
            sampled_available=attribution_available,
            content_available=attribution_available,
        )


__all__ = ["PrimeQwen3Renderer"]
