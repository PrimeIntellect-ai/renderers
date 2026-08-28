"""PrimeIntellect Qwen3 renderer with exact chat-template tokenization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    Tokenizer,
    _content_mask_or_empty,
    _get_offset_tokenizer,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import PrimeQwen3RendererConfig
from renderers.parsing import parse_qwen35

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


def _render_extra_keys(
    value: Any,
    handled_keys: frozenset[str],
) -> str:
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
                        param_fields,
                        frozenset({"name", "type", "description"}),
                    )
                rendered += "\n</parameter>"
        rendered += _render_extra_keys(
            parameters,
            frozenset({"type", "properties"}),
        )

    rendered += "\n</parameters>"
    rendered += _render_extra_keys(
        raw_tool,
        frozenset({"type", "name", "description", "parameters"}),
    )
    rendered += "\n</function>"
    return rendered


class _TokenBuilder:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.message_indices: list[int] = []
        self.sampled_mask: list[bool] = []
        self.is_content: list[bool] = []

    def emit_special(
        self,
        token_id: int,
        message_index: int,
        *,
        sampled: bool,
        content: bool,
    ) -> None:
        self.token_ids.append(token_id)
        self.message_indices.append(message_index)
        self.sampled_mask.append(sampled)
        self.is_content.append(content)

    def emit_text(
        self,
        text: str,
        message_index: int,
        *,
        sampled: bool,
        content: bool,
    ) -> None:
        if not text:
            return
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.token_ids.extend(token_ids)
        self.message_indices.extend([message_index] * len(token_ids))
        self.sampled_mask.extend([sampled] * len(token_ids))
        self.is_content.extend([content] * len(token_ids))

    def emit_segments(
        self,
        segments: list[tuple[str, bool]],
        message_index: int,
        *,
        sampled: bool,
    ) -> None:
        for token_id, content in attribute_text_segments(self.tokenizer, segments):
            self.token_ids.append(token_id)
            self.message_indices.append(message_index)
            self.sampled_mask.append(sampled)
            self.is_content.append(content)

    def emit_assistant_segments(
        self,
        segments: list[tuple[str, bool]],
        message_index: int,
    ) -> None:
        for token_id, content in attribute_text_segments(self.tokenizer, segments):
            self.token_ids.append(token_id)
            self.message_indices.append(message_index)
            self.sampled_mask.append(content)
            self.is_content.append(content)

    def prepend_prior(self, token_ids: list[int]) -> None:
        self.token_ids.extend(token_ids)
        self.message_indices.extend([-1] * len(token_ids))
        self.sampled_mask.extend([False] * len(token_ids))
        self.is_content.extend([False] * len(token_ids))


class PrimeQwen3Renderer:
    """Renderer for PrimeIntellect/Qwen3-0.6B and Qwen3-1.7B."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: PrimeQwen3RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or PrimeQwen3RendererConfig()
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config,
            "all",
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

        builder = _TokenBuilder(self._tokenizer)
        first_is_system = messages[0].get("role") == "system"
        loop_start = 1 if first_is_system else 0

        if first_is_system or tools:
            system_index = 0 if first_is_system else -1
            builder.emit_special(
                self._im_start,
                system_index,
                sampled=False,
                content=False,
            )
            system_segments: list[tuple[str, bool]] = [("system\n", False)]
            if first_is_system:
                system_segments.append((self._content(messages[0]), True))
            else:
                system_segments.append((_DEFAULT_TOOL_SYSTEM, False))

            if tools:
                tools_text = _TOOLS_HEADER
                for tool in tools:
                    tools_text += _tool_definition(tool)
                tools_text += _TOOLS_FOOTER
                system_segments.append((tools_text, False))

            builder.emit_segments(system_segments, system_index, sampled=False)
            builder.emit_special(
                self._im_end,
                system_index,
                sampled=False,
                content=False,
            )
            builder.emit_text(
                "\n",
                system_index,
                sampled=False,
                content=False,
            )

        loop_messages = messages[loop_start:]
        for loop_index, message in enumerate(loop_messages):
            message_index = loop_start + loop_index
            role = message.get("role", "")
            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    self._render_assistant_tool_calls(
                        message,
                        message_index,
                        builder,
                    )
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

        return RenderedTokens(
            token_ids=builder.token_ids,
            message_indices=builder.message_indices,
            sampled_mask=(
                builder.sampled_mask
                if _get_offset_tokenizer(self._tokenizer) is not None
                else []
            ),
            is_content=_content_mask_or_empty(self._tokenizer, builder.is_content),
            message_roles=[message.get("role") or "" for message in messages],
            message_tool_names=extract_message_tool_names(messages),
        )

    def _render_history_message(
        self,
        message: Message,
        message_index: int,
        builder: _TokenBuilder,
    ) -> None:
        role = message.get("role", "")
        builder.emit_special(
            self._im_start,
            message_index,
            sampled=False,
            content=False,
        )
        builder.emit_segments(
            [(role + "\n", False), (self._content(message), True)],
            message_index,
            sampled=False,
        )
        builder.emit_special(
            self._im_end,
            message_index,
            sampled=False,
            content=False,
        )
        builder.emit_text(
            "\n",
            message_index,
            sampled=False,
            content=False,
        )

    def _render_assistant(
        self,
        message: Message,
        message_index: int,
        builder: _TokenBuilder,
    ) -> None:
        content = self._content(message)
        builder.emit_special(
            self._im_start,
            message_index,
            sampled=False,
            content=False,
        )

        if "reasoning_content" in message:
            builder.emit_text(
                "assistant\n",
                message_index,
                sampled=False,
                content=False,
            )
            builder.emit_special(
                self._think,
                message_index,
                sampled=True,
                content=True,
            )
            reasoning = message.get("reasoning_content")
            if reasoning:
                builder.emit_text(
                    str(reasoning).strip(),
                    message_index,
                    sampled=True,
                    content=True,
                )
            builder.emit_special(
                self._think_end,
                message_index,
                sampled=True,
                content=True,
            )
            if content.strip():
                builder.emit_text(
                    "\n" + content.strip(),
                    message_index,
                    sampled=True,
                    content=True,
                )
        else:
            builder.emit_assistant_segments(
                [("assistant\n", False), (content, True)],
                message_index,
            )

        builder.emit_special(
            self._im_end,
            message_index,
            sampled=True,
            content=True,
        )
        builder.emit_text(
            "\n",
            message_index,
            sampled=False,
            content=False,
        )

    def _render_assistant_tool_calls(
        self,
        message: Message,
        message_index: int,
        builder: _TokenBuilder,
    ) -> None:
        builder.emit_special(
            self._im_start,
            message_index,
            sampled=False,
            content=False,
        )
        content = message.get("content")
        trimmed_content = content.strip() if isinstance(content, str) else ""
        opener_segments = [("assistant\n", False)]
        if trimmed_content:
            opener_segments.append((trimmed_content + "\n\n", True))
        builder.emit_assistant_segments(opener_segments, message_index)

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
                self._tool_call,
                message_index,
                sampled=True,
                content=True,
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
                call_text,
                message_index,
                sampled=True,
                content=True,
            )
            builder.emit_special(
                self._tool_call_end,
                message_index,
                sampled=True,
                content=True,
            )
            if tool_call_index < len(tool_calls) - 1:
                builder.emit_text(
                    "\n",
                    message_index,
                    sampled=True,
                    content=True,
                )

        builder.emit_special(
            self._im_end,
            message_index,
            sampled=True,
            content=True,
        )
        builder.emit_text(
            "\n",
            message_index,
            sampled=False,
            content=False,
        )

    def _render_tool(
        self,
        message: Message,
        message_index: int,
        builder: _TokenBuilder,
        *,
        opens_group: bool,
        closes_group: bool,
    ) -> None:
        if opens_group:
            builder.emit_special(
                self._im_start,
                message_index,
                sampled=False,
                content=False,
            )
            builder.emit_text(
                "user\n",
                message_index,
                sampled=False,
                content=False,
            )

        builder.emit_special(
            self._tool_response,
            message_index,
            sampled=False,
            content=False,
        )
        builder.emit_segments(
            [("\n", False), (self._content(message), True), ("\n", False)],
            message_index,
            sampled=False,
        )
        builder.emit_special(
            self._tool_response_end,
            message_index,
            sampled=False,
            content=False,
        )
        builder.emit_text(
            "\n",
            message_index,
            sampled=False,
            content=False,
        )

        if closes_group:
            builder.emit_special(
                self._im_end,
                message_index,
                sampled=False,
                content=False,
            )
            builder.emit_text(
                "\n",
                message_index,
                sampled=False,
                content=False,
            )

    def _render_generation_prompt(self, builder: _TokenBuilder) -> None:
        builder.emit_special(
            self._im_start,
            -1,
            sampled=False,
            content=False,
        )
        builder.emit_text(
            "assistant\n",
            -1,
            sampled=False,
            content=False,
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
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
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002
    ) -> RenderedTokens | None:
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention,
            new_messages,
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

        builder = _TokenBuilder(self._tokenizer)
        builder.prepend_prior(previous_ids)
        builder.emit_text("\n", -1, sampled=False, content=False)

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
        return RenderedTokens(
            token_ids=builder.token_ids,
            message_indices=builder.message_indices,
            sampled_mask=(
                builder.sampled_mask
                if _get_offset_tokenizer(self._tokenizer) is not None
                else []
            ),
            is_content=_content_mask_or_empty(self._tokenizer, builder.is_content),
            message_roles=[message.get("role") or "" for message in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )


__all__ = ["PrimeQwen3Renderer"]
