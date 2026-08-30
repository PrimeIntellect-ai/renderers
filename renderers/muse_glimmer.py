"""Muse Glimmer ATEM renderer (text-only).

This is a direct Python transcription of the chat template pinned by
``meta-models/Muse-Glimmer-30B``.  It deliberately does not call Jinja at
runtime: Prime-RL needs exact sampled/content masks and a parse/bridge contract
that an opaque template cannot provide.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from renderers.base import (
    Message,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
    ToolSpec,
    _get_offset_tokenizer,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import MuseGlimmerRendererConfig
from renderers.parsing import _build_param_type_index, _coerce_arg_value
from transformers.tokenization_utils import PreTrainedTokenizer

_TOOLS_PREAMBLE = (
    "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
    'You can invoke a function by writing a "<atem:function_calls>" block like the following:\n'
    "<atem:function_calls>\n"
    '<atem:invoke name="$FUNCTION_NAME">\n'
    '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n'
    "...\n"
    "</atem:invoke>\n"
    "</atem:function_calls>\n\n"
    "String and scalar parameters should be specified as is, while lists and objects should use JSON format. "
    "Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n"
    "Here are the functions available in JSONSchema format:\n"
    "// Tool metadata\n"
)

_TOOLS_EXAMPLE = (
    "\n\nHere's an example of how to call a function in the tool set:\n"
    "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n"
    "to=example_tool_name.example_function_name\n\n"
    "<atem:function_calls>\n"
    '<atem:invoke name="example_tool_name.example_function_name">\n'
    '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
    '<atem:parameter name="example_parameter_2">This is the value for the second parameter\n'
    "that can span\n"
    '"multiple" lines\n'
    "</atem:parameter>\n"
    "</atem:invoke>\n"
    "</atem:function_calls>"
)


def _json(value: Any) -> str:
    """Match Jinja's ``tojson`` formatting and HTML-safe escapes."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


def _function(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise TypeError("Tool definitions must be mappings")
    fn = tool.get("function", tool)
    if not isinstance(fn, dict):
        raise TypeError("tool.function must be a mapping")
    return fn


class MuseGlimmerRenderer:
    """Deterministic text renderer for Muse Glimmer's ATEM protocol."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: MuseGlimmerRendererConfig | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.config = config or MuseGlimmerRendererConfig()
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )
        self._eot = self._token_id("<|eot|>")
        self._eom = self._token_id("<|eom|>")

    def _token_id(self, text: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(text)
        if not isinstance(token_id, int) or token_id == self._tokenizer.unk_token_id:
            raise ValueError(f"Muse Glimmer tokenizer lacks {text!r}")
        return token_id

    @staticmethod
    def _content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if not isinstance(value, list):
            raise TypeError("Muse Glimmer message content must be text")
        out: list[str] = []
        for part in value:
            if not isinstance(part, dict):
                raise TypeError("Muse Glimmer content parts must be mappings")
            kind = part.get("type")
            if kind in {"image", "image_url", "video", "video_url"}:
                raise ValueError(
                    "MuseGlimmerRenderer is text-only and rejects image/video content"
                )
            if kind != "text":
                raise ValueError(f"Unsupported Muse Glimmer content part: {kind!r}")
            out.append(str(part.get("text", "")))
        return "".join(out)

    @staticmethod
    def _namespaces(tools: list[ToolSpec] | None) -> list[str]:
        result: list[str] = []
        for tool in tools or []:
            name = str(_function(tool).get("name", ""))
            namespace = name.split(".", 1)[0]
            if namespace not in result:
                result.append(namespace)
        return result

    def _tool_defs(self, tools: list[ToolSpec]) -> str:
        namespaces = self._namespaces(tools)
        text = _TOOLS_PREAMBLE
        descriptions = self.config.tool_namespace_descriptions
        for namespace in namespaces:
            text += (
                '{"name": '
                + _json(namespace)
                + ', "description": '
                + _json(descriptions.get(namespace, ""))
                + "}\n"
            )
        text += "// Function schemas"
        for tool in tools:
            fn = _function(tool)
            text += (
                '\n{"name": '
                + _json(fn.get("name"))
                + ', "description": '
                + _json(fn.get("description"))
                + ', "parameters": '
                + _json(fn.get("parameters"))
                + "}"
            )
        return text + _TOOLS_EXAMPLE

    def _system_meta(self, tools: list[ToolSpec] | None) -> str:
        recipients = ['"self"']
        recipients.extend(f'"{namespace}.*"' for namespace in self._namespaces(tools))
        recipients.append('"user"')
        return "# Valid recipients: " + ", ".join(recipients) + "."

    def _reasoning(self) -> str:
        return f"Reasoning strength: {self.config.reasoning_strength or 'high'}."

    @staticmethod
    def _atem(tool_call: Any) -> str:
        fn = (
            tool_call.get("function", tool_call)
            if isinstance(tool_call, dict)
            else None
        )
        if not isinstance(fn, dict):
            raise TypeError("tool_call.function must be a mapping")
        name = str(fn.get("name", ""))
        arguments = fn.get("arguments")
        if not isinstance(arguments, dict):
            raise TypeError(
                "Muse Glimmer ATEM tool_call.function.arguments must be a mapping"
            )
        text = f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
        for key, value in arguments.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif value is None:
                rendered = "null"
            elif isinstance(value, (dict, list, tuple)):
                rendered = _json(value)
            else:
                rendered = str(value)
            text += f'<atem:parameter name="{key}">{rendered}</atem:parameter>\n'
        return text + "</atem:invoke>\n</atem:function_calls>"

    @staticmethod
    def _tool_name(messages: list[Message], message: Message) -> str:
        explicit = message.get("name")
        if explicit:
            return str(explicit)
        tool_call_id = message.get("tool_call_id")
        fallback = str(tool_call_id or "")
        for candidate in messages:
            for call in candidate.get("tool_calls") or []:
                if tool_call_id is not None and call.get("id") == tool_call_id:
                    fn = call.get("function") or {}
                    return str(fn.get("name", fallback))
        return fallback

    def _chunks(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        add_generation_prompt: bool,
        include_prefix: bool,
        inject_default_system: bool,
    ) -> list[tuple[str, int, bool, bool]]:
        """Return ``(text, message_index, sampled, content)`` chunks."""
        chunks: list[tuple[str, int, bool, bool]] = []

        def add(
            text: str,
            idx: int,
            *,
            sampled: bool = False,
            content: bool = False,
        ) -> None:
            if text:
                chunks.append((text, idx, sampled, content))

        if include_prefix:
            add(str(self._tokenizer.bos_token), -1)
        has_system = any(message.get("role") == "system" for message in messages)
        if inject_default_system and not has_system:
            current_date = self.config.current_date or datetime.now().strftime(
                "%Y-%m-%d"
            )
            text = (
                "<|start|>system<|message|>You are a helpful AI assistant.\n"
                f"Knowledge cutoff: {self.config.knowledge_cutoff or '2026-01-04'}.\n"
                f"Current date: {current_date}.\n\n" + self._reasoning()
            )
            if tools:
                text += "\n\n" + self._tool_defs(tools)
            text += "\n\n" + self._system_meta(tools) + "<|eot|>"
            add(text, -1)

        previous_role: str | None = None
        for idx, message in enumerate(messages):
            role = message.get("role")
            same_role_next = (
                idx + 1 < len(messages) and messages[idx + 1].get("role") == role
            )
            end_token = "<|eom|>" if same_role_next else "<|eot|>"
            if role == "system":
                body = self._content(message.get("content"))
                body = (
                    body.replace("Reasoning effort", "Reasoning strength")
                    .replace("Reasoning Effort", "Reasoning Strength")
                    .replace("reasoning effort", "reasoning strength")
                    .replace("REASONING EFFORT", "REASONING STRENGTH")
                )
                add("<|start|>system<|message|>", idx)
                add(body, idx, content=True)
                if "reasoning strength" not in body.lower():
                    add("\n\n" + self._reasoning(), idx)
                if tools:
                    add("\n\n" + self._tool_defs(tools), idx)
                add("\n\n" + self._system_meta(tools) + "<|eot|>", idx)
            elif role == "user":
                add("<|start|>user<|message|>", idx)
                add(self._content(message.get("content")), idx, content=True)
                add("<|eot|>", idx)
            elif role == "tool":
                name = self._tool_name(messages, message)
                add(
                    f'<|start|>tool {name}<|message|><tool_output name="{name}">\n',
                    idx,
                )
                add(self._content(message.get("content")), idx, content=True)
                add("\n</tool_output><|eot|>", idx)
            elif role == "assistant":
                # A generation prompt contributes the first bare
                # ``<|start|>assistant``. Everything after it is model sampled.
                first_segment = True

                def assistant_segment(text: str) -> None:
                    nonlocal first_segment
                    opener_sampled = previous_role == "assistant" or not first_segment
                    add(
                        "<|start|>assistant",
                        idx,
                        sampled=opener_sampled,
                        content=opener_sampled,
                    )
                    add(text, idx, sampled=True, content=True)
                    first_segment = False

                reasoning = message.get("reasoning_content")
                if reasoning:
                    assistant_segment(
                        " to=self<|message|>" + str(reasoning) + "<|eom|>"
                    )
                calls = message.get("tool_calls") or []
                if calls:
                    for call_idx, call in enumerate(calls):
                        fn = call.get("function") or call
                        close = end_token if call_idx == len(calls) - 1 else "<|eom|>"
                        assistant_segment(
                            " to="
                            + str(fn.get("name", ""))
                            + "<|message|>"
                            + self._atem(call)
                            + close
                        )
                else:
                    recipient = message.get("recipient") or "user"
                    end_turn = message.get("end_turn")
                    if end_turn is None:
                        end_turn = not (recipient and recipient != "user")
                    close = "<|eot|>" if end_turn else "<|eom|>"
                    assistant_segment(
                        " to="
                        + str(recipient)
                        + "<|message|>"
                        + self._content(message.get("content"))
                        + close
                    )
            else:
                raise ValueError(f"Unsupported Muse Glimmer role: {role!r}")
            previous_role = str(role) if role is not None else None

        if add_generation_prompt:
            add("<|start|>assistant", -1)
        return chunks

    def _tokenize(
        self,
        chunks: list[tuple[str, int, bool, bool]],
        message_roles: list[str],
        message_tool_names: list[str | None],
    ) -> RenderedTokens:
        text = "".join(chunk[0] for chunk in chunks)
        if not text:
            return RenderedTokens(
                message_roles=message_roles,
                message_tool_names=message_tool_names,
            )
        tokenizer = _get_offset_tokenizer(self._tokenizer)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        spans: list[tuple[int, int, int, bool, bool]] = []
        cursor = 0
        for chunk_text, idx, sampled, content in chunks:
            spans.append((cursor, cursor + len(chunk_text), idx, sampled, content))
            cursor += len(chunk_text)

        indices: list[int] = []
        sampled_mask: list[bool] = []
        content_mask: list[bool] = []
        span_idx = 0
        for start, end in offsets:
            while span_idx + 1 < len(spans) and start >= spans[span_idx][1]:
                span_idx += 1
            overlapping = [span for span in spans if span[0] < end and start < span[1]]
            chosen = spans[span_idx]
            indices.append(chosen[2])
            # Conservatively include a boundary-merged token in sampled/body
            # masks if any source bytes belong there. For assistant chunks the
            # two bits remain identical by construction.
            sampled_mask.append(any(span[3] for span in overlapping))
            content_mask.append(any(span[4] for span in overlapping))
        return RenderedTokens(
            token_ids=ids,
            message_indices=indices,
            sampled_mask=sampled_mask,
            is_content=content_mask,
            message_roles=message_roles,
            message_tool_names=message_tool_names,
        )

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")
        chunks = self._chunks(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            include_prefix=True,
            inject_default_system=True,
        )
        return self._tokenize(
            chunks,
            [str(message.get("role") or "") for message in messages],
            extract_message_tool_names(messages),
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        ids = list(token_ids)
        if self._eot in ids:
            ids = ids[: ids.index(self._eot)]
        decoded = self._tokenizer.decode(ids, skip_special_tokens=False)
        # Completion tokens follow a prompt ending in bare
        # ``<|start|>assistant``; prepend it only for parsing convenience.
        wire = decoded
        if wire.startswith(" to="):
            wire = "<|start|>assistant" + wire
        pattern = re.compile(
            r"<\|start\|>assistant to=([^<]+)<\|message\|>(.*?)(?:<\|eom\|>|$)",
            re.DOTALL,
        )
        matches = list(pattern.finditer(wire))
        if not matches:
            return ParsedResponse(
                content=decoded, reasoning_content=None, tool_calls=[]
            )

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        parsed_calls: list[ParsedToolCall] = []
        param_index = _build_param_type_index(tools)
        declared_names = set(param_index)
        for match in matches:
            recipient, body = match.group(1), match.group(2)
            if recipient == "self":
                reasoning_parts.append(body)
                continue
            blocks = list(
                re.finditer(
                    r"<atem:function_calls>(.*?)</atem:function_calls>",
                    body,
                    re.DOTALL,
                )
            )
            if not blocks:
                content_parts.append(body)
                continue
            for block in blocks:
                invokes = list(
                    re.finditer(
                        r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>',
                        block.group(1),
                        re.DOTALL,
                    )
                )
                if not invokes:
                    parsed_calls.append(
                        ParsedToolCall(
                            raw=block.group(1),
                            token_span=(0, len(ids)),
                            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                        )
                    )
                    continue
                for invoke in invokes:
                    name, invoke_body = invoke.group(1), invoke.group(2)
                    arguments: dict[str, Any] = {}
                    invalid_json = False
                    for parameter in re.finditer(
                        r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>',
                        invoke_body,
                        re.DOTALL,
                    ):
                        pname, raw = parameter.group(1), parameter.group(2)
                        value, fallback = _coerce_arg_value(
                            raw, param_index.get(name, {}).get(pname)
                        )
                        arguments[pname] = value
                        invalid_json = invalid_json or fallback
                    status = (
                        ToolCallParseStatus.INVALID_JSON
                        if invalid_json
                        else ToolCallParseStatus.OK
                    )
                    if declared_names and name not in declared_names:
                        status = ToolCallParseStatus.UNKNOWN_TOOL
                    parsed_calls.append(
                        ParsedToolCall(
                            raw=invoke.group(0),
                            name=name,
                            arguments=arguments,
                            token_span=(0, len(ids)),
                            status=status,
                        )
                    )
        return ParsedResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=parsed_calls,
        )

    def get_stop_token_ids(self) -> list[int]:
        # <|eom|> separates reasoning/tool channels inside one model turn;
        # stopping on it would truncate the final answer.
        return [self._eot]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
            or should_rerender_for_thinking_retention(
                self.effective_thinking_retention, new_messages
            )
        ):
            return None
        for message in new_messages:
            if message.get("role") == "tool" and not message.get("name"):
                # A raw prior token stream cannot resolve tool_call_id to the
                # issuing function name. Re-render when the caller omitted it.
                return None
        previous = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._eot, self._eom},
            synthesize_close=self._eot,
        )
        if previous is None:
            return None
        chunks = self._chunks(
            new_messages,
            tools=tools,
            add_generation_prompt=True,
            include_prefix=False,
            inject_default_system=False,
        )
        extension = self._tokenize(
            chunks,
            [str(message.get("role") or "") for message in new_messages],
            extract_message_tool_names(new_messages),
        )
        total = len(previous) + len(extension.token_ids)
        return RenderedTokens(
            token_ids=previous + extension.token_ids,
            message_indices=[-1] * len(previous) + extension.message_indices,
            sampled_mask=[False] * total,
            is_content=[False] * len(previous) + extension.is_content,
            message_roles=extension.message_roles,
            message_tool_names=extension.message_tool_names,
        )


__all__ = ["MuseGlimmerRenderer"]
