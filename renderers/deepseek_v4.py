"""DeepSeek V4 Flash 0731 renderer.

The checkpoint does not ship a Jinja chat template.  Its source of truth is
``encoding/encoding_dsv4.py`` in the model repository.  This module mirrors
that encoder while adapting it to the renderer protocol:

* ``enable_thinking=False`` selects the reference encoder's ``chat`` mode;
  ``True`` selects ``thinking`` mode.
* tools are accepted through :meth:`render` and injected on the first
  developer message, otherwise the first system message (the two locations
  supported by the reference encoder).
* OpenAI ``tool`` messages are merged into a DeepSeek user turn as
  ``<tool_result>`` blocks and parallel results are sorted by call order.
* DSML tool calls are parsed back to :class:`ParsedToolCall` records.

Special-token spelling matters: ``｜`` is U+FF5C and ``▁`` is U+2581.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    Tokenizer,
    ToolSpec,
    _content_mask_or_empty,
    _get_offset_tokenizer,
    _infer_offsets_from_decode,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import DeepSeekV4RendererConfig
from renderers.parsing import parse_deepseek_v4


_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_LATEST_REMINDER = "<｜latest_reminder｜>"
_THINK_START = "<think>"
_THINK_END = "</think>"
_DSML = "｜DSML｜"
_QUERY_ROLES = frozenset({"user", "developer"})

_TASK_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}

_REASONING_EFFORT_PROMPTS = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively "
        "decompose the problem to resolve the root cause, rigorously "
        "stress-testing your logic against all potential paths, edge cases, "
        "and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting "
        "every intermediate step, considered alternative, and rejected "
        "hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
        "uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely "
        "nothing to chance: exhaustively decompose the problem into its most "
        "fundamental components, trace every causal chain to its root, and "
        "resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the "
        "solution from multiple angles and are certain that no assumption "
        "remains unchecked and no error remains undiscovered.\n\n"
    ),
}

_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


@dataclass
class _ContentBlock:
    kind: str
    content: str
    message_index: int
    tool_call_id: str = ""


@dataclass
class _LogicalMessage:
    role: str
    message_index: int
    content: str = ""
    blocks: list[_ContentBlock] = field(default_factory=list)
    tool_calls: list[Mapping[str, Any]] = field(default_factory=list)
    reasoning_content: str = ""
    task: str | None = None
    wo_eos: bool = False
    response_format: Any = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)

    parts: list[str] = []
    for part in value:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type == "text":
            parts.append(str(part.get("text", "")))
        elif part_type == "thinking":
            # Structured thinking belongs in ``reasoning_content``.  Ignore it
            # in visible content rather than leaking it into the answer.
            continue
        else:
            parts.append(f"[Unsupported {part_type}]")
    return "\n\n".join(parts)


def _tool_result_content(value: Any) -> str:
    if not isinstance(value, list):
        return "" if value is None else str(value)
    parts: list[str] = []
    for part in value:
        if isinstance(part, Mapping) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, Mapping):
            parts.append(f"[Unsupported {part.get('type')}]")
        else:
            parts.append(f"[Unsupported {type(part).__name__}]")
    return "\n\n".join(parts)


def _reasoning_content(message: Mapping[str, Any]) -> str:
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    if reasoning is not None:
        return str(reasoning)
    content = message.get("content")
    if isinstance(content, list):
        return "".join(
            str(part.get("thinking", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "thinking"
        )
    return ""


def _tool_function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool.get("function")
    return function if isinstance(function, Mapping) else tool


def _tool_call_function(tool_call: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool_call.get("function")
    return function if isinstance(function, Mapping) else tool_call


def _is_query_message(message: Message) -> bool:
    """Match the reference encoder's user/developer query boundary."""
    return message.get("role") in _QUERY_ROLES


def _prepare_messages(messages: list[Message]) -> list[_LogicalMessage]:
    """Apply DeepSeek's user/tool merge and parallel-result ordering."""
    merged: list[_LogicalMessage] = []

    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        if role == "tool":
            block = _ContentBlock(
                kind="tool_result",
                content=_tool_result_content(message.get("content")),
                message_index=index,
                tool_call_id=str(message.get("tool_call_id") or ""),
            )
            if merged and merged[-1].role == "user":
                merged[-1].blocks.append(block)
            else:
                merged.append(
                    _LogicalMessage(
                        role="user",
                        message_index=index,
                        blocks=[block],
                    )
                )
            continue

        if role == "user":
            block = _ContentBlock(
                kind="text",
                content=_text_content(message.get("content")),
                message_index=index,
            )
            if merged and merged[-1].role == "user" and merged[-1].task is None:
                merged[-1].blocks.append(block)
            else:
                merged.append(
                    _LogicalMessage(
                        role="user",
                        message_index=index,
                        content=block.content,
                        blocks=[block],
                        task=message.get("task"),
                        wo_eos=bool(message.get("wo_eos", False)),
                    )
                )
            continue

        logical = _LogicalMessage(
            role=role,
            message_index=index,
            content=_text_content(message.get("content")),
            task=message.get("task"),
            wo_eos=bool(message.get("wo_eos", False)),
            response_format=message.get("response_format"),
        )
        if role == "assistant":
            logical.reasoning_content = _reasoning_content(message)
            logical.tool_calls = list(message.get("tool_calls") or [])
        merged.append(logical)

    call_order: dict[str, int] = {}
    for message in merged:
        if message.role == "assistant" and message.tool_calls:
            call_order = {}
            for index, tool_call in enumerate(message.tool_calls):
                function = _tool_call_function(tool_call)
                call_id = tool_call.get("id") or function.get("id")
                if call_id:
                    call_order[str(call_id)] = index
        elif message.role == "user":
            result_blocks = [b for b in message.blocks if b.kind == "tool_result"]
            if len(result_blocks) < 2 or not call_order:
                continue
            result_blocks.sort(key=lambda b: call_order.get(b.tool_call_id, 0))
            ordered = iter(result_blocks)
            message.blocks = [
                next(ordered) if block.kind == "tool_result" else block
                for block in message.blocks
            ]

    return merged


class DeepSeekV4Renderer:
    """Renderer for ``deepseek-ai/DeepSeek-V4-Flash-0731``."""

    _implied_thinking_retention = "tool_cycle"

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: DeepSeekV4RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or DeepSeekV4RendererConfig()
        implied_retention = (
            "tool_cycle"
            if self.config.enable_thinking and self.config.drop_thinking
            else "all"
        )
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config,
            implied_retention,
        )

        self._bos = self._special_id(_BOS)
        self._eos = self._special_id(_EOS)
        self._user = self._special_id(_USER)
        self._assistant = self._special_id(_ASSISTANT)
        self._latest_reminder = self._special_id(_LATEST_REMINDER)
        self._think_start = self._special_id(_THINK_START)
        self._think_end = self._special_id(_THINK_END)
        self._dsml = self._special_id(_DSML)
        self._task_ids = {
            name: self._special_id(token) for name, token in _TASK_TOKENS.items()
        }

    def _special_id(self, token: str) -> int:
        ids = self._tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected one token for {token!r}, got {ids}")
        return ids[0]

    @staticmethod
    def _render_tools(tools: list[ToolSpec]) -> str:
        schemas = [_json(dict(_tool_function(tool))) for tool in tools]
        return _TOOLS_TEMPLATE.format(
            dsml=_DSML,
            think_start=_THINK_START,
            think_end=_THINK_END,
            tool_schemas="\n".join(schemas),
        )

    @staticmethod
    def _render_tool_call(tool_call: Mapping[str, Any]) -> str:
        function = _tool_call_function(tool_call)
        name = function.get("name")
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"arguments": raw_arguments}
        elif isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            arguments = {"arguments": raw_arguments}

        params: list[str] = []
        for key, value in arguments.items():
            is_string = isinstance(value, str)
            rendered_value = value if is_string else _json(value)
            params.append(
                f'<{_DSML}parameter name="{key}" '
                f'string="{str(is_string).lower()}">{rendered_value}'
                f"</{_DSML}parameter>"
            )
        arguments_text = "\n".join(params)
        return f'<{_DSML}invoke name="{name}">\n{arguments_text}\n</{_DSML}invoke>'

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        return self._render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            add_bos=True,
            add_effort_prompt=True,
        )

    def _render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        add_generation_prompt: bool,
        add_bos: bool,
        add_effort_prompt: bool,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        logical_messages = _prepare_messages(messages)
        if not logical_messages:
            raise ValueError("No renderable messages provided.")

        effective_drop_thinking = self.config.drop_thinking and not tools
        if self.config.enable_thinking and effective_drop_thinking:
            last_query = max(
                (
                    index
                    for index, message in enumerate(logical_messages)
                    if message.role in _QUERY_ROLES
                ),
                default=-1,
            )
            # The reference encoder removes internal developer/search-agent
            # messages before the latest query when historical thinking is
            # dropped.  Public tool flows retain them because tools force
            # ``effective_drop_thinking=False``.
            logical_messages = [
                message
                for index, message in enumerate(logical_messages)
                if message.role != "developer" or index >= last_query
            ]

        token_ids: list[int] = []
        message_indices: list[int] = []
        sampled_mask: list[bool] = []
        is_content: list[bool] = []
        pending_text: list[tuple[str, int, bool, bool]] = []

        def emit_ids(
            ids: list[int],
            message_index: int,
            *,
            sampled: bool = False,
            content: bool = False,
        ) -> None:
            token_ids.extend(ids)
            message_indices.extend([message_index] * len(ids))
            sampled_mask.extend([sampled] * len(ids))
            is_content.extend([content] * len(ids))

        def flush_text() -> None:
            """Tokenize contiguous text once, preserving source metadata.

            The official encoder builds one prompt string. Encoding renderer
            fragments independently can therefore change BPE merges at their
            boundaries even when the decoded text is identical. Offset maps
            let us recover message/sample/content attribution after the
            required single encoding pass.
            """
            if not pending_text:
                return

            full_text = "".join(text for text, *_ in pending_text)
            if not full_text:
                pending_text.clear()
                return

            spans: list[tuple[int, int, int, bool, bool]] = []
            position = 0
            for text, message_index, sampled, content in pending_text:
                end = position + len(text)
                if end > position:
                    spans.append((position, end, message_index, sampled, content))
                position = end

            offset_tokenizer = _get_offset_tokenizer(self._tokenizer)
            if offset_tokenizer is None:
                text_ids = self._tokenizer.encode(
                    full_text,
                    add_special_tokens=False,
                )
                offsets = _infer_offsets_from_decode(
                    self._tokenizer,
                    text_ids,
                    full_text,
                )
                has_content_attribution = False
                if offsets is None:
                    # Token IDs remain exact even when a lossy decoder makes
                    # source boundaries unrecoverable. Text runs separated by
                    # special tokens have one sampled state, so retain that
                    # signal and associate the opaque run with a contributing
                    # caller message.
                    fallback_message_index = next(
                        (
                            span_message_index
                            for _, _, span_message_index, _, _ in spans
                            if span_message_index >= 0
                        ),
                        spans[-1][2] if spans else -1,
                    )
                    fallback_sampled = spans[-1][3] if spans else False
                    emit_ids(
                        text_ids,
                        fallback_message_index,
                        sampled=fallback_sampled,
                    )
                    pending_text.clear()
                    return
            else:
                encoding = offset_tokenizer(
                    full_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                text_ids = list(encoding["input_ids"])
                offsets = list(encoding["offset_mapping"])
                has_content_attribution = True

            fallback = spans[-1][2:] if spans else (-1, False, False)
            for token_id, (start, end) in zip(text_ids, offsets):
                metadata: tuple[int, bool, bool] = fallback
                for (
                    span_start,
                    span_end,
                    span_message_index,
                    span_sampled,
                    span_content,
                ) in spans:
                    if span_start <= start < span_end:
                        metadata = (
                            span_message_index,
                            span_sampled,
                            span_content,
                        )
                        break

                message_index, sampled, content = metadata
                if has_content_attribution and end > start:
                    # Preserve every body byte when a BPE token straddles a
                    # scaffold/content boundary. This intentionally permits a
                    # few adjacent scaffold bytes to share the content bit.
                    content = any(
                        span_content
                        for span_start, span_end, _, _, span_content in spans
                        if span_start < end and start < span_end
                    )
                elif not has_content_attribution:
                    content = False
                emit_ids(
                    [token_id],
                    message_index,
                    sampled=sampled,
                    content=content,
                )

            pending_text.clear()

        def emit_special(
            token_id: int,
            message_index: int,
            *,
            sampled: bool = False,
            content: bool = False,
        ) -> None:
            flush_text()
            emit_ids(
                [token_id],
                message_index,
                sampled=sampled,
                content=content,
            )

        def emit_text(
            text: str,
            message_index: int,
            *,
            sampled: bool = False,
            content: bool = False,
        ) -> None:
            if text:
                pending_text.append((text, message_index, sampled, content))

        if add_bos:
            emit_special(self._bos, -1)
        if add_effort_prompt and self.config.enable_thinking:
            emit_text(
                _REASONING_EFFORT_PROMPTS[self.config.reasoning_effort],
                -1,
            )

        last_query_index = -1
        for index, message in enumerate(logical_messages):
            if message.role in _QUERY_ROLES:
                last_query_index = index

        tool_target: int | None = None
        if tools:
            tool_target = next(
                (
                    index
                    for index, message in enumerate(logical_messages)
                    if message.role == "developer"
                ),
                None,
            )
            if tool_target is None:
                tool_target = next(
                    (
                        index
                        for index, message in enumerate(logical_messages)
                        if message.role == "system"
                    ),
                    None,
                )
            if tool_target is None:
                # Equivalent to an empty synthetic system message carrying
                # tools in the reference encoder.
                emit_text("\n\n" + self._render_tools(tools), -1)

        for index, message in enumerate(logical_messages):
            role = message.role
            msg_idx = message.message_index

            if role == "system":
                emit_text(message.content, msg_idx, content=True)

            elif role == "developer":
                if not message.content:
                    raise ValueError("Developer messages require content.")
                emit_special(self._user, msg_idx)
                emit_text(message.content, msg_idx, content=True)

            elif role == "user":
                emit_special(self._user, msg_idx)
                for block_index, block in enumerate(message.blocks):
                    if block_index:
                        emit_text("\n\n", block.message_index)
                    if block.kind == "tool_result":
                        emit_text("<tool_result>", block.message_index)
                        emit_text(
                            block.content,
                            block.message_index,
                            content=True,
                        )
                        emit_text("</tool_result>", block.message_index)
                    else:
                        emit_text(
                            block.content,
                            block.message_index,
                            content=True,
                        )

            elif role == "latest_reminder":
                emit_special(self._latest_reminder, msg_idx)
                emit_text(message.content, msg_idx, content=True)

            elif role == "assistant":
                previous_has_task = (
                    index > 0 and logical_messages[index - 1].task is not None
                )
                keep_reasoning = (
                    self.config.enable_thinking
                    and not previous_has_task
                    and (not effective_drop_thinking or index > last_query_index)
                )
                if keep_reasoning:
                    emit_text(
                        message.reasoning_content,
                        msg_idx,
                        sampled=True,
                        content=True,
                    )
                    emit_special(
                        self._think_end,
                        msg_idx,
                        sampled=True,
                        content=True,
                    )

                emit_text(
                    message.content,
                    msg_idx,
                    sampled=True,
                    content=True,
                )
                if message.tool_calls:
                    rendered_calls = "\n".join(
                        self._render_tool_call(call) for call in message.tool_calls
                    )
                    emit_text(
                        f"\n\n<{_DSML}tool_calls>\n{rendered_calls}\n"
                        f"</{_DSML}tool_calls>",
                        msg_idx,
                        sampled=True,
                        content=True,
                    )
                if not message.wo_eos:
                    emit_special(
                        self._eos,
                        msg_idx,
                        sampled=True,
                        content=True,
                    )

            else:
                raise ValueError(f"Unsupported DeepSeek V4 role: {role!r}")

            if tools and index == tool_target:
                emit_text("\n\n" + self._render_tools(tools), msg_idx)
            if message.response_format is not None and role in {
                "system",
                "developer",
            }:
                emit_text(
                    "\n\n## Response Format:\n\n"
                    "You MUST strictly adhere to the following schema to reply:\n"
                    + _json(message.response_format),
                    msg_idx,
                )

            next_role = (
                logical_messages[index + 1].role
                if index + 1 < len(logical_messages)
                else None
            )
            transition_needed = next_role in {"assistant", "latest_reminder"}
            if next_role is None:
                transition_needed = message.task is not None or add_generation_prompt
            if not transition_needed:
                continue

            task = message.task
            transition_msg_idx = -1
            if next_role is not None:
                for following in logical_messages[index + 1 :]:
                    if following.role == "assistant":
                        transition_msg_idx = following.message_index
                        break

            if task is not None:
                if task not in self._task_ids:
                    raise ValueError(
                        f"Invalid DeepSeek V4 task {task!r}; expected one of "
                        f"{sorted(self._task_ids)}"
                    )
                if task != "action":
                    emit_special(self._task_ids[task], transition_msg_idx)
                    continue
                emit_special(self._assistant, transition_msg_idx)
                emit_special(
                    self._think_start
                    if self.config.enable_thinking
                    else self._think_end,
                    transition_msg_idx,
                )
                emit_special(self._task_ids[task], transition_msg_idx)
                continue

            if role not in _QUERY_ROLES:
                continue
            emit_special(self._assistant, transition_msg_idx)
            open_thinking = self.config.enable_thinking and (
                not effective_drop_thinking or index >= last_query_index
            )
            emit_special(
                self._think_start if open_thinking else self._think_end,
                transition_msg_idx,
            )

        flush_text()
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            sampled_mask=sampled_mask,
            is_content=_content_mask_or_empty(self._tokenizer, is_content),
            message_roles=[message.get("role") or "" for message in messages],
            message_tool_names=extract_message_tool_names(messages),
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
        tools: list[ToolSpec] | None = None,  # noqa: ARG002
    ) -> ParsedResponse:
        return parse_deepseek_v4(
            self._tokenizer,
            token_ids,
            stop_ids={self._eos},
            thinking_enabled=self.config.enable_thinking,
            think_end_id=self._think_end,
            dsml_id=self._dsml,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._eos]

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
            is_user_query=_is_query_message,
        ):
            return None
        # Full rendering sorts parallel tool results by IDs from the issuing
        # assistant.  The bridge only receives the new slice, so it cannot
        # prove parity when more than one result is present.
        if sum(message.get("role") == "tool" for message in new_messages) > 1:
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._eos},
            synthesize_close=self._eos,
        )
        if previous_ids is None:
            return None

        try:
            extension = self._render(
                new_messages,
                tools=None,
                add_generation_prompt=True,
                add_bos=False,
                add_effort_prompt=False,
            )
        except ValueError:
            return None

        prior_length = len(previous_ids)
        return RenderedTokens(
            token_ids=previous_ids + extension.token_ids,
            message_indices=[-1] * prior_length + extension.message_indices,
            sampled_mask=[False] * (prior_length + len(extension.token_ids)),
            is_content=_content_mask_or_empty(
                self._tokenizer,
                [False] * prior_length + extension.is_content,
            ),
            message_roles=extension.message_roles,
            message_tool_names=extension.message_tool_names,
        )


__all__ = ["DeepSeekV4Renderer"]
