"""Pluggable tool-call and reasoning parsers.

Each parser knows how to pull a specific kind of structured output out of a
completion. They're registered by name so callers can compose a
DefaultRenderer with their own tool / reasoning parsing logic without having
to author a whole Renderer.

ToolParser operates on token ids (most tool formats use special tokens as
delimiters, so matching by id is more reliable than regex on decoded text).

ReasoningParser operates on decoded text (reasoning delimiters are typically
``<think>...</think>`` and the text-vs-token distinction here rarely matters
for the final extraction).
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from renderers.base import ParsedResponse, ParsedToolCall, ToolCallParseStatus


# ── Shared helpers ───────────────────────────────────────────────────


def _find(ids: list[int], target: int, start: int = 0) -> int:
    for i in range(start, len(ids)):
        if ids[i] == target:
            return i
    return -1


def _decode(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


def _token_id(tokenizer, token: str) -> int | None:
    """Return the id for *token* if present in the vocab, else None."""
    tid = tokenizer.convert_tokens_to_ids(token)
    # HF returns either the unk id or None for missing tokens. Treat the unk id
    # as missing here so we don't accidentally collide on it.
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or tid == unk:
        return None
    return tid


# ── Protocols ────────────────────────────────────────────────────────


@runtime_checkable
class ToolParser(Protocol):
    """Extracts tool calls from completion token ids.

    ``extract`` returns ``(content_ids, tool_calls)`` where ``content_ids``
    is the remaining content token ids with the tool-call section removed,
    and ``tool_calls`` is a list of :class:`ParsedToolCall` records — one
    per attempted block. Empty list = the model emitted no tool calls;
    callers filter by ``status == OK`` for the clean subset.
    """

    def __init__(self, tokenizer): ...
    def extract(
        self, token_ids: list[int]
    ) -> tuple[list[int], list[ParsedToolCall]]: ...


@runtime_checkable
class ReasoningParser(Protocol):
    """Splits reasoning content from final content.

    ``extract`` returns ``(reasoning_text_or_None, remaining_text)``.
    """

    def __init__(self, tokenizer): ...
    def extract(self, text: str) -> tuple[str | None, str]: ...


# ── Tool parsers ─────────────────────────────────────────────────────


class Qwen3ToolParser:
    """Hermes-style JSON tool calls: ``<tool_call>{"name": .., "arguments": ..}</tool_call>``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tc_id = _token_id(tokenizer, "<tool_call>")
        self._tc_end_id = _token_id(tokenizer, "</tool_call>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        if self._tc_id is None:
            return ids, []
        tc_start = _find(ids, self._tc_id)
        if tc_start == -1:
            return ids, []
        content_ids = ids[:tc_start]
        tool_calls: list[ParsedToolCall] = []
        i = tc_start
        while i < len(ids):
            if ids[i] == self._tc_id:
                end = (
                    _find(ids, self._tc_end_id, i + 1)
                    if self._tc_end_id is not None
                    else -1
                )
                unclosed = end == -1
                if end == -1:
                    end = len(ids)
                tc_text = _decode(self._tokenizer, ids[i + 1 : end]).strip()
                span = (i, end + (0 if unclosed else 1))
                if unclosed:
                    tool_calls.append(
                        ParsedToolCall(
                            raw=tc_text,
                            token_span=span,
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                try:
                    parsed = json.loads(tc_text)
                except json.JSONDecodeError:
                    tool_calls.append(
                        ParsedToolCall(
                            raw=tc_text,
                            token_span=span,
                            status=ToolCallParseStatus.INVALID_JSON,
                        )
                    )
                else:
                    name = parsed.get("name", "") if isinstance(parsed, dict) else ""
                    arguments = (
                        parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
                    )
                    tool_calls.append(
                        ParsedToolCall(
                            raw=tc_text,
                            name=name or None,
                            arguments=arguments,
                            token_span=span,
                            status=(
                                ToolCallParseStatus.MISSING_NAME
                                if not name
                                else ToolCallParseStatus.OK
                            ),
                        )
                    )
                i = end + 1
            else:
                i += 1
        return content_ids, tool_calls


class Qwen35ToolParser:
    """XML-style tool calls: ``<tool_call><function=N><parameter=K>V</parameter>...</function></tool_call>``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tc_id = _token_id(tokenizer, "<tool_call>")
        self._tc_end_id = _token_id(tokenizer, "</tool_call>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        if self._tc_id is None:
            return ids, []
        tc_start = _find(ids, self._tc_id)
        if tc_start == -1:
            return ids, []
        tool_calls: list[ParsedToolCall] = []
        i = tc_start
        while i < len(ids):
            if ids[i] == self._tc_id:
                end = (
                    _find(ids, self._tc_end_id, i + 1)
                    if self._tc_end_id is not None
                    else -1
                )
                if end == -1:
                    raw = _decode(self._tokenizer, ids[i + 1 :])
                    tool_calls.append(
                        ParsedToolCall(
                            raw=raw,
                            token_span=(i, len(ids)),
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                block_text = _decode(self._tokenizer, ids[i + 1 : end])
                span = (i, end + 1)
                name_match = re.search(r"<function=([^>]+)>", block_text)
                if not name_match:
                    tool_calls.append(
                        ParsedToolCall(
                            raw=block_text,
                            token_span=span,
                            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                        )
                    )
                    i = end + 1
                    continue
                name = name_match.group(1)
                arguments: dict = {}
                any_json_fallback = False
                for pm in re.finditer(
                    r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>",
                    block_text,
                    re.DOTALL,
                ):
                    arg_name = pm.group(1)
                    arg_value = pm.group(2).strip()
                    try:
                        arguments[arg_name] = json.loads(arg_value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[arg_name] = arg_value
                        any_json_fallback = True
                tool_calls.append(
                    ParsedToolCall(
                        raw=block_text,
                        name=name,
                        arguments=arguments,
                        token_span=span,
                        status=(
                            ToolCallParseStatus.INVALID_JSON
                            if any_json_fallback
                            else ToolCallParseStatus.OK
                        ),
                    )
                )
                i = end + 1
            else:
                i += 1
        return ids[:tc_start], tool_calls


class GlmToolParser:
    """GLM-5/4.5 token-level tool calls with ``<arg_key>``/``<arg_value>`` pairs."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tc_id = _token_id(tokenizer, "<tool_call>")
        self._tc_end_id = _token_id(tokenizer, "</tool_call>")
        self._ak_id = _token_id(tokenizer, "<arg_key>")
        self._ake_id = _token_id(tokenizer, "</arg_key>")
        self._av_id = _token_id(tokenizer, "<arg_value>")
        self._ave_id = _token_id(tokenizer, "</arg_value>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        if self._tc_id is None:
            return ids, []
        tc_start = _find(ids, self._tc_id)
        if tc_start == -1:
            return ids, []
        tool_calls: list[ParsedToolCall] = []
        i = tc_start
        while i < len(ids):
            if ids[i] == self._tc_id:
                end = (
                    _find(ids, self._tc_end_id, i + 1)
                    if self._tc_end_id is not None
                    else -1
                )
                if end == -1:
                    raw = _decode(self._tokenizer, ids[i + 1 :])
                    tool_calls.append(
                        ParsedToolCall(
                            raw=raw,
                            token_span=(i, len(ids)),
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                block = ids[i + 1 : end]
                block_text = _decode(self._tokenizer, block)
                span = (i, end + 1)
                first_ak = _find(block, self._ak_id) if self._ak_id is not None else -1
                any_json_fallback = False
                structure_broke = False
                if first_ak == -1:
                    name = _decode(self._tokenizer, block).strip()
                    arguments: dict = {}
                else:
                    name = _decode(self._tokenizer, block[:first_ak]).strip()
                    arguments = {}
                    j = first_ak
                    while j < len(block):
                        if block[j] == self._ak_id:
                            ake = (
                                _find(block, self._ake_id, j + 1)
                                if self._ake_id is not None
                                else -1
                            )
                            if ake == -1:
                                structure_broke = True
                                break
                            key = _decode(self._tokenizer, block[j + 1 : ake]).strip()
                            av = (
                                _find(block, self._av_id, ake + 1)
                                if self._av_id is not None
                                else -1
                            )
                            if av == -1:
                                structure_broke = True
                                break
                            ave = (
                                _find(block, self._ave_id, av + 1)
                                if self._ave_id is not None
                                else -1
                            )
                            if ave == -1:
                                structure_broke = True
                                break
                            val_text = _decode(
                                self._tokenizer, block[av + 1 : ave]
                            ).strip()
                            try:
                                arguments[key] = json.loads(val_text)
                            except (json.JSONDecodeError, ValueError):
                                arguments[key] = val_text
                                any_json_fallback = True
                            j = ave + 1
                        else:
                            j += 1
                if not name:
                    status = ToolCallParseStatus.MISSING_NAME
                elif structure_broke:
                    status = ToolCallParseStatus.MALFORMED_STRUCTURE
                elif any_json_fallback:
                    status = ToolCallParseStatus.INVALID_JSON
                else:
                    status = ToolCallParseStatus.OK
                tool_calls.append(
                    ParsedToolCall(
                        raw=block_text,
                        name=name or None,
                        arguments=arguments,
                        token_span=span,
                        status=status,
                    )
                )
                i = end + 1
            else:
                i += 1
        return ids[:tc_start], tool_calls


class DeepSeekV3ToolParser:
    """DeepSeek V3 tool calls, delimited by ``<｜tool▁calls▁begin｜> ... <｜tool▁calls▁end｜>``."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tcs_begin = _token_id(tokenizer, "<｜tool▁calls▁begin｜>")
        self._tcs_end = _token_id(tokenizer, "<｜tool▁calls▁end｜>")
        self._tc_begin = _token_id(tokenizer, "<｜tool▁call▁begin｜>")
        self._tc_end = _token_id(tokenizer, "<｜tool▁call▁end｜>")
        self._sep = _token_id(tokenizer, "<｜tool▁sep｜>")

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        if self._tcs_begin is None:
            return ids, []
        section_start = _find(ids, self._tcs_begin)
        if section_start == -1:
            return ids, []
        content_ids = ids[:section_start]
        section_end = (
            _find(ids, self._tcs_end, section_start + 1)
            if self._tcs_end is not None
            else -1
        )
        if section_end == -1:
            section_end = len(ids)
        inner_offset = section_start + 1
        section_ids = ids[section_start + 1 : section_end]

        tool_calls: list[ParsedToolCall] = []
        i = 0
        while i < len(section_ids):
            if self._tc_begin is None or section_ids[i] != self._tc_begin:
                i += 1
                continue
            end = (
                _find(section_ids, self._tc_end, i + 1)
                if self._tc_end is not None
                else -1
            )
            unclosed = end == -1
            if end == -1:
                end = len(section_ids)
            block_text = _decode(self._tokenizer, section_ids[i + 1 : end])
            span = (inner_offset + i, inner_offset + end + (0 if unclosed else 1))
            # Format: "function<｜tool▁sep｜>{name}\n```json\n{args}\n```"
            name_match = re.search(r"^\s*\w+.*?([A-Za-z0-9_]+)\s*\n", block_text)
            name = name_match.group(1) if name_match else ""
            args: dict | str = {}
            invalid_json = False
            json_match = re.search(r"```json\s*(.*?)\s*```", block_text, re.DOTALL)
            if json_match:
                try:
                    args = json.loads(json_match.group(1))
                except (json.JSONDecodeError, ValueError):
                    args = json_match.group(1)
                    invalid_json = True
            if unclosed:
                status = ToolCallParseStatus.UNCLOSED_BLOCK
            elif not name:
                status = ToolCallParseStatus.MISSING_NAME
            elif invalid_json:
                status = ToolCallParseStatus.INVALID_JSON
            else:
                status = ToolCallParseStatus.OK
            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    name=name or None,
                    arguments=args,
                    token_span=span,
                    status=status,
                )
            )
            i = end + 1
            if unclosed:
                break

        return content_ids, tool_calls


# ── Reasoning parsers ────────────────────────────────────────────────


class ThinkTextReasoningParser:
    """Extracts ``<think>...</think>`` from decoded text.

    Works regardless of whether ``<think>`` is a special token or plain text —
    we decode with ``skip_special_tokens=False`` so both render identically.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer  # unused, for Protocol compat

    def extract(self, text: str) -> tuple[str | None, str]:
        if not text.startswith("<think>"):
            return None, text
        reasoning, close, after = text[len("<think>") :].partition("</think>")
        if not close:
            return reasoning, ""
        # Preserve whitespace on both sides of the `</think>` boundary —
        # chat templates round-trip it verbatim (GLM-4.5 emits reasoning and
        # content back-to-back with no separator), so stripping here causes
        # re-render drift that breaks the trajectory-step extension property.
        return (reasoning or None), after


class Inkling2ToolParser:
    """Parse Inkling 2's token-delimited reasoning, text and invocation blocks."""

    MODEL = "<|message_model|>"
    END_MESSAGE = "<|end_message|>"
    KINDS = {
        "<|content_thinking|>": "thinking",
        "<|content_text|>": "text",
        "<|content_invoke_tool_json|>": "tool_json",
        "<|content_invoke_tool_text|>": "tool_text",
        "<|content_tool_error|>": "tool_error",
    }

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._model_id = _token_id(tokenizer, self.MODEL)
        self._end_id = _token_id(tokenizer, self.END_MESSAGE)
        self._kind_ids: dict[int, str] = {}
        for token, kind in self.KINDS.items():
            tid = _token_id(tokenizer, token)
            if tid is not None:
                self._kind_ids[tid] = kind

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        content, _, _, tool_calls = self._split(ids, parse_tools=True)
        return content, tool_calls

    def parse(self, ids: list[int], *, parse_tools: bool) -> ParsedResponse:
        content, reasoning, complete, tool_calls = self._split(
            ids, parse_tools=parse_tools
        )
        return ParsedResponse(
            content=_decode(self._tokenizer, content),
            reasoning_content=_decode(self._tokenizer, reasoning)
            if reasoning is not None
            else None,
            reasoning_complete=complete,
            tool_calls=tool_calls,
        )

    def _split(
        self, ids: list[int], *, parse_tools: bool
    ) -> tuple[list[int], list[int] | None, bool, list[ParsedToolCall]]:
        if not ids:
            return [], None, True, []
        starts = [0, *(i for i in range(1, len(ids)) if ids[i] == self._model_id)]
        bounds = [*starts, len(ids)]
        content: list[int] = []
        reasoning: list[int] | None = None
        reasoning_complete = True
        tool_calls: list[ParsedToolCall] = []
        for start, end in zip(bounds, bounds[1:]):
            header_start = start + (ids[start] == self._model_id)
            body_end = next(
                (pos for pos in range(header_start, end) if ids[pos] == self._end_id),
                end,
            )
            for marker in range(header_start, body_end):
                kind = self._kind_ids.get(ids[marker])
                if kind is not None:
                    break
            else:
                content.extend(ids[header_start:body_end])
                continue

            body = ids[marker + 1 : body_end]
            closed = body_end < end
            if kind == "thinking":
                if reasoning is None:
                    reasoning = []
                reasoning.extend(body)
                reasoning_complete = closed
            elif kind == "tool_json" and parse_tools:
                header = _decode(self._tokenizer, ids[header_start:marker]).strip()
                payload = _decode(self._tokenizer, body)
                tool_calls.append(
                    self._parse_call(
                        header, payload, (start, min(body_end + 1, end)), closed
                    )
                )
            else:
                if kind == "tool_json":
                    content.extend(ids[header_start:marker])
                content.extend(body)

        return content, reasoning, reasoning_complete, tool_calls

    @staticmethod
    def _parse_call(
        header: str, payload: str, span: tuple[int, int], closed: bool
    ) -> ParsedToolCall:
        call = ParsedToolCall(
            raw=payload, name=header or None, arguments=payload, token_span=span
        )
        if not closed:
            call.status = ToolCallParseStatus.UNCLOSED_BLOCK
            return call

        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            call.status = ToolCallParseStatus.INVALID_JSON
            return call

        name = parsed.get("name", header)
        arguments = parsed.get("args", parsed.get("arguments", {}))
        if not isinstance(name, str) or not isinstance(arguments, dict):
            call.status = ToolCallParseStatus.MALFORMED_STRUCTURE
            return call
        call.name = name.strip() or header or None
        call.arguments = arguments
        if call.name is None:
            call.status = ToolCallParseStatus.MISSING_NAME
        return call


# ── Registries ───────────────────────────────────────────────────────


TOOL_PARSERS: dict[str, type] = {
    "inkling2": Inkling2ToolParser,
    "qwen3": Qwen3ToolParser,
    "qwen3.5": Qwen35ToolParser,
    "glm": GlmToolParser,
    "deepseek-v3": DeepSeekV3ToolParser,
}

REASONING_PARSERS: dict[str, type] = {
    "think": ThinkTextReasoningParser,
}


def get_tool_parser(name: str, tokenizer) -> ToolParser:
    """Look up a tool parser by name and instantiate it with *tokenizer*."""
    if name not in TOOL_PARSERS:
        available = ", ".join(sorted(TOOL_PARSERS))
        raise ValueError(f"Unknown tool_parser {name!r}. Available: {available}.")
    return TOOL_PARSERS[name](tokenizer)


def get_reasoning_parser(name: str, tokenizer) -> ReasoningParser:
    """Look up a reasoning parser by name and instantiate it with *tokenizer*."""
    if name not in REASONING_PARSERS:
        available = ", ".join(sorted(REASONING_PARSERS))
        raise ValueError(f"Unknown reasoning_parser {name!r}. Available: {available}.")
    return REASONING_PARSERS[name](tokenizer)
