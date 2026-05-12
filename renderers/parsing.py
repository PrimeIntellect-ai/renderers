"""Token-level parsing — operates on token IDs directly.

Finds special token boundaries by scanning token IDs, then decodes only
the text segments between them. No regex on decoded text, no false positives
from content that happens to look like special tokens.

Every parser emits ``list[ParsedToolCall]`` covering every attempt —
successful and malformed alike — with a ``status`` enum classifying the
outcome and a ``token_span`` recording where in the (stop-stripped)
token stream the attempt sat. Callers filter on ``status == OK`` for the
clean subset; verifier and RL-loss code uses the rest. This diverges from
vLLM's ``ExtractedToolCallInformation`` (single ``tools_called`` bool, no
per-call status) and SGLang's ``StreamingParseResult`` (silent drop on
failure) — see ``ToolCallParseStatus`` docstring for the rationale.
"""

from __future__ import annotations

import json

from renderers.base import ParsedResponse, ParsedToolCall, ToolCallParseStatus


def _find(ids: list[int], target: int, start: int = 0) -> int:
    """Find index of target in ids, or -1."""
    for i in range(start, len(ids)):
        if ids[i] == target:
            return i
    return -1


def _find_any(ids: list[int], targets: set[int], start: int = 0) -> int:
    """Find first index in ids whose value is in targets, or -1."""
    for i in range(start, len(ids)):
        if ids[i] in targets:
            return i
    return -1


def _find_all(ids: list[int], target: int) -> list[int]:
    """Find all indices of target in ids."""
    return [i for i, t in enumerate(ids) if t == target]


def _strip_stop_tokens(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Truncate at first stop token (model shouldn't generate past it)."""
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[:i]
    return ids


def _decode(tokenizer, ids: list[int]) -> str:
    """Decode token IDs to text, skipping special tokens."""
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


# ── Qwen3: <tool_call> JSON </tool_call> ────────────────────────────


def parse_qwen3(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_call_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Qwen3 completion tokens. Hermes-style JSON tool calls."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_ids = ids[:tc_start]
        i = tc_start
        while i < len(ids):
            if ids[i] == tool_call_id:
                end = _find(ids, tool_call_end_id, i + 1)
                if end == -1:
                    # No closing delim — block runs to end of stripped ids.
                    raw = _decode(tokenizer, ids[i + 1 :]).strip()
                    tool_calls.append(
                        ParsedToolCall(
                            raw=raw,
                            token_span=(i, len(ids)),
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                tc_text = _decode(tokenizer, ids[i + 1 : end]).strip()
                span = (i, end + 1)
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
                    if not name:
                        tool_calls.append(
                            ParsedToolCall(
                                raw=tc_text,
                                name=None,
                                arguments=arguments,
                                token_span=span,
                                status=ToolCallParseStatus.MISSING_NAME,
                            )
                        )
                    else:
                        tool_calls.append(
                            ParsedToolCall(
                                raw=tc_text,
                                name=name,
                                arguments=arguments,
                                token_span=span,
                                status=ToolCallParseStatus.OK,
                            )
                        )
                i = end + 1
            else:
                i += 1
    else:
        content_ids = ids

    text = _decode(tokenizer, content_ids)
    # Extract reasoning from text (Qwen3 doesn't have <think> as special token)
    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").strip("\n").strip()
        text = after.strip("\n")

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


# ── Qwen3.5: <tool_call> <function=name> <parameter=name> v </parameter> </function> </tool_call>


def parse_qwen35(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Qwen3.5 completion tokens. XML-style tool calls, token-level thinking."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Thinking: find </think> by token ID
    reasoning = None
    parse_offset = 0  # shift to map local indices back to stop-stripped ids
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif think_id in set(ids):
        # <think> present but no </think> — truncated reasoning
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        tool_calls = _parse_xml_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            section_offset=parse_offset + tc_start,
        )
    else:
        content_text = _decode(tokenizer, ids).strip()

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_xml_tool_calls(
    tokenizer,
    ids: list[int],
    tc_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
) -> list[ParsedToolCall]:
    """Parse Qwen3.5-style XML tool calls from token IDs."""
    import re

    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_id:
            end = _find(ids, tc_end_id, i + 1)
            if end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            block_text = _decode(tokenizer, ids[i + 1 : end])
            span = (section_offset + i, section_offset + end + 1)
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
                r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", block_text, re.DOTALL
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
    return tool_calls


# ── GLM-5/4.7/4.5: <tool_call> name <arg_key>k</arg_key> <arg_value>v</arg_value> </tool_call>


def parse_glm(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    arg_key_id: int,
    arg_key_end_id: int,
    arg_value_id: int,
    arg_value_end_id: int,
) -> ParsedResponse:
    """Parse GLM completion tokens. Token-level thinking + arg_key/arg_value tool calls."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif think_id in set(ids):
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        tool_calls = _parse_glm_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            arg_key_id,
            arg_key_end_id,
            arg_value_id,
            arg_value_end_id,
            section_offset=parse_offset + tc_start,
        )
    else:
        content_text = _decode(tokenizer, ids).strip()

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_glm_tool_calls(
    tokenizer,
    ids,
    tc_id,
    tc_end_id,
    ak_id,
    ake_id,
    av_id,
    ave_id,
    *,
    section_offset: int,
) -> list[ParsedToolCall]:
    """Parse GLM-style tool calls: name + arg_key/arg_value pairs, all by token ID."""
    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_id:
            end = _find(ids, tc_end_id, i + 1)
            if end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            block = ids[i + 1 : end]
            block_text = _decode(tokenizer, block)
            span = (section_offset + i, section_offset + end + 1)
            first_ak = _find(block, ak_id)
            any_json_fallback = False
            structure_broke = False
            if first_ak == -1:
                name = _decode(tokenizer, block).strip()
                arguments: dict = {}
            else:
                name = _decode(tokenizer, block[:first_ak]).strip()
                arguments = {}
                j = first_ak
                while j < len(block):
                    if block[j] == ak_id:
                        ake = _find(block, ake_id, j + 1)
                        if ake == -1:
                            structure_broke = True
                            break
                        key = _decode(tokenizer, block[j + 1 : ake]).strip()
                        av = _find(block, av_id, ake + 1)
                        if av == -1:
                            structure_broke = True
                            break
                        ave = _find(block, ave_id, av + 1)
                        if ave == -1:
                            structure_broke = True
                            break
                        val_text = _decode(tokenizer, block[av + 1 : ave]).strip()
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
    return tool_calls


# ── DeepSeek V3: <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> + text <think> tags ──


def parse_deepseek_v3(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_calls_begin_id: int,
    tool_calls_end_id: int,
    tool_call_begin_id: int,
    tool_call_end_id: int,
    tool_sep_id: int,
) -> ParsedResponse:
    """Parse DeepSeek V3 completion tokens.

    Thinking is embedded as plain text <think>...</think> tags (not special tokens).
    Tool calls are delimited by special tokens:
        <｜tool▁calls▁begin｜>
          <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n{args}\\n```<｜tool▁call▁end｜>
        <｜tool▁calls▁end｜>
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    tc_section_start = _find(ids, tool_calls_begin_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_section_start != -1:
        content_ids = ids[:tc_section_start]
        tool_calls = _parse_deepseek_tool_calls(
            tokenizer,
            ids[tc_section_start:],
            tool_calls_begin_id,
            tool_calls_end_id,
            tool_call_begin_id,
            tool_call_end_id,
            tool_sep_id,
            section_offset=tc_section_start,
        )
    else:
        content_ids = ids

    text = _decode(tokenizer, content_ids)

    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").lstrip("\n").rstrip("\n").strip()
        text = after.lstrip("\n")

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_deepseek_tool_calls(
    tokenizer,
    ids: list[int],
    tc_begin_id: int,
    tc_end_id: int,
    call_begin_id: int,
    call_end_id: int,
    sep_id: int,
    *,
    section_offset: int,
) -> list[ParsedToolCall]:
    """Parse DeepSeek V3-style tool calls from token IDs."""
    import re

    tool_calls: list[ParsedToolCall] = []

    section_start = _find(ids, tc_begin_id)
    if section_start == -1:
        return tool_calls
    section_end = _find(ids, tc_end_id, section_start + 1)
    section_end_clipped = section_end == -1
    if section_end == -1:
        section_end = len(ids)

    inner_offset = section_offset + section_start + 1
    section_ids = ids[section_start + 1 : section_end]

    i = 0
    while i < len(section_ids):
        if section_ids[i] == call_begin_id:
            end = _find(section_ids, call_end_id, i + 1)
            unclosed = end == -1
            if unclosed:
                end = len(section_ids)
            call_ids = section_ids[i + 1 : end]
            block_text = _decode(tokenizer, call_ids)
            # Span for this call covers its <tool_call_begin>..<tool_call_end> range
            # within the (stop-stripped) parent token stream.
            span = (
                inner_offset + i,
                inner_offset + end + (0 if unclosed else 1),
            )

            sep_pos = _find(call_ids, sep_id)
            if sep_pos == -1:
                tool_calls.append(
                    ParsedToolCall(
                        raw=block_text,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                i = end + 1
                continue

            after_sep_ids = call_ids[sep_pos + 1 :]
            after_sep_text = _decode(tokenizer, after_sep_ids).strip()

            name = ""
            args_str = ""
            newline_pos = after_sep_text.find("\n")
            if newline_pos != -1:
                name = after_sep_text[:newline_pos].strip()
                rest = after_sep_text[newline_pos + 1 :].strip()
                fence_match = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```$", rest)
                args_str = fence_match.group(1).strip() if fence_match else rest
            else:
                name = after_sep_text

            arguments: dict | str
            invalid_json = False
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = args_str
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
                    arguments=arguments,
                    token_span=span,
                    status=status,
                )
            )
            i = end + 1
            if unclosed:
                break
        else:
            i += 1

    # If the outer <tool_calls_begin> had no matching <tool_calls_end>, any
    # call inside that didn't itself flag UNCLOSED_BLOCK is still nested in
    # a truncated section — but we already mark individual unclosed calls,
    # so we don't double-flag here. The section_end_clipped variable is
    # carried for the (rare) caller that wants section-level UX.
    _ = section_end_clipped
    return tool_calls


# ── MiniMax: <minimax:tool_call> ... </minimax:tool_call> ────────────


def parse_minimax(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse MiniMax M2 completion tokens."""
    import re

    ids = _strip_stop_tokens(token_ids, stop_ids)

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif think_id in set(ids):
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        i = tc_start
        while i < len(ids):
            if ids[i] == tool_call_id:
                end = _find(ids, tool_call_end_id, i + 1)
                if end == -1:
                    raw = _decode(tokenizer, ids[i + 1 :])
                    tool_calls.append(
                        ParsedToolCall(
                            raw=raw,
                            token_span=(
                                parse_offset + i,
                                parse_offset + len(ids),
                            ),
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                block_text = _decode(tokenizer, ids[i + 1 : end])
                span = (parse_offset + i, parse_offset + end + 1)

                invokes = list(
                    re.finditer(
                        r'<invoke name="([^"]+)">(.*?)</invoke>',
                        block_text,
                        re.DOTALL,
                    )
                )
                if not invokes:
                    # Block exists but contains no <invoke> — model emitted
                    # the wrapper without a usable body.
                    tool_calls.append(
                        ParsedToolCall(
                            raw=block_text,
                            token_span=span,
                            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                        )
                    )
                else:
                    for invoke_match in invokes:
                        name = invoke_match.group(1)
                        body = invoke_match.group(2)
                        arguments: dict = {}
                        any_json_fallback = False
                        for pm in re.finditer(
                            r'<parameter name="([^"]+)">(.*?)</parameter>',
                            body,
                            re.DOTALL,
                        ):
                            pname = pm.group(1)
                            pval = pm.group(2).strip()
                            try:
                                arguments[pname] = json.loads(pval)
                            except (json.JSONDecodeError, ValueError):
                                arguments[pname] = pval
                                any_json_fallback = True
                        tool_calls.append(
                            ParsedToolCall(
                                raw=block_text,
                                name=name,
                                arguments=arguments,
                                # All invokes in a block share the wrapper span.
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
    else:
        content_text = _decode(tokenizer, ids).strip()

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


# ── Kimi K2: <|tool_calls_section_begin|> ... <|tool_calls_section_end|> ────


def parse_kimi_k2_section(
    tokenizer,
    ids: list[int],
    *,
    tool_calls_section_begin_ids: set[int],
    tool_calls_section_end_ids: set[int],
    tool_call_begin_id: int,
    tool_call_argument_begin_id: int,
    tool_call_end_id: int,
) -> tuple[list[int], list[ParsedToolCall]]:
    """Split ``ids`` into ``(content_before_section, tool_calls)`` by finding
    the Kimi-style tool-call section delimiters.

    Accepts *sets* of begin/end token IDs so callers can express models with
    multiple delimiter variants (K2.5 has both plural ``<|tool_calls_section_*|>``
    and singular ``<|tool_call_section_*|>`` forms, though only the plural form
    is in the special-token vocab in practice). Returns the content ids ahead
    of the section and a list of ``ParsedToolCall`` covering every attempted
    block inside it; an unclosed section is still walked to whatever the model
    emitted before EOS. Returns ``(ids, [])`` when no section is present.
    """
    section_start = _find_any(ids, tool_calls_section_begin_ids)
    if section_start == -1:
        return list(ids), []
    content_ids = ids[:section_start]
    section_end = _find_any(ids, tool_calls_section_end_ids, section_start + 1)
    if section_end == -1:
        section_end = len(ids)
    section_ids = ids[section_start + 1 : section_end]
    tool_calls = _parse_kimi_k2_tool_calls(
        tokenizer,
        section_ids,
        tool_call_begin_id,
        tool_call_argument_begin_id,
        tool_call_end_id,
        section_offset=section_start + 1,
    )
    return content_ids, tool_calls


def parse_kimi_k2(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_calls_section_begin_id: int,
    tool_calls_section_end_id: int,
    tool_call_begin_id: int,
    tool_call_argument_begin_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Kimi K2 completion tokens.

    Thinking is encoded as text tags <think>...</think>.
    Tool calls use section/call-level special tokens.
    Tool call IDs are in format ``functions.name:index``.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    content_ids, tool_calls = parse_kimi_k2_section(
        tokenizer,
        ids,
        tool_calls_section_begin_ids={tool_calls_section_begin_id},
        tool_calls_section_end_ids={tool_calls_section_end_id},
        tool_call_begin_id=tool_call_begin_id,
        tool_call_argument_begin_id=tool_call_argument_begin_id,
        tool_call_end_id=tool_call_end_id,
    )

    text = _decode(tokenizer, content_ids)
    reasoning: str | None = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        raw_think = before.replace("<think>", "", 1)
        reasoning = raw_think.strip("\n").strip() or None
        text = after.strip("\n")
    elif "<think>" in text:
        # Truncated thinking (no closing tag)
        raw_think = text.split("<think>", 1)[1]
        reasoning = raw_think.strip("\n").strip() or None
        return ParsedResponse(
            content="",
            reasoning_content=reasoning,
            tool_calls=[],
        )

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )


def _parse_kimi_k2_tool_calls(
    tokenizer,
    ids: list[int],
    tc_begin_id: int,
    tc_arg_begin_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
) -> list[ParsedToolCall]:
    """Parse individual Kimi K2 tool calls from the section token IDs.

    Format per call:
        <|tool_call_begin|>{id}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>

    The ``id`` is in format ``functions.name:index``; the function name is
    extracted by stripping the ``functions.`` prefix and ``:index`` suffix.
    """
    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_begin_id:
            arg_begin = _find(ids, tc_arg_begin_id, i + 1)
            if arg_begin == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                break
            tc_end = _find(ids, tc_end_id, arg_begin + 1)
            unclosed = tc_end == -1
            if tc_end == -1:
                tc_end = len(ids)

            raw_id = _decode(tokenizer, ids[i + 1 : arg_begin]).strip()
            args_str = _decode(tokenizer, ids[arg_begin + 1 : tc_end]).strip()
            block_text = _decode(tokenizer, ids[i + 1 : tc_end])
            span = (
                section_offset + i,
                section_offset + tc_end + (0 if unclosed else 1),
            )

            name_part = raw_id.split(":", 1)[0]
            if "." in name_part:
                _, func_name = name_part.split(".", 1)
            else:
                func_name = name_part

            arguments: dict | str
            invalid_json = False
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = args_str
                invalid_json = True

            if unclosed:
                status = ToolCallParseStatus.UNCLOSED_BLOCK
            elif not func_name:
                status = ToolCallParseStatus.MISSING_NAME
            elif invalid_json:
                status = ToolCallParseStatus.INVALID_JSON
            else:
                status = ToolCallParseStatus.OK

            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    name=func_name or None,
                    arguments=arguments,
                    token_span=span,
                    status=status,
                    id=raw_id or None,
                )
            )
            i = tc_end + 1
            if unclosed:
                break
        else:
            i += 1
    return tool_calls


# ── GptOss (Harmony): <|start|>role<|channel|>ch<|message|>content<|end|/return|/call|>


def parse_gpt_oss(
    tokenizer,
    token_ids: list[int],
    *,
    return_id: int,
    call_id: int,
    start_id: int,
    end_id: int,
    channel_id: int,
    message_id: int,
    constrain_id: int,
) -> ParsedResponse:
    """Parse GptOss (Harmony) completion tokens.

    Finds the earliest terminal token (<|return|> or <|call|>), then walks the
    token stream block-by-block to extract:

    - analysis channel              → reasoning_content
    - final channel                 → content
    - commentary with to=functions.*  → tool_calls (JSON arguments)
    - commentary without recipient  → content (preamble text)
    """
    import re

    # Only <|return|> terminates the whole turn. <|call|> closes an
    # individual tool-call commentary block — a single turn may contain
    # several, so we must NOT truncate at the first <|call|>.
    return_pos = _find(token_ids, return_id)
    if return_pos != -1:
        ids = token_ids[:return_pos]
    else:
        ids = list(token_ids)

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[ParsedToolCall] = []

    i = 0
    while i < len(ids):
        if ids[i] != start_id:
            i += 1
            continue

        block_start = i
        msg_pos = _find(ids, message_id, i + 1)
        if msg_pos == -1:
            break

        header_ids = ids[i + 1 : msg_pos]
        header_text = _decode(tokenizer, header_ids)

        body_start = msg_pos + 1
        candidates = [
            pos
            for pos in (
                _find(ids, start_id, body_start),
                _find(ids, end_id, body_start),
                _find(ids, call_id, body_start),
            )
            if pos != -1
        ]
        body_end = min(candidates) if candidates else len(ids)
        body_closed = bool(candidates) and ids[body_end] in (end_id, call_id)

        body_text = _decode(tokenizer, ids[body_start:body_end])

        channel = _gptoss_extract_after_token(tokenizer, header_ids, channel_id)

        recipient_match = re.search(r"to=([^\s<]+)", header_text)
        recipient = recipient_match.group(1) if recipient_match else None

        if recipient and recipient.startswith("functions."):
            tool_name = recipient[len("functions.") :]
            block_end = body_end + 1 if body_closed else body_end
            span = (block_start, block_end)
            try:
                arguments = json.loads(body_text)
            except json.JSONDecodeError:
                tool_calls.append(
                    ParsedToolCall(
                        raw=body_text,
                        name=tool_name or None,
                        arguments=body_text,
                        token_span=span,
                        status=ToolCallParseStatus.INVALID_JSON,
                    )
                )
            else:
                if not body_closed:
                    status = ToolCallParseStatus.UNCLOSED_BLOCK
                elif not tool_name:
                    status = ToolCallParseStatus.MISSING_NAME
                else:
                    status = ToolCallParseStatus.OK
                tool_calls.append(
                    ParsedToolCall(
                        raw=body_text,
                        name=tool_name or None,
                        arguments=arguments,
                        token_span=span,
                        status=status,
                    )
                )
        elif channel == "analysis":
            reasoning_parts.append(body_text)
        elif channel == "final":
            content_parts.append(body_text)
        elif channel == "commentary":
            content_parts.append(body_text)

        i = body_end
        if i < len(ids) and ids[i] in (end_id, call_id):
            i += 1

    reasoning = "".join(reasoning_parts).strip() or None
    content = "".join(content_parts).strip()

    return ParsedResponse(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )


def _gptoss_extract_after_token(
    tokenizer,
    header_ids: list[int],
    marker_id: int,
) -> str | None:
    """Return the first decoded word appearing after marker_id in header_ids."""
    pos = _find(header_ids, marker_id)
    if pos == -1:
        return None
    after = _decode(tokenizer, header_ids[pos + 1 :]).strip()
    return after.split()[0] if after else None
