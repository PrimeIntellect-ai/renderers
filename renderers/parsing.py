"""Token-level parsing — operates on token IDs directly.

Finds special token boundaries by scanning token IDs, then decodes only
the text segments between them. No regex on decoded text, no false positives
from content that happens to look like special tokens.

Every parser emits semantic tool-call records aligned with one packed
fixed-width span array. Callers filter on ``status == OK`` for the clean
subset; verifier and RL-loss code uses the rest. This diverges from
vLLM's ``ExtractedToolCallInformation`` (single ``tools_called`` bool, no
per-call status) and SGLang's ``StreamingParseResult`` (silent drop on
failure) — see ``ToolCallParseStatus`` docstring for the rationale.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from renderers.base import ParsedResponse, ParsedToolCall, ParsedToolCallBuilder, ToolCallParseStatus, ToolSpec
from renderers.parsers import ParsedToolCallResult
from renderers.token_arrays import TOKEN_IDS_DTYPE, empty_span_array, require_1d_array, require_readonly


# ── Schema-aware argument coercion ──────────────────────────────────
#
# XML-style tool-call formats render argument values verbatim inside
# ``<arg_value>`` tags with no quoting. ``true`` and the string
# ``"true"`` produce identical wire bytes; without the tool schema, the
# parser has no signal to distinguish them and defaults to
# ``json.loads`` (the historical behavior). When the caller passes
# ``tools=[...]``, parsers consult the per-parameter declared type to
# keep string args verbatim, matching vLLM / SGLang reference parsers.


def _build_param_type_index(tools: list[ToolSpec] | None) -> dict[str, dict[str, dict[str, Any]]]:
    """Map tool name → param name → param JSON-schema fragment.

    Accepts both flat ``ToolSpec`` (``{name, description, parameters}``)
    and the OpenAI envelope (``{"type": "function", "function": {...}}``)
    so callers can pass either shape.
    """
    if not tools:
        return {}
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for tool in tools:
        spec = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not isinstance(name, str):
            continue
        params = spec.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(props, dict):
            index[name] = {k: v for k, v in props.items() if isinstance(v, dict)}
    return index


def _extract_tool_names(tools: list[ToolSpec] | None) -> set[str] | None:
    """Set of declared tool names, or ``None`` when ``tools`` is empty.

    ``None`` disables name validation — mirroring vLLM's
    ``ParserEngine._is_valid_tool_name``, which returns ``True`` whenever
    the request carries no tools. Accepts both flat ``ToolSpec`` and the
    OpenAI ``{"type": "function", "function": {...}}`` envelope, like
    ``_build_param_type_index`` (but independent of it: a tool with no
    ``parameters.properties`` still counts as a known name).
    """
    if not tools:
        return None
    names: set[str] = set()
    for tool in tools:
        spec = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(spec, dict) and isinstance(spec.get("name"), str):
            names.add(spec["name"])
    return names


def _coerce_arg_value(text: str, param_schema: dict[str, Any] | None) -> tuple[Any, bool]:
    """Coerce a raw ``<arg_value>`` body to its declared type.

    Returns ``(value, used_json_fallback)``. The boolean is ``True`` only
    when ``json.loads`` was attempted, raised, AND the schema doesn't
    permit a string. Returning a string verbatim because the schema
    permits strings is NOT a fallback.

    Rule (matches vLLM / SGLang reference parsers):

    - If the param's declared ``type`` is ``"string"`` (or single-element
      ``["string"]``), return ``text`` verbatim — never ``json.loads``.
    - Otherwise try ``json.loads``. If that fails, return raw ``text``.
      The ``used_json_fallback`` flag is ``True`` only when the schema
      does NOT permit a string — i.e. the fallback is truly suspect.

    Union types (``anyOf``/``oneOf``) that include ``"string"`` alongside
    other types still attempt ``json.loads`` first so an explicit
    integer / bool can parse; the string branch wins as fallback, and
    landing there is expected — not a malformed-JSON signal.
    """
    string_is_allowed = False
    if param_schema is not None:
        declared = param_schema.get("type")
        if declared == "string" or declared == ["string"]:
            return text, False
        for branch in param_schema.get("anyOf") or param_schema.get("oneOf") or []:
            if isinstance(branch, dict) and branch.get("type") == "string":
                string_is_allowed = True
                break
    try:
        return json.loads(text), False
    except (json.JSONDecodeError, ValueError):
        return text, not string_is_allowed


def _find(ids: np.ndarray, target: int, start: int = 0) -> int:
    """Find index of target in ids, or -1."""
    positions = np.flatnonzero(ids[start:] == target)
    return start + int(positions[0]) if positions.size else -1


def _find_any(ids: np.ndarray, targets: set[int], start: int = 0) -> int:
    """Find first index in ids whose value is in targets, or -1."""
    if not targets:
        return -1
    positions = np.flatnonzero(np.isin(ids[start:], np.fromiter(targets, dtype=TOKEN_IDS_DTYPE)))
    return start + int(positions[0]) if positions.size else -1


def _find_all(ids: np.ndarray, target: int) -> np.ndarray:
    """Find all indices of target in ids."""
    positions = np.flatnonzero(ids == target).astype(np.dtype("<i8"), copy=False)
    positions.flags.writeable = False
    return positions


def _strip_stop_tokens(ids: np.ndarray, stop_ids: set[int]) -> np.ndarray:
    """Truncate at first stop token (model shouldn't generate past it)."""
    require_1d_array("token_ids", ids, dtype=TOKEN_IDS_DTYPE, minimum=0)
    require_readonly("token_ids", ids)
    if not stop_ids:
        return ids
    positions = np.flatnonzero(np.isin(ids, np.fromiter(stop_ids, dtype=TOKEN_IDS_DTYPE)))
    return ids[: int(positions[0])] if positions.size else ids


def _decode(tokenizer, ids: np.ndarray) -> str:
    """Decode token IDs to text, skipping special tokens."""
    if ids.size == 0:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


def _reasoning_end_token_index(tokenizer, ids: np.ndarray, marker: str = "</think>") -> int:
    """Token index immediately past the first ``</think>`` in ``ids``.

    Returns 0 when ``ids`` has no closed reasoning region — callers treat
    that as "scan from the start" (preserves pre-existing behavior for
    non-thinking / truncated-reasoning completions).

    Used by parsers whose ``</think>`` is *not* a single special token
    (DeepSeek-V3, Kimi-K2.5) — where it tokenizes to several pieces and is
    context-sensitive (the closing ``>`` merges differently depending on the
    next char), so a token-id or fixed-subsequence search isn't reliable. We
    instead locate the boundary in decoded text via binary search over prefix
    decodes, which holds as long as ``decode(ids[:k])`` is prefix-stable in
    ``k`` (true for the byte-level BPE tokenizers here; ``</think>`` is clean
    ASCII that won't straddle a byte boundary). Single-token ``</think>``
    parsers (Qwen3) anchor on the token id directly and don't need this.
    """
    if ids.size == 0 or marker not in _decode(tokenizer, ids):
        return 0
    # Smallest prefix length (in tokens) whose decode already contains the
    # full marker — i.e. the index just past where </think> completes.
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if marker in _decode(tokenizer, ids[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _parsed_response(
    *, content: str, reasoning_content: str | None, tool_calls: ParsedToolCallBuilder
) -> ParsedResponse:
    calls, spans = tool_calls.finish()
    return ParsedResponse(
        content=content, reasoning_content=reasoning_content, tool_calls=calls, tool_call_token_spans=spans
    )


def _empty_parsed_response(*, content: str, reasoning_content: str | None = None) -> ParsedResponse:
    return ParsedResponse(
        content=content, reasoning_content=reasoning_content, tool_calls=(), tool_call_token_spans=empty_span_array()
    )


# ── Qwen3: <tool_call> JSON </tool_call> ────────────────────────────


def parse_qwen3(
    tokenizer,
    token_ids: np.ndarray,
    *,
    stop_ids: set[int],
    tool_call_id: int,
    tool_call_end_id: int,
    reasoning_end_id: int | None = None,
) -> ParsedResponse:
    """Parse Qwen3 completion tokens. Hermes-style JSON tool calls."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Reasoning is resolved before tool calls. Thinking models (e.g.
    # Qwen3-*-Thinking) routinely draft ``<tool_call>`` blocks *inside* their
    # ``<think>...</think>`` trace while planning; those are reasoning, not
    # real invocations. Anchoring the tool-call scan after the ``</think>``
    # boundary keeps in-think drafts out of ``tool_calls`` (otherwise they
    # surface as phantom/duplicate calls) and out of the reasoning/content
    # split. Mirrors vLLM's DelegatingParser, which runs the reasoning parser
    # first and tool-parses only the post-``</think>`` content.
    # ``reasoning_end_id`` is the ``</think>`` token id; when it's absent
    # (``None``) or the model never closed its reasoning, the scan falls back
    # to the whole stream (prior behavior).
    reasoning_end = _find(ids, reasoning_end_id) if reasoning_end_id is not None else -1
    scan_start = reasoning_end + 1 if reasoning_end != -1 else 0

    tc_start = _find(ids, tool_call_id, scan_start)
    tool_calls = ParsedToolCallBuilder()
    if tc_start != -1:
        content_ids = ids[:tc_start]
        i = tc_start
        while i < len(ids):
            i = _find(ids, tool_call_id, i)
            if i == -1:
                break
            end = _find(ids, tool_call_end_id, i + 1)
            if end == -1:
                raw = _decode(tokenizer, ids[i + 1 :]).strip()
                tool_calls.append(ParsedToolCall(raw=raw, status=ToolCallParseStatus.UNCLOSED_BLOCK), i, len(ids))
                break
            tc_text = _decode(tokenizer, ids[i + 1 : end]).strip()
            try:
                parsed = json.loads(tc_text)
            except json.JSONDecodeError:
                call = ParsedToolCall(raw=tc_text, status=ToolCallParseStatus.INVALID_JSON)
            else:
                name = parsed.get("name", "") if isinstance(parsed, dict) else ""
                arguments = parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
                call = ParsedToolCall(
                    raw=tc_text,
                    name=name or None,
                    arguments=arguments,
                    status=ToolCallParseStatus.MISSING_NAME if not name else ToolCallParseStatus.OK,
                )
            tool_calls.append(call, i, end + 1)
            i = end + 1
    else:
        content_ids = ids

    text = _decode(tokenizer, content_ids)
    # Extract reasoning from text (Qwen3 doesn't have <think> as special token)
    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").strip("\n").strip()
        text = after.strip("\n")

    return _parsed_response(content=text.strip(), reasoning_content=reasoning or None, tool_calls=tool_calls)


# ── Qwen3.5: <tool_call> <function=name> <parameter=name> v </parameter> </function> </tool_call>


def parse_qwen35(
    tokenizer,
    token_ids: np.ndarray,
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse Qwen3.5 completion tokens. XML-style tool calls, token-level thinking.

    ``reasoning_content`` contract: ``None`` means the completion carried no
    think block at all; a string (possibly empty) means a think block was
    present. The distinction matters for renderers with key-presence
    semantics (PrimeQwen3Renderer): a sampled ``<think></think>`` must
    round-trip parse → message → render back to the same tokens, so an
    empty-but-present block is ``""``, never ``None``.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Thinking: find </think> by token ID
    reasoning = None
    parse_offset = 0  # shift to map local indices back to stop-stripped ids
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = reasoning_ids[reasoning_ids != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif _find(ids, think_id) != -1:
        # <think> present but no </think> — truncated reasoning. Block
        # present ⇒ string (see docstring), even when nothing follows the
        # opening tag.
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return _empty_parsed_response(content="", reasoning_content=reasoning)

    tc_start = _find(ids, tool_call_id)
    tool_calls = ParsedToolCallBuilder()
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        _parse_xml_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            section_offset=parse_offset + tc_start,
            param_index=_build_param_type_index(tools),
            tool_calls=tool_calls,
        )
    else:
        content_text = _decode(tokenizer, ids).strip()

    # ``reasoning`` is None only when no think block was found; keep ""
    # (empty-but-present block) intact so ``<think></think>`` samples
    # round-trip — collapsing it to None made PrimeQwen3Renderer drop the
    # block on re-render, forking the token stream for the rest of the
    # rollout.
    return _parsed_response(content=content_text, reasoning_content=reasoning, tool_calls=tool_calls)


def _parse_xml_tool_calls(
    tokenizer,
    ids: np.ndarray,
    tc_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
    param_index: dict[str, dict[str, dict[str, Any]]],
    tool_calls: ParsedToolCallBuilder,
) -> None:
    """Parse Qwen3.5-style XML tool calls from token IDs."""
    import re

    i = 0
    while i < len(ids):
        i = _find(ids, tc_id, i)
        if i == -1:
            break
        end = _find(ids, tc_end_id, i + 1)
        if end == -1:
            raw = _decode(tokenizer, ids[i + 1 :])
            tool_calls.append(
                ParsedToolCall(raw=raw, status=ToolCallParseStatus.UNCLOSED_BLOCK),
                section_offset + i,
                section_offset + len(ids),
            )
            break
        block_text = _decode(tokenizer, ids[i + 1 : end])
        name_match = re.search(r"<function=([^>]+)>", block_text)
        if not name_match:
            tool_calls.append(
                ParsedToolCall(raw=block_text, status=ToolCallParseStatus.MALFORMED_STRUCTURE),
                section_offset + i,
                section_offset + end + 1,
            )
            i = end + 1
            continue

        name = name_match.group(1)
        params = param_index.get(name, {})
        arguments: dict = {}
        any_json_fallback = False
        for pm in re.finditer(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", block_text, re.DOTALL):
            arg_name = pm.group(1)
            arg_value = pm.group(2).strip()
            value, used_fallback = _coerce_arg_value(arg_value, params.get(arg_name))
            arguments[arg_name] = value
            any_json_fallback = any_json_fallback or used_fallback
        tool_calls.append(
            ParsedToolCall(
                raw=block_text,
                name=name,
                arguments=arguments,
                status=ToolCallParseStatus.INVALID_JSON if any_json_fallback else ToolCallParseStatus.OK,
            ),
            section_offset + i,
            section_offset + end + 1,
        )
        i = end + 1


# ── GLM-5/4.7/4.5: <tool_call> name <arg_key>k</arg_key> <arg_value>v</arg_value> </tool_call>


def parse_glm(
    tokenizer,
    token_ids: np.ndarray,
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
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse GLM completion tokens. Token-level thinking + arg_key/arg_value tool calls.

    When ``tools`` is passed, tool names are validated against it: a call
    whose name isn't declared gets ``status=UNKNOWN_TOOL`` instead of
    ``OK``. This mirrors vLLM ≥ 0.24, where the ``glm45``/``glm47`` tool
    parsers run with ``validate_tool_names=True`` and silently drop
    unknown-name calls (``vllm/parser/glm47_moe.py``,
    ``ParserEngine._is_valid_tool_name``) — so a completion that yields no
    tool call from the engine also yields no ``OK`` call here, and
    downstream finish-reason promotion (``renderers/client.py``) agrees
    with an OpenAI chat-completions client talking to the same engine.
    Notably this covers the missing-``<arg_key>`` shape
    (``<tool_call>bash\\n<arg_value>...</arg_value></tool_call>``): both
    this parser and vLLM's resolve the whole block as the name, which then
    fails validation. Without ``tools``, no validation happens (vLLM
    behaves the same when the request carries no tools).
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = reasoning_ids[reasoning_ids != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif _find(ids, think_id) != -1:
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return _empty_parsed_response(content="", reasoning_content=reasoning or None)

    tc_start = _find(ids, tool_call_id)
    tool_calls = ParsedToolCallBuilder()
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        _parse_glm_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            arg_key_id,
            arg_key_end_id,
            arg_value_id,
            arg_value_end_id,
            section_offset=parse_offset + tc_start,
            param_index=_build_param_type_index(tools),
            known_names=_extract_tool_names(tools),
            tool_calls=tool_calls,
        )
    else:
        content_text = _decode(tokenizer, ids).strip()

    return _parsed_response(content=content_text, reasoning_content=reasoning or None, tool_calls=tool_calls)


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
    param_index: dict[str, dict[str, dict[str, Any]]],
    known_names: set[str] | None = None,
    tool_calls: ParsedToolCallBuilder,
) -> None:
    """Parse GLM-style tool calls: name + arg_key/arg_value pairs, all by token ID."""
    i = 0
    while i < len(ids):
        i = _find(ids, tc_id, i)
        if i == -1:
            break
        end = _find(ids, tc_end_id, i + 1)
        if end == -1:
            raw = _decode(tokenizer, ids[i + 1 :])
            tool_calls.append(
                ParsedToolCall(raw=raw, status=ToolCallParseStatus.UNCLOSED_BLOCK),
                section_offset + i,
                section_offset + len(ids),
            )
            break
        block = ids[i + 1 : end]
        block_text = _decode(tokenizer, block)
        first_ak = _find(block, ak_id)
        any_json_fallback = False
        structure_broke = False
        if first_ak == -1:
            name = _decode(tokenizer, block).strip()
            arguments: dict = {}
        else:
            name = _decode(tokenizer, block[:first_ak]).strip()
            params = param_index.get(name, {})
            arguments = {}
            j = first_ak
            while j < len(block):
                j = _find(block, ak_id, j)
                if j == -1:
                    break
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
                value, used_fallback = _coerce_arg_value(val_text, params.get(key))
                arguments[key] = value
                any_json_fallback = any_json_fallback or used_fallback
                j = ave + 1
        if not name:
            status = ToolCallParseStatus.MISSING_NAME
        elif known_names is not None and name not in known_names:
            status = ToolCallParseStatus.UNKNOWN_TOOL
        elif structure_broke:
            status = ToolCallParseStatus.MALFORMED_STRUCTURE
        elif any_json_fallback:
            status = ToolCallParseStatus.INVALID_JSON
        else:
            status = ToolCallParseStatus.OK
        tool_calls.append(
            ParsedToolCall(raw=block_text, name=name or None, arguments=arguments, status=status),
            section_offset + i,
            section_offset + end + 1,
        )
        i = end + 1


# ── Hy3: <think>…</think> content <tool_calls> <tool_call>name<tool_sep> …
#        <arg_key>k</arg_key> <arg_value>v</arg_value> … </tool_call> </tool_calls>
# Same arg_key/arg_value token scheme as GLM, but each call names the function
# between <tool_call> and <tool_sep>, and the calls are wrapped in an outer
# <tool_calls></tool_calls> pair. All markers are single special tokens.


def parse_hy3(
    tokenizer,
    token_ids: np.ndarray,
    *,
    stop_ids: set[int],
    assistant_id: int,
    think_id: int,
    think_end_id: int,
    tool_calls_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    tool_sep_id: int,
    arg_key_id: int,
    arg_key_end_id: int,
    arg_value_id: int,
    arg_value_end_id: int,
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse Hy3 completion tokens.

    Handles both the inference stream (``reasoning</think>content…`` in
    ``low``/``high`` mode, or bare ``content…`` in ``no_think`` mode, since
    the ``<think>`` opener lives in the generation prompt) and a round-trip
    slice that still carries a leading ``<｜hy_Assistant｜>`` opener and a
    full ``<think>…</think>`` block.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Packed span rows are reported relative to this stop-stripped stream
    # (the documented contract), so track every prefix we slice off below.
    offset = 0

    # A round-trip slice includes the assistant role marker; the live
    # inference stream does not. Drop a single leading opener so both paths
    # land on the same downstream logic.
    if ids.size and ids[0] == assistant_id:
        ids = ids[1:]
        offset += 1

    reasoning = None
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = reasoning_ids[reasoning_ids != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        offset += think_end + 1
    elif _find(ids, think_id) != -1:
        # Reasoning opened but never closed (truncation): everything after
        # the opener is reasoning; there is no committed content yet.
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return _empty_parsed_response(content="", reasoning_content=reasoning or None)

    # Content ends at the first tool marker — the outer <tool_calls> wrapper
    # or, defensively, a bare <tool_call> the model emitted without it.
    marker_positions = [p for p in (_find(ids, tool_calls_id), _find(ids, tool_call_id)) if p != -1]
    tool_calls = ParsedToolCallBuilder()
    if marker_positions:
        tool_start = min(marker_positions)
        content_text = _decode(tokenizer, ids[:tool_start]).strip()
        _parse_hy3_tool_calls(
            tokenizer,
            ids[tool_start:],
            tool_call_id,
            tool_call_end_id,
            tool_sep_id,
            arg_key_id,
            arg_key_end_id,
            arg_value_id,
            arg_value_end_id,
            section_offset=offset + tool_start,
            param_index=_build_param_type_index(tools),
            tool_calls=tool_calls,
        )
    else:
        content_text = _decode(tokenizer, ids).strip()

    return _parsed_response(content=content_text, reasoning_content=reasoning or None, tool_calls=tool_calls)


def _parse_hy3_tool_calls(
    tokenizer,
    ids,
    tc_id,
    tc_end_id,
    sep_id,
    ak_id,
    ake_id,
    av_id,
    ave_id,
    *,
    section_offset: int,
    param_index: dict[str, dict[str, dict[str, Any]]],
    tool_calls: ParsedToolCallBuilder,
) -> None:
    """Parse Hy3-style tool calls: ``<tool_call>name<tool_sep>`` then
    arg_key/arg_value pairs, all by token ID. The outer ``<tool_calls>``
    wrapper and inter-block newlines are skipped by scanning for the
    per-call ``<tool_call>`` opener. ``section_offset`` shifts recorded
    packed span rows back into the stop-stripped completion stream
    (``ids`` here is a suffix of it)."""
    i = 0
    while i < len(ids):
        i = _find(ids, tc_id, i)
        if i == -1:
            break
        end = _find(ids, tc_end_id, i + 1)
        if end == -1:
            raw = _decode(tokenizer, ids[i + 1 :])
            tool_calls.append(
                ParsedToolCall(raw=raw, status=ToolCallParseStatus.UNCLOSED_BLOCK),
                section_offset + i,
                section_offset + len(ids),
            )
            break
        block = ids[i + 1 : end]
        block_text = _decode(tokenizer, block)

        # Name sits between <tool_call> and <tool_sep>. Fall back to the
        # first <arg_key> boundary if the separator is missing.
        sep = _find(block, sep_id)
        if sep != -1:
            name = _decode(tokenizer, block[:sep]).strip()
            arg_ids = block[sep + 1 :]
        else:
            first_ak = _find(block, ak_id)
            if first_ak == -1:
                name = _decode(tokenizer, block).strip()
                arg_ids = block[:0]
            else:
                name = _decode(tokenizer, block[:first_ak]).strip()
                arg_ids = block[first_ak:]

        params = param_index.get(name, {})
        arguments: dict = {}
        structure_broke = False
        any_json_fallback = False
        j = 0
        while j < len(arg_ids):
            j = _find(arg_ids, ak_id, j)
            if j == -1:
                break
            ake = _find(arg_ids, ake_id, j + 1)
            if ake == -1:
                structure_broke = True
                break
            key = _decode(tokenizer, arg_ids[j + 1 : ake]).strip()
            av = _find(arg_ids, av_id, ake + 1)
            if av == -1:
                structure_broke = True
                break
            ave = _find(arg_ids, ave_id, av + 1)
            if ave == -1:
                structure_broke = True
                break
            val_text = _decode(tokenizer, arg_ids[av + 1 : ave]).strip()
            value, used_fallback = _coerce_arg_value(val_text, params.get(key))
            arguments[key] = value
            any_json_fallback = any_json_fallback or used_fallback
            j = ave + 1

        if not name:
            status = ToolCallParseStatus.MISSING_NAME
        elif structure_broke:
            status = ToolCallParseStatus.MALFORMED_STRUCTURE
        elif any_json_fallback:
            status = ToolCallParseStatus.INVALID_JSON
        else:
            status = ToolCallParseStatus.OK
        tool_calls.append(
            ParsedToolCall(raw=block_text, name=name or None, arguments=arguments, status=status),
            section_offset + i,
            section_offset + end + 1,
        )
        i = end + 1


# ── Laguna-XS.2: <tool_call> name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value> </tool_call>
# Same outer skeleton as parse_glm, but <arg_key>/<arg_value> are plain text
# (multi-token BPE), not single special tokens — so the inner block is decoded
# to text and the key/value pairs are pulled out by regex.


def parse_laguna_xs2(
    tokenizer,
    token_ids: np.ndarray,
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    tools: list[ToolSpec] | None = None,
    strip_newlines: bool = True,
) -> ParsedResponse:
    """Parse Laguna-XS.2 / XS-2.1 completion tokens.

    Thinking uses single-token ``<think>`` / ``</think>`` (ids found by
    scan). Tool calls are delimited by single-token ``<tool_call>`` /
    ``</tool_call>``, but ``<arg_key>`` / ``<arg_value>`` inside are
    plain text — regex-extracted from the decoded inner block.

    ``strip_newlines`` mirrors the template's whitespace around reasoning
    and content. XS.2 wraps reasoning with ``\\n`` on both sides
    (``<think>\\n{r}\\n</think>``) and brackets post-think content with
    ``\\n`` too (``</think>\\n{c}\\n``) — strip exactly those newlines,
    never a bare ``.strip()``, which would also eat whitespace the model
    emitted intentionally. XS-2.1 renders both segments verbatim, so it
    parses with ``strip_newlines=False``.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    def _segment(segment_ids: np.ndarray) -> str:
        text = _decode(tokenizer, segment_ids)
        return text.strip("\n") if strip_newlines else text

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = reasoning_ids[reasoning_ids != think_id]
        reasoning = _segment(reasoning_ids)
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif (think_start := _find(ids, think_id)) != -1:
        reasoning = _segment(ids[think_start + 1 :])
        return _empty_parsed_response(content="", reasoning_content=reasoning or None)

    tc_start = _find(ids, tool_call_id)
    tool_calls = ParsedToolCallBuilder()
    if tc_start != -1:
        content_text = _segment(ids[:tc_start])
        _parse_laguna_xs2_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            section_offset=parse_offset + tc_start,
            param_index=_build_param_type_index(tools),
            tool_calls=tool_calls,
        )
    else:
        content_text = _segment(ids)

    return _parsed_response(content=content_text, reasoning_content=reasoning or None, tool_calls=tool_calls)


def _parse_laguna_xs2_tool_calls(
    tokenizer,
    ids: np.ndarray,
    tc_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
    param_index: dict[str, dict[str, dict[str, Any]]],
    tool_calls: ParsedToolCallBuilder,
) -> None:
    """Parse Laguna-XS.2 / XS-2.1 tool calls.

    Inside each ``<tool_call>...</tool_call>`` block, the format is::

        {name}
        <arg_key>{k1}</arg_key><arg_value>{v1}</arg_value>
        ...
        <arg_key>{kn}</arg_key><arg_value>{vn}</arg_value>

    XS.2 puts a ``\\n`` after the name and between the tag pairs; XS-2.1
    packs everything tightly. The function name is everything before the
    first ``<arg_key>`` literal in the decoded block (stripped), and the
    key/value regex allows optional whitespace between the tags, so both
    layouts parse identically.
    """
    import re

    i = 0
    while i < len(ids):
        i = _find(ids, tc_id, i)
        if i == -1:
            break
        tc_end = _find(ids, tc_end_id, i + 1)
        if tc_end == -1:
            raw = _decode(tokenizer, ids[i + 1 :])
            tool_calls.append(
                ParsedToolCall(raw=raw, status=ToolCallParseStatus.UNCLOSED_BLOCK),
                section_offset + i,
                section_offset + len(ids),
            )
            break
        block_text = _decode(tokenizer, ids[i + 1 : tc_end])

        ak_pos = block_text.find("<arg_key>")
        if ak_pos != -1:
            name = block_text[:ak_pos].strip()
            args_section = block_text[ak_pos:]
        else:
            name = block_text.strip()
            args_section = ""

        params = param_index.get(name, {})
        arguments: dict = {}
        any_json_fallback = False
        for m in re.finditer(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", args_section, re.DOTALL):
            k = m.group(1).strip()
            v = m.group(2).strip()
            value, used_fallback = _coerce_arg_value(v, params.get(k))
            arguments[k] = value
            any_json_fallback = any_json_fallback or used_fallback

        if not name:
            status = ToolCallParseStatus.MISSING_NAME
        elif any_json_fallback:
            status = ToolCallParseStatus.INVALID_JSON
        else:
            status = ToolCallParseStatus.OK

        tool_calls.append(
            ParsedToolCall(raw=block_text, name=name or None, arguments=arguments, status=status),
            section_offset + i,
            section_offset + tc_end + 1,
        )
        i = tc_end + 1


# ── DeepSeek V3: <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> + text <think> tags ──


def parse_deepseek_v3(
    tokenizer,
    token_ids: np.ndarray,
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

    # Reasoning first: skip past </think> before looking for the tool-call
    # section, so a section the model drafts *inside* its <think> trace isn't
    # parsed as a real call (regression #78 — cf. parse_qwen3). content_ids
    # still starts at 0, so the </think> text-split below recovers reasoning.
    # DeepSeek-V3 renders </think> as multi-token text, hence the decode-based
    # boundary finder rather than a token-id anchor.
    reasoning_end = _reasoning_end_token_index(tokenizer, ids)

    tc_section_start = _find(ids, tool_calls_begin_id, reasoning_end)
    tool_calls = ParsedToolCallBuilder()
    if tc_section_start != -1:
        content_ids = ids[:tc_section_start]
        _parse_deepseek_tool_calls(
            tokenizer,
            ids[tc_section_start:],
            tool_calls_begin_id,
            tool_calls_end_id,
            tool_call_begin_id,
            tool_call_end_id,
            tool_sep_id,
            section_offset=tc_section_start,
            tool_calls=tool_calls,
        )
    else:
        content_ids = ids

    text = _decode(tokenizer, content_ids)

    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").lstrip("\n").rstrip("\n").strip()
        text = after.lstrip("\n")

    return _parsed_response(content=text.strip(), reasoning_content=reasoning or None, tool_calls=tool_calls)


def _parse_deepseek_tool_calls(
    tokenizer,
    ids: np.ndarray,
    tc_begin_id: int,
    tc_end_id: int,
    call_begin_id: int,
    call_end_id: int,
    sep_id: int,
    *,
    section_offset: int,
    tool_calls: ParsedToolCallBuilder,
) -> None:
    """Parse DeepSeek V3-style tool calls from token IDs."""
    import re

    section_start = _find(ids, tc_begin_id)
    if section_start == -1:
        return
    section_end = _find(ids, tc_end_id, section_start + 1)
    section_end_clipped = section_end == -1
    if section_end == -1:
        section_end = len(ids)

    inner_offset = section_offset + section_start + 1
    section_ids = ids[section_start + 1 : section_end]

    i = 0
    while i < len(section_ids):
        i = _find(section_ids, call_begin_id, i)
        if i == -1:
            break
        end = _find(section_ids, call_end_id, i + 1)
        unclosed = end == -1
        if unclosed:
            end = len(section_ids)
        call_ids = section_ids[i + 1 : end]
        block_text = _decode(tokenizer, call_ids)
        span_end = inner_offset + end + (0 if unclosed else 1)

        sep_pos = _find(call_ids, sep_id)
        if sep_pos == -1:
            tool_calls.append(
                ParsedToolCall(raw=block_text, status=ToolCallParseStatus.MALFORMED_STRUCTURE),
                inner_offset + i,
                span_end,
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
            ParsedToolCall(raw=block_text, name=name or None, arguments=arguments, status=status),
            inner_offset + i,
            span_end,
        )
        i = end + 1
        if unclosed:
            break

    # If the outer <tool_calls_begin> had no matching <tool_calls_end>, any
    # call inside that didn't itself flag UNCLOSED_BLOCK is still nested in
    # a truncated section — but we already mark individual unclosed calls,
    # so we don't double-flag here. The section_end_clipped variable is
    # carried for the (rare) caller that wants section-level UX.
    _ = section_end_clipped


# ── DeepSeek V4: DSML tool calls + single-token </think> ────────────


def parse_deepseek_v4(
    tokenizer, token_ids: np.ndarray, *, stop_ids: set[int], thinking_enabled: bool, think_end_id: int, dsml_id: int
) -> ParsedResponse:
    """Parse a DeepSeek V4 completion.

    Thinking mode prefills ``<think>`` in the prompt, so the completion starts
    with reasoning and closes it with the single-token ``</think>``.  Tool
    calls use DSML markup; the ``｜DSML｜`` marker is a special token, and the
    decoded grammar carries an explicit ``string=`` flag for lossless argument
    type recovery.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    reasoning: str | None = None
    content_offset = 0
    if thinking_enabled:
        think_end = _find(ids, think_end_id)
        if think_end == -1:
            return _empty_parsed_response(content="", reasoning_content=_decode(tokenizer, ids) or None)
        reasoning = _decode(tokenizer, ids[:think_end])
        content_offset = think_end + 1

    content_ids = ids[content_offset:]
    decoded = _decode(tokenizer, content_ids)
    section_marker = "\n\n<｜DSML｜tool_calls>"
    section_pos = decoded.find(section_marker)
    if section_pos == -1 or _find(content_ids, dsml_id) == -1:
        return _empty_parsed_response(content=decoded, reasoning_content=reasoning or None)

    content = decoded[:section_pos]
    section_text = decoded[section_pos:]
    section_token_offset = content_offset + _decoded_char_to_token_index(tokenizer, content_ids, section_pos)
    tool_calls = ParsedToolCallBuilder()
    _parse_deepseek_v4_tool_calls(
        tokenizer, section_text, ids[section_token_offset:], section_offset=section_token_offset, tool_calls=tool_calls
    )
    return _parsed_response(content=content, reasoning_content=reasoning or None, tool_calls=tool_calls)


def _decoded_char_to_token_index(tokenizer, ids: np.ndarray, char_index: int) -> int:
    """Return the first token boundary at or beyond a decoded char offset."""
    if char_index <= 0:
        return 0
    for boundary in range(1, len(ids) + 1):
        if len(_decode(tokenizer, ids[:boundary])) >= char_index:
            return boundary
    return len(ids)


def _parse_deepseek_v4_tool_calls(
    tokenizer, section_text: str, section_ids: np.ndarray, *, section_offset: int, tool_calls: ParsedToolCallBuilder
) -> None:
    """Parse every DSML ``invoke`` attempt from one tool-calls section."""
    import re

    invoke_start = '<｜DSML｜invoke name="'
    invoke_end = "</｜DSML｜invoke>"
    section_end = "</｜DSML｜tool_calls>"
    parameter_pattern = re.compile(
        r'<｜DSML｜parameter name="(.*?)" string="(true|false)">'
        r"(.*?)</｜DSML｜parameter>",
        re.DOTALL,
    )

    outer_end = section_text.find(section_end)
    for invoke_match in re.finditer(r"<｜DSML｜invoke", section_text):
        start = invoke_match.start()
        if outer_end != -1 and outer_end < start:
            break

        close = section_text.find(invoke_end, start + len(invoke_start))
        unclosed = close == -1 or (outer_end != -1 and outer_end < close)
        block_end = outer_end if unclosed and outer_end != -1 else len(section_text)
        if not unclosed:
            block_end = close + len(invoke_end)
        raw = section_text[start:block_end]

        token_start = _decoded_char_to_token_index(tokenizer, section_ids, start)
        token_end = _decoded_char_to_token_index(tokenizer, section_ids, block_end)

        header_end = section_text.find(">\n", start, block_end)
        name: str | None = None
        malformed = False
        if header_end == -1:
            malformed = True
            body = ""
        else:
            header = section_text[start : header_end + 2]
            name_match = re.fullmatch(r'<｜DSML｜invoke name="(.*?)">\n', header, flags=re.DOTALL)
            if name_match:
                name = name_match.group(1)
            else:
                malformed = True
            body_end = close if not unclosed else block_end
            body = section_text[header_end + 2 : body_end]

        arguments: dict[str, Any] = {}
        invalid_json = False
        for match in parameter_pattern.finditer(body):
            key, is_string, raw_value = match.groups()
            if key in arguments:
                malformed = True
                continue
            if is_string == "true":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = raw_value
                    invalid_json = True

        remainder = parameter_pattern.sub("", body)
        # The canonical form has one newline before ``</invoke>`` and newlines
        # between parameters.  Anything else left after removing parameter
        # blocks is structural debris.
        if remainder.strip("\n"):
            malformed = True

        if unclosed:
            status = ToolCallParseStatus.UNCLOSED_BLOCK
        elif not name:
            status = ToolCallParseStatus.MISSING_NAME
        elif malformed:
            status = ToolCallParseStatus.MALFORMED_STRUCTURE
        elif invalid_json:
            status = ToolCallParseStatus.INVALID_JSON
        else:
            status = ToolCallParseStatus.OK

        tool_calls.append(
            ParsedToolCall(raw=raw, name=name, arguments=arguments, status=status),
            section_offset + token_start,
            section_offset + token_end,
        )
        if unclosed:
            break


# ── MiniMax: <minimax:tool_call> ... </minimax:tool_call> ────────────


def parse_minimax(
    tokenizer,
    token_ids: np.ndarray,
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse MiniMax M2 completion tokens."""
    import re

    ids = _strip_stop_tokens(token_ids, stop_ids)
    param_index = _build_param_type_index(tools)

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = reasoning_ids[reasoning_ids != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif _find(ids, think_id) != -1:
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return _empty_parsed_response(content="", reasoning_content=reasoning or None)

    tc_start = _find(ids, tool_call_id)
    tool_calls = ParsedToolCallBuilder()
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        i = tc_start
        while i < len(ids):
            i = _find(ids, tool_call_id, i)
            if i == -1:
                break
            end = _find(ids, tool_call_end_id, i + 1)
            if end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(raw=raw, status=ToolCallParseStatus.UNCLOSED_BLOCK),
                    parse_offset + i,
                    parse_offset + len(ids),
                )
                break
            block_text = _decode(tokenizer, ids[i + 1 : end])

            matched_invoke = False
            for invoke_match in re.finditer(r'<invoke name="([^"]+)">(.*?)</invoke>', block_text, re.DOTALL):
                matched_invoke = True
                name = invoke_match.group(1)
                body = invoke_match.group(2)
                params = param_index.get(name, {})
                arguments: dict = {}
                any_json_fallback = False
                for pm in re.finditer(r'<parameter name="([^"]+)">(.*?)</parameter>', body, re.DOTALL):
                    pname = pm.group(1)
                    pval = pm.group(2).strip()
                    value, used_fallback = _coerce_arg_value(pval, params.get(pname))
                    arguments[pname] = value
                    any_json_fallback = any_json_fallback or used_fallback
                tool_calls.append(
                    ParsedToolCall(
                        raw=block_text,
                        name=name,
                        arguments=arguments,
                        status=(ToolCallParseStatus.INVALID_JSON if any_json_fallback else ToolCallParseStatus.OK),
                    ),
                    parse_offset + i,
                    parse_offset + end + 1,
                )
            if not matched_invoke:
                tool_calls.append(
                    ParsedToolCall(raw=block_text, status=ToolCallParseStatus.MALFORMED_STRUCTURE),
                    parse_offset + i,
                    parse_offset + end + 1,
                )
            i = end + 1
    else:
        content_text = _decode(tokenizer, ids).strip()

    return _parsed_response(content=content_text, reasoning_content=reasoning or None, tool_calls=tool_calls)


# ── Kimi K2: <|tool_calls_section_begin|> ... <|tool_calls_section_end|> ────


def parse_kimi_k2_section(
    tokenizer,
    ids: np.ndarray,
    *,
    tool_calls_section_begin_ids: set[int],
    tool_calls_section_end_ids: set[int],
    tool_call_begin_id: int,
    tool_call_argument_begin_id: int,
    tool_call_end_id: int,
    scan_start: int = 0,
) -> ParsedToolCallResult:
    """Split Kimi content and calls into one typed fixed-width result.

    Accepts *sets* of begin/end token IDs so callers can express models with
    multiple delimiter variants (K2.5 has both plural ``<|tool_calls_section_*|>``
    and singular ``<|tool_call_section_*|>`` forms, though only the plural form
    is in the special-token vocab in practice). The result keeps content ids,
    an immutable tuple of semantic calls, and one aligned packed span array;
    an unclosed section is still walked to whatever the model emitted before
    EOS.

    ``scan_start`` restricts the section search to ``ids[scan_start:]`` while
    keeping ``content_ids = ids[:section_start]`` and all token spans relative
    to the full ``ids``. Callers pass the post-``</think>`` index so a section
    the model drafts inside its reasoning trace isn't parsed as a real call;
    because ``content_ids`` still starts at 0, downstream text-based reasoning
    extraction is unaffected (regression #78).
    """
    require_1d_array("token_ids", ids, dtype=TOKEN_IDS_DTYPE, minimum=0)
    require_readonly("token_ids", ids)
    tool_calls = ParsedToolCallBuilder()
    section_start = _find_any(ids, tool_calls_section_begin_ids, scan_start)
    if section_start == -1:
        calls, spans = tool_calls.finish()
        return ParsedToolCallResult(ids, calls, spans)
    content_ids = ids[:section_start]
    section_end = _find_any(ids, tool_calls_section_end_ids, section_start + 1)
    if section_end == -1:
        section_end = len(ids)
    section_ids = ids[section_start + 1 : section_end]
    _parse_kimi_k2_tool_calls(
        tokenizer,
        section_ids,
        tool_call_begin_id,
        tool_call_argument_begin_id,
        tool_call_end_id,
        section_offset=section_start + 1,
        tool_calls=tool_calls,
    )
    calls, spans = tool_calls.finish()
    return ParsedToolCallResult(content_ids, calls, spans)


def parse_kimi_k2(
    tokenizer,
    token_ids: np.ndarray,
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

    parsed_tools = parse_kimi_k2_section(
        tokenizer,
        ids,
        tool_calls_section_begin_ids={tool_calls_section_begin_id},
        tool_calls_section_end_ids={tool_calls_section_end_id},
        tool_call_begin_id=tool_call_begin_id,
        tool_call_argument_begin_id=tool_call_argument_begin_id,
        tool_call_end_id=tool_call_end_id,
    )
    content_ids = parsed_tools.content_ids

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
        return _empty_parsed_response(content="", reasoning_content=reasoning)

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning,
        tool_calls=parsed_tools.tool_calls,
        tool_call_token_spans=parsed_tools.tool_call_token_spans,
    )


def _parse_kimi_k2_tool_calls(
    tokenizer,
    ids: np.ndarray,
    tc_begin_id: int,
    tc_arg_begin_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
    tool_calls: ParsedToolCallBuilder,
) -> None:
    """Parse individual Kimi K2 tool calls from the section token IDs.

    Format per call:
        <|tool_call_begin|>{id}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>

    The ``id`` is in format ``functions.name:index``; the function name is
    extracted by stripping the ``functions.`` prefix and ``:index`` suffix.
    """
    i = 0
    while i < len(ids):
        i = _find(ids, tc_begin_id, i)
        if i == -1:
            break
        arg_begin = _find(ids, tc_arg_begin_id, i + 1)
        if arg_begin == -1:
            raw = _decode(tokenizer, ids[i + 1 :])
            tool_calls.append(
                ParsedToolCall(raw=raw, status=ToolCallParseStatus.MALFORMED_STRUCTURE),
                section_offset + i,
                section_offset + len(ids),
            )
            break
        tc_end = _find(ids, tc_end_id, arg_begin + 1)
        unclosed = tc_end == -1
        if unclosed:
            tc_end = len(ids)

        raw_id = _decode(tokenizer, ids[i + 1 : arg_begin]).strip()
        args_str = _decode(tokenizer, ids[arg_begin + 1 : tc_end]).strip()
        block_text = _decode(tokenizer, ids[i + 1 : tc_end])
        span_end = section_offset + tc_end + (0 if unclosed else 1)

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
                raw=block_text, name=func_name or None, arguments=arguments, status=status, id=raw_id or None
            ),
            section_offset + i,
            span_end,
        )
        i = tc_end + 1
        if unclosed:
            break


# ── Llama-3: single JSON tool call {"name": "...", "parameters": {...}} ─


def parse_llama_3(tokenizer, token_ids: np.ndarray, *, stop_ids: set[int]) -> ParsedResponse:
    """Parse Llama-3 completion tokens.

    The Llama-3 chat template emits tool calls as a single JSON blob in
    the assistant body — ``{"name": "...", "parameters": {...}}`` — with
    no surrounding XML tags or special tokens. Plain replies are just
    text. We detect the tool-call shape with a strict starts-with-``{``
    + parses-as-dict-with-name-key check; anything else is treated as
    content. Llama-3 doesn't have a built-in reasoning channel, so
    ``reasoning_content`` is always ``None``.

    Unlike the delimiter-based formats (Qwen/GLM), the tool call has no
    special token to anchor on, so a leading assistant role-header
    (``<|start_header_id|>assistant<|end_header_id|>\\n\\n``) would defeat
    the starts-with-``{`` check. Callers that slice a completion without
    dropping the generation prompt include that scaffold; we skip past the
    final ``<|end_header_id|>`` so the body is what we parse. The sampled
    stream in production carries no header, making this a no-op there.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Skip a leading assistant role-header scaffold if present.
    body_start = 0
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    if isinstance(end_header_id, int):
        eh_positions = _find_all(ids, end_header_id)
        if eh_positions.size:
            body_start = int(eh_positions[-1]) + 1
    body_ids = ids[body_start:]
    text = _decode(tokenizer, body_ids).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("name"):
            arguments = parsed.get("parameters", parsed.get("arguments", {}))
            tool_calls = ParsedToolCallBuilder()
            tool_calls.append(
                ParsedToolCall(raw=text, name=parsed["name"], arguments=arguments, status=ToolCallParseStatus.OK),
                body_start,
                len(ids),
            )
            return _parsed_response(content="", reasoning_content=None, tool_calls=tool_calls)

    # Not a tool-call shape (plain reply, or a ``{...}`` body that didn't
    # parse / lacked a name). Llama-3 has no delimiter to anchor a
    # "malformed attempt" against, so it falls through to content rather
    # than producing a non-OK ParsedToolCall.
    return _empty_parsed_response(content=text)


def parse_inkling(
    tokenizer,
    token_ids: np.ndarray,
    *,
    stop_ids: set[int],
    message_model_id: int,
    content_text_id: int,
    content_thinking_id: int,
    invoke_json_id: int,
    invoke_text_id: int,
    end_message_id: int,
) -> ParsedResponse:
    """Parse Inkling completion tokens.

    Inkling's assistant turn is a sequence of ``<|end_message|>``-terminated
    segments; the model re-emits ``<|message_model|>`` before each segment
    after the first (the first's opener is the generation prompt). Each
    segment is classified by its leading content marker:

    - ``<|content_thinking|>{reasoning}`` → reasoning
    - ``<|content_text|>{content}`` → visible content
    - ``{name}<|content_invoke_tool_json|>{"name":…,"args":…}`` → tool call

    Content and reasoning are decoded verbatim (the template renders them
    without whitespace normalisation). Tool-call arguments arrive as native
    JSON (the template ``tojson``-encodes them), so types are preserved
    without the schema-driven coercion the XML formats need — ``tools`` is
    accepted for signature uniformity but unused.

    Packed response-level spans cover each tool-call segment in the
    stop-stripped stream.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls = ParsedToolCallBuilder()

    pos = 0
    n = len(ids)
    while pos < n:
        em = _find(ids, end_message_id, pos)
        terminated = em != -1
        seg_end = em if terminated else n
        seg = ids[pos:seg_end]
        # A non-first segment re-opens with <|message_model|>; drop it so the
        # content marker is at the head. (The first segment's opener lives in
        # the prompt, so it is usually already absent.)
        s = 1 if seg.size and seg[0] == message_model_id else 0
        body = seg[s:]

        if body.size == 0:
            pass
        elif body[0] == content_thinking_id:
            reasoning_parts.append(_decode(tokenizer, body[1:]))
        elif body[0] == content_text_id:
            content_parts.append(_decode(tokenizer, body[1:]))
        else:
            marker = _find_any(body, {invoke_json_id, invoke_text_id})
            if marker != -1:
                name = _decode(tokenizer, body[:marker]).strip()
                payload = _decode(tokenizer, body[marker + 1 :])
                tool_calls.append(
                    _build_inkling_tool_call(name=name, payload=payload, terminated=terminated),
                    pos,
                    seg_end + 1 if terminated else n,
                )
            else:
                # No content marker and no invoke marker — treat the decoded
                # bytes as content rather than dropping them.
                content_parts.append(_decode(tokenizer, body))

        if not terminated:
            break
        pos = em + 1

    return _parsed_response(
        content="".join(content_parts), reasoning_content="".join(reasoning_parts) or None, tool_calls=tool_calls
    )


def _build_inkling_tool_call(*, name: str, payload: str, terminated: bool) -> ParsedToolCall:
    """Build a ``ParsedToolCall`` from an Inkling ``<|content_invoke_tool_json|>``
    payload — ``{"name": …, "args": …}`` (native JSON, types preserved).

    The function name inside the JSON is authoritative; the pre-marker text
    (rendered as ``<|message_model|>{name}<|content_invoke_tool_json|>``) is a
    fallback for a truncated / malformed payload.
    """
    parsed: Any = None
    invalid_json = False
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        invalid_json = True

    fn_name: str | None = None
    arguments: dict[str, Any] | str | None = None
    if isinstance(parsed, dict):
        raw_name = parsed.get("name")
        fn_name = raw_name if isinstance(raw_name, str) and raw_name else (name or None)
        arguments = parsed.get("args", {})
    else:
        fn_name = name or None
        arguments = payload

    if not terminated:
        status = ToolCallParseStatus.UNCLOSED_BLOCK
    elif invalid_json:
        status = ToolCallParseStatus.INVALID_JSON
    elif fn_name is None:
        status = ToolCallParseStatus.MISSING_NAME
    else:
        status = ToolCallParseStatus.OK

    return ParsedToolCall(raw=payload, name=fn_name, arguments=arguments, status=status)
