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
import re
import uuid
from typing import Any

from renderers.base import ParsedResponse, ParsedToolCall, ToolCallParseStatus, ToolSpec


# ── vLLM Qwen3CoderToolParser regex patterns ────────────────────────
#
# Lifted verbatim from ``vllm/tool_parsers/qwen3coder_tool_parser.py``
# (``self.tool_call_regex`` / ``tool_call_function_regex`` /
# ``tool_call_parameter_regex``). Each pattern has two branches: one
# for the closed form (with the matching end tag) and one anchored at
# end-of-string for unclosed output. The parameter regex uses
# positive lookaheads so a missing ``</parameter>`` is recovered from
# the next ``<parameter=`` / ``</function>`` boundary instead of
# silently dropping the value.


# Mask matching vLLM's ``vllm.utils.MASK_64_BITS`` so the random id we
# generate is byte-shape compatible with vLLM's ``random_uuid()``.
_MASK_64_BITS = (1 << 64) - 1


def _make_tool_call_id() -> str:
    """Mint a tool-call id in vLLM's default format.

    vLLM's ``ToolCall`` dataclass uses ``default_factory=make_tool_call_id``
    which returns ``f"chatcmpl-tool-{random_uuid()}"`` where
    ``random_uuid`` is ``f"{uuid.uuid4().int & MASK_64_BITS:016x}"`` —
    16 hex characters drawn from the low 64 bits of a v4 UUID. We
    reproduce the format so OpenAI-shaped clients get the same id
    surface across renderers and vLLM serving paths.
    """
    return f"chatcmpl-tool-{uuid.uuid4().int & _MASK_64_BITS:016x}"


_FUNCTION_BLOCK_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
)
_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


# ── Schema-aware argument coercion ──────────────────────────────────
#
# XML-style tool-call formats render argument values verbatim inside
# ``<parameter=…>`` tags with no quoting. ``true`` and the string
# ``"true"`` produce identical wire bytes; without the tool schema, the
# parser has no signal to distinguish them. We mirror vLLM's shared
# ``extract_types_from_schema`` + ``coerce_to_schema_type`` helpers in
# ``vllm/tool_parsers/utils.py``:
#
# 1. Flatten the JSON-Schema fragment to a set of types, recursing
#    through ``anyOf`` / ``oneOf`` / ``allOf`` and inferring types from
#    ``enum`` values. A schema with no extractable type information
#    defaults to ``["string"]``.
# 2. Walk the priority-ordered ladder ``null > integer > number >
#    boolean > object > array > string``, returning the first
#    successful coercion. ``string`` is the always-succeeding terminal
#    whenever it appears in the type set, which subsumes both pure
#    string params and union types like ``Union[str, bool]``.
#
# Why ``qwen3_coder`` semantics and not ``qwen3_xml``: vLLM ships two
# parsers for the same wire format. ``qwen3_xml`` is the newer
# streaming-first parser (uses ``xml.parsers.expat`` and is the
# recommended choice for *live serving* because ``qwen3_coder`` has
# known infinite-loop bugs in vLLM's streaming path). For offline
# token-level extraction we follow ``qwen3_coder``, which itself routes
# through the shared ``coerce_to_schema_type`` helper above — so the
# two converge on scalar semantics in non-streaming use.
# TODO: when a streaming parse API lands, switch the streaming path to
# ``qwen3_xml``-style state-machine semantics to dodge vLLM's
# regex-streaming bugs (see HF Qwen3-Coder-Next discussion #17).


# Type aliases mirror ``vllm/tool_parsers/utils.py:_TYPE_ALIASES``.
_TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "long": "integer",
    "short": "integer",
    "unsigned": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array",
    "list": "array",
    "sequence": "array",
}


# Priority order mirrors ``vllm/tool_parsers/utils.py:coerce_to_schema_type``.
_TYPE_PRIORITY = (
    "null",
    "integer",
    "number",
    "boolean",
    "object",
    "array",
    "string",
)


def _build_param_type_index(
    tools: list[ToolSpec] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
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


def _extract_types_from_schema(schema: Any) -> list[str]:
    """Flatten a JSON-Schema fragment to a list of type strings.

    Byte-for-byte port of ``vllm/tool_parsers/utils.py:extract_types_from_schema``:
    handles top-level ``type`` (string or list), infers types from
    ``enum`` values, and recurses through ``anyOf`` / ``oneOf`` /
    ``allOf``. Returns ``["string"]`` when no type information can be
    determined — which makes the no-schema branch coerce to ``string``
    via the priority-ordered ladder in :func:`_coerce_arg_value`.
    """
    if schema is None or not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            types.add(type_value)
        elif isinstance(type_value, list):
            for t in type_value:
                if isinstance(t, str):
                    types.add(t)

    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for value in schema["enum"]:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for choice_field in ("anyOf", "oneOf", "allOf"):
        if choice_field in schema and isinstance(schema[choice_field], list):
            for choice in schema[choice_field]:
                types.update(_extract_types_from_schema(choice))

    return list(types) if types else ["string"]


def _coerce_arg_value(
    text: str, param_schema: dict[str, Any] | None
) -> tuple[Any, bool]:
    """Coerce a raw ``<parameter=…>`` body to its schema type.

    Mirrors vLLM ``coerce_to_schema_type`` (``vllm/tool_parsers/utils.py``):
    flatten the schema to a set of JSON-Schema types (recursing through
    ``anyOf``/``oneOf``/``allOf`` and ``enum``), then walk a priority-
    ordered ladder ``null > integer > number > boolean > object > array
    > string``, returning the first successful coercion. ``string`` is
    the always-succeeding terminal whenever it is in the type set.

    Returns ``(value, used_fallback)``. ``used_fallback=True`` iff
    every type in the schema declined the value AND the last-ditch
    ``json.loads`` of the raw text also failed — i.e. the value
    couldn't be expressed as anything the schema permits. The renderer
    propagates that flag to ``ToolCallParseStatus.INVALID_JSON`` for the
    verifier / RL-loss signal; vLLM has no such signal.
    """
    schema_types = _extract_types_from_schema(param_schema)
    normalized_types = {
        _TYPE_ALIASES.get(key, key)
        for t in schema_types
        if isinstance(t, str)
        for key in [t.strip().lower()]
    }

    for candidate in _TYPE_PRIORITY:
        if candidate not in normalized_types:
            continue
        if candidate == "null":
            if text.lower() == "null":
                return None, False
            continue
        if candidate == "string":
            return text, False
        if candidate == "integer":
            try:
                return int(text), False
            except (ValueError, TypeError):
                continue
        if candidate == "number":
            # ``int(val)`` is inside the try block so ``nan`` (raises
            # ``ValueError``) and ``inf`` (raises ``OverflowError``) skip
            # the number branch instead of aborting the whole parse.
            # vLLM's reference catches ``ValueError`` / ``TypeError`` only
            # — we widen to ``OverflowError`` so ``"inf"`` doesn't crash.
            try:
                val = float(text)
                return (val if val != int(val) else int(val)), False
            except (ValueError, TypeError, OverflowError):
                continue
        if candidate == "boolean":
            lowered = text.lower().strip()
            if lowered in ("true", "1"):
                return True, False
            if lowered in ("false", "0"):
                return False, False
            continue
        if candidate in ("object", "array"):
            try:
                return json.loads(text), False
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    try:
        return json.loads(text), False
    except (json.JSONDecodeError, ValueError):
        return text, True


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


def _reasoning_end_token_index(
    tokenizer, ids: list[int], marker: str = "</think>"
) -> int:
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
    if not ids or marker not in _decode(tokenizer, ids):
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


# ── Qwen3: <tool_call> JSON </tool_call> ────────────────────────────


def parse_qwen3(
    tokenizer,
    token_ids: list[int],
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
    tools: list[ToolSpec] | None = None,
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

    full_text = _decode(tokenizer, ids)
    tc_start = _find(ids, tool_call_id)
    param_index = _build_param_type_index(tools)
    try:
        if tc_start != -1:
            # vLLM ``qwen3coder_tool_parser.py:316-319``: content text
            # is the raw slice up to the first ``<tool_call>`` (or
            # first ``<function=`` if no ``<tool_call>`` is present).
            # No whitespace stripping — the model's prefix prose is
            # surfaced verbatim.
            content_text = _decode(tokenizer, ids[:tc_start])
            tool_calls = _parse_xml_tool_calls(
                tokenizer,
                ids[tc_start:],
                tool_call_id,
                tool_call_end_id,
                section_offset=parse_offset + tc_start,
                param_index=param_index,
            )
        else:
            # vLLM ``qwen3coder_tool_parser.py:269-271`` back-off: no
            # ``<tool_call>`` markers — scan whole output for raw
            # ``<function=…>`` blocks. Content text is whatever sits
            # before the first ``<function=`` (vLLM line 317-319), or
            # the full text if nothing matches.
            tool_calls = _parse_xml_function_blocks(
                full_text,
                param_index=param_index,
                token_span=(parse_offset, parse_offset + len(ids)),
                wrapper_unclosed=False,
            )
            if tool_calls:
                first_fn = full_text.find("<function=")
                content_text = full_text[:first_fn] if first_fn >= 0 else full_text
            else:
                content_text = full_text
    except Exception:
        # vLLM ``qwen3coder_tool_parser.py:327-331`` catch-all: on any
        # extraction error, drop every recovered call and surface the
        # raw text as content so downstream consumers never see a
        # half-parsed call.
        content_text = full_text
        tool_calls = []

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
    param_index: dict[str, dict[str, dict[str, Any]]],
) -> list[ParsedToolCall]:
    """Parse Qwen3.5-style XML tool calls from token IDs.

    Uses token IDs to demarcate ``<tool_call>`` / ``</tool_call>``
    boundaries (so ``token_span`` stays precise for trainer-side
    masking) and vLLM-parity regex on the decoded block text to
    extract the function/parameter content tolerantly. Mirrors
    ``Qwen3CoderToolParser._parse_xml_function_call`` plus the
    tag-tolerance branches of the three module-level patterns.
    """
    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] != tc_id:
            i += 1
            continue

        end = _find(ids, tc_end_id, i + 1)
        if end == -1:
            block_ids = ids[i + 1 :]
            block_text = _decode(tokenizer, block_ids)
            span = (section_offset + i, section_offset + len(ids))
            wrapper_unclosed = True
        else:
            block_text = _decode(tokenizer, ids[i + 1 : end])
            span = (section_offset + i, section_offset + end + 1)
            wrapper_unclosed = False

        block_calls = _parse_xml_function_blocks(
            block_text,
            param_index=param_index,
            token_span=span,
            wrapper_unclosed=wrapper_unclosed,
        )
        if block_calls:
            tool_calls.extend(block_calls)
        elif wrapper_unclosed:
            # vLLM would silently drop a content-less unclosed
            # ``<tool_call>``; we keep the diagnostic so verifier /
            # RL-loss code can mask the malformed span.
            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    token_span=span,
                    status=ToolCallParseStatus.UNCLOSED_BLOCK,
                )
            )
        else:
            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    token_span=span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            )

        if wrapper_unclosed:
            break
        i = end + 1
    return tool_calls


def _parse_xml_function_blocks(
    text: str,
    *,
    param_index: dict[str, dict[str, dict[str, Any]]],
    token_span: tuple[int, int] | None,
    wrapper_unclosed: bool,
) -> list[ParsedToolCall]:
    """Apply vLLM's ``<function=…></function>`` regex over *text*.

    ``token_span`` is shared by every call recovered from this block;
    the granularity is the surrounding ``<tool_call>`` region (or the
    whole completion in the no-marker back-off path), matching how
    vLLM treats multiple ``<function=>`` siblings — it does not try to
    attribute character offsets back to token positions.
    """
    tool_calls: list[ParsedToolCall] = []
    for match in _FUNCTION_BLOCK_RE.finditer(text):
        closed = match.group(1)
        body = closed if closed is not None else (match.group(2) or "")
        function_unclosed = closed is None
        end_index = body.find(">")
        if end_index == -1:
            tool_calls.append(
                ParsedToolCall(
                    raw=body,
                    token_span=token_span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            )
            continue

        name = body[:end_index]
        params_text = body[end_index + 1 :]
        params = param_index.get(name, {})
        arguments: dict = {}
        any_fallback = False
        for capture in _PARAMETER_BLOCK_RE.findall(params_text):
            idx = capture.find(">")
            if idx == -1:
                continue
            arg_name = capture[:idx]
            arg_value = _strip_one_newline(capture[idx + 1 :])
            value, used_fallback = _coerce_arg_value(arg_value, params.get(arg_name))
            arguments[arg_name] = value
            any_fallback = any_fallback or used_fallback

        # Precedence: structural failure dominates coercion failure.
        # When a block is both unclosed AND has args that couldn't be
        # coerced, surface ``UNCLOSED_BLOCK`` only — the truncation is
        # the root cause and the verifier already discounts unclosed
        # blocks. ``INVALID_JSON`` is reserved for well-formed blocks
        # whose recovered args couldn't satisfy the schema.
        if wrapper_unclosed or function_unclosed:
            status = ToolCallParseStatus.UNCLOSED_BLOCK
        elif any_fallback:
            status = ToolCallParseStatus.INVALID_JSON
        else:
            status = ToolCallParseStatus.OK
        tool_calls.append(
            ParsedToolCall(
                raw=body,
                name=name,
                arguments=arguments,
                id=_make_tool_call_id(),
                token_span=token_span,
                status=status,
            )
        )
    return tool_calls


def _strip_one_newline(s: str) -> str:
    """vLLM ``qwen3coder_tool_parser.py:246-250`` strips exactly one
    leading and one trailing newline from a parameter value — no
    further whitespace trimming."""
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


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
    tools: list[ToolSpec] | None = None,
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
            param_index=_build_param_type_index(tools),
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
    param_index: dict[str, dict[str, dict[str, Any]]],
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
                params = param_index.get(name, {})
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
                        value, used_fallback = _coerce_arg_value(
                            val_text, params.get(key)
                        )
                        arguments[key] = value
                        any_json_fallback = any_json_fallback or used_fallback
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


# ── Hy3: <think>…</think> content <tool_calls> <tool_call>name<tool_sep> …
#        <arg_key>k</arg_key> <arg_value>v</arg_value> … </tool_call> </tool_calls>
# Same arg_key/arg_value token scheme as GLM, but each call names the function
# between <tool_call> and <tool_sep>, and the calls are wrapped in an outer
# <tool_calls></tool_calls> pair. All markers are single special tokens.


def parse_hy3(
    tokenizer,
    token_ids: list[int],
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

    # ``token_span`` values are reported relative to this stop-stripped stream
    # (the documented contract), so track every prefix we slice off below.
    offset = 0

    # A round-trip slice includes the assistant role marker; the live
    # inference stream does not. Drop a single leading opener so both paths
    # land on the same downstream logic.
    if ids and ids[0] == assistant_id:
        ids = ids[1:]
        offset += 1

    reasoning = None
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = [t for t in ids[:think_end] if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        offset += think_end + 1
    elif think_id in set(ids):
        # Reasoning opened but never closed (truncation): everything after
        # the opener is reasoning; there is no committed content yet.
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    # Content ends at the first tool marker — the outer <tool_calls> wrapper
    # or, defensively, a bare <tool_call> the model emitted without it.
    marker_positions = [
        p for p in (_find(ids, tool_calls_id), _find(ids, tool_call_id)) if p != -1
    ]
    tool_calls: list[ParsedToolCall] = []
    if marker_positions:
        tool_start = min(marker_positions)
        content_text = _decode(tokenizer, ids[:tool_start]).strip()
        tool_calls = _parse_hy3_tool_calls(
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
        )
    else:
        content_text = _decode(tokenizer, ids).strip()

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


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
) -> list[ParsedToolCall]:
    """Parse Hy3-style tool calls: ``<tool_call>name<tool_sep>`` then
    arg_key/arg_value pairs, all by token ID. The outer ``<tool_calls>``
    wrapper and inter-block newlines are skipped by scanning for the
    per-call ``<tool_call>`` opener. ``section_offset`` shifts recorded
    ``token_span`` values back into the stop-stripped completion stream
    (``ids`` here is a suffix of it)."""
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
                    arg_ids = []
                else:
                    name = _decode(tokenizer, block[:first_ak]).strip()
                    arg_ids = block[first_ak:]

            params = param_index.get(name, {})
            arguments: dict = {}
            structure_broke = False
            any_json_fallback = False
            j = 0
            while j < len(arg_ids):
                if arg_ids[j] == ak_id:
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


# ── Laguna-XS.2: <tool_call> name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value> </tool_call>
# Same outer skeleton as parse_glm, but <arg_key>/<arg_value> are plain text
# (multi-token BPE), not single special tokens — so the inner block is decoded
# to text and the key/value pairs are pulled out by regex.


def parse_laguna_xs2(
    tokenizer,
    token_ids: list[int],
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

    def _segment(segment_ids: list[int]) -> str:
        text = _decode(tokenizer, segment_ids)
        return text.strip("\n") if strip_newlines else text

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _segment(reasoning_ids)
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif (think_start := _find(ids, think_id)) != -1:
        reasoning = _segment(ids[think_start + 1 :])
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_text = _segment(ids[:tc_start])
        tool_calls = _parse_laguna_xs2_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            section_offset=parse_offset + tc_start,
            param_index=_build_param_type_index(tools),
        )
    else:
        content_text = _segment(ids)

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_laguna_xs2_tool_calls(
    tokenizer,
    ids: list[int],
    tc_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
    param_index: dict[str, dict[str, dict[str, Any]]],
) -> list[ParsedToolCall]:
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

    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_id:
            tc_end = _find(ids, tc_end_id, i + 1)
            if tc_end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            block_text = _decode(tokenizer, ids[i + 1 : tc_end])
            span = (section_offset + i, section_offset + tc_end + 1)

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
            for m in re.finditer(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                args_section,
                re.DOTALL,
            ):
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
                ParsedToolCall(
                    raw=block_text,
                    name=name or None,
                    arguments=arguments,
                    token_span=span,
                    status=status,
                )
            )
            i = tc_end + 1
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

    # Reasoning first: skip past </think> before looking for the tool-call
    # section, so a section the model drafts *inside* its <think> trace isn't
    # parsed as a real call (regression #78 — cf. parse_qwen3). content_ids
    # still starts at 0, so the </think> text-split below recovers reasoning.
    # DeepSeek-V3 renders </think> as multi-token text, hence the decode-based
    # boundary finder rather than a token-id anchor.
    reasoning_end = _reasoning_end_token_index(tokenizer, ids)

    tc_section_start = _find(ids, tool_calls_begin_id, reasoning_end)
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
                        params = param_index.get(name, {})
                        arguments: dict = {}
                        any_json_fallback = False
                        for pm in re.finditer(
                            r'<parameter name="([^"]+)">(.*?)</parameter>',
                            body,
                            re.DOTALL,
                        ):
                            pname = pm.group(1)
                            pval = pm.group(2).strip()
                            value, used_fallback = _coerce_arg_value(
                                pval, params.get(pname)
                            )
                            arguments[pname] = value
                            any_json_fallback = any_json_fallback or used_fallback
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
    scan_start: int = 0,
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

    ``scan_start`` restricts the section search to ``ids[scan_start:]`` while
    keeping ``content_ids = ids[:section_start]`` and all token spans relative
    to the full ``ids``. Callers pass the post-``</think>`` index so a section
    the model drafts inside its reasoning trace isn't parsed as a real call;
    because ``content_ids`` still starts at 0, downstream text-based reasoning
    extraction is unaffected (regression #78).
    """
    section_start = _find_any(ids, tool_calls_section_begin_ids, scan_start)
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


# ── Llama-3: single JSON tool call {"name": "...", "parameters": {...}} ─


def parse_llama_3(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
) -> ParsedResponse:
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
        if eh_positions:
            body_start = eh_positions[-1] + 1
    body_ids = ids[body_start:]
    text = _decode(tokenizer, body_ids).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("name"):
            arguments = parsed.get("parameters", parsed.get("arguments", {}))
            return ParsedResponse(
                content="",
                reasoning_content=None,
                tool_calls=[
                    ParsedToolCall(
                        raw=text,
                        name=parsed["name"],
                        arguments=arguments,
                        token_span=(body_start, len(ids)),
                        status=ToolCallParseStatus.OK,
                    )
                ],
            )

    # Not a tool-call shape (plain reply, or a ``{...}`` body that didn't
    # parse / lacked a name). Llama-3 has no delimiter to anchor a
    # "malformed attempt" against, so it falls through to content rather
    # than producing a non-OK ParsedToolCall.
    return ParsedResponse(content=text, reasoning_content=None)
