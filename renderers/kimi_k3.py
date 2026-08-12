"""Kimi-K3 Renderer — hard-coded Python mirroring Moonshot's K3 chat encoder.

K3 does not ship a Jinja chat template. Its prompt format is produced by Python in
the model repo (``encoding_k3.py``), so the strings below mirror that encoder rather
than a template file.

The format is a tag envelope Moonshot calls XTML rather than the ``<|im_*|>`` turn
markers of Kimi K2/K2.5, so this renderer shares nothing structural with
``kimi_k25.py`` beyond the media tokens:

    <|open|>message role="user"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>

Assistant turns carry two channels inside one message, which is what stops an
opaque-template renderer from bridging turns for this model — extending a
trajectory requires knowing where ``think`` closes:

    <|open|>message role="assistant"<|sep|>
        <|open|>think<|sep|>...<|close|>think<|sep|>
        <|open|>response<|sep|>...<|close|>response<|sep|>
    <|close|>message<|sep|><|end_of_msg|>

A generation prompt opens the assistant message and the think channel and stops,
leaving the model to produce reasoning first.

Two synthetic system messages precede the conversation, in this order: a
``tool-declare`` block carrying JSON-Schema when tools are supplied, then a
``thinking-effort`` block whenever thinking is enabled. The effort preamble
advertises ``medium`` while the encoder only accepts ``low``/``high``/``max``;
that inconsistency is upstream's and is reproduced verbatim.

Images render as ``<|media_begin|>image {w}x{h}<|media_content|><|media_pad|><|media_end|>``
where ``{w}x{h}`` is the source pixel size — K3 embeds the resolution in the block,
unlike K2.5, and it is the original size rather than the patch grid or any resized size. Exactly one ``<|media_pad|>`` lands in the stream
because the model expands per-patch attention itself from ``grid_thws``, so
``mm_placeholders.length`` is 1 per image. K2.5's trailing newline is absent here.

``apply_chat_template`` emits a bare ``<|kimi_image_placeholder|>`` and substitutes the
per-image block later from processor output, so image renders here diverge from that
call by construction — the renderer emits what the model is actually served.
"""

from __future__ import annotations

import json
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    MultiModalData,
    ParsedResponse,
    ParsedToolCall,
    PlaceholderRange,
    RenderedTokens,
    ToolSpec,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
    ToolCallParseStatus,
)
from renderers.configs import KimiK3RendererConfig
from renderers.kimi_k25 import _image_hash, _is_image_part, _load_pil_image

# ---------------------------------------------------------------------------
# Constants — must match encoding_k3.py's literal strings exactly.
# ---------------------------------------------------------------------------
OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

MEDIA_BEGIN = "<|media_begin|>"
MEDIA_CONTENT = "<|media_content|>"
MEDIA_PAD = "<|media_pad|>"
MEDIA_END = "<|media_end|>"

VALID_THINKING_EFFORTS = ("low", "high", "max")

# Upstream advertises "medium" in the prose while rejecting it as a value.
_THINKING_EFFORT_BODY = (
    "`thinking_effort` guides on how much to think in your "
    "thinking channel (not including the response channel), "
    "supported values include `low`, `medium`, `high`, and `max`.\n"
    "Now the system is invoked with `thinking_effort={effort}`."
)

_TOOL_DECLARE_BODY = "# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n{tools}\n```"

_THINK_CHANNEL = "think"
_RESPONSE_CHANNEL = "response"
_MESSAGE_TAG = "message"
_TOOLS_CHANNEL = "tools"
_CALL_TAG = "call"
_ARGUMENT_TAG = "argument"
_JSON_TAG = "json"

# A render op is either literal text or an image part awaiting the processor.
_TEXT = "text"
_IMAGE = "image"
Op = tuple[str, Any]


def _json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def _escape_attr_value(value: Any) -> str:
    return str(value).replace('"', '\\"')


def _xtml_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    return "array"


def _xtml_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _xtml_parse(value: str, declared_type: str) -> Any:
    """Inverse of ``_xtml_value``; an undeclared or malformed value stays a string."""
    if declared_type == "string":
        return value
    if declared_type == "null":
        return None
    if declared_type == "boolean":
        return value.strip() == "true"
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _attr(header: str, key: str) -> str:
    marker = f'{key}="'
    if marker not in header:
        return ""
    return header.split(marker, 1)[1].split('"', 1)[0]


def _open_tag(tag: str, attrs: tuple[tuple[str, Any], ...] = ()) -> str:
    rendered = f"{OPEN_TOKEN}{tag}"
    for key, value in attrs:
        rendered += f' {key}="{_escape_attr_value(value)}"'
    return rendered + SEP_TOKEN


def _close_tag(tag: str) -> str:
    return f"{CLOSE_TOKEN}{tag}{SEP_TOKEN}"


class KimiK3Renderer:
    """Deterministic message → token renderer for Kimi K3 models."""

    _config_cls = KimiK3RendererConfig

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: KimiK3RendererConfig | None = None,
        *,
        processor: Any = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or KimiK3RendererConfig()
        self._processor = processor
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )
        self._image_cache: dict[str, tuple[Any, tuple[int, int]]] = {}
        self._media_pad = self._token_id(MEDIA_PAD)
        self._end_of_msg = self._token_id(END_OF_MSG_TOKEN)
        self._open_id = self._token_id(OPEN_TOKEN)
        self._close_id = self._token_id(CLOSE_TOKEN)
        self._sep_id = self._token_id(SEP_TOKEN)

    # ------------------------------------------------------------------ ids

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"tokenizer has no id for special token {token!r}")
        return token_id

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def mm_token_type_id_map(self) -> dict[int, int]:
        """Token id → modality marker. Only ``<|media_pad|>`` carries an image."""
        return {self._media_pad: 1}

    # ------------------------------------------------------------------ images

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        from transformers import AutoProcessor

        name = getattr(self.tokenizer, "name_or_path", None)
        if not name:
            raise RuntimeError(
                "KimiK3Renderer needs a processor to render image content. Pass "
                "`processor=AutoProcessor.from_pretrained(name, trust_remote_code=True, "
                "revision=<pinned sha>)`, or load the tokenizer with a name_or_path so the "
                "processor can be auto-loaded."
            )
        # K3's processor is custom Python in the model repo, so it needs trust_remote_code.
        self._processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        return self._processor

    def _process_image(self, part: dict[str, Any]):
        """-> (processor_out, (width, height), image_hash), memoised by content.

        The returned size is the *source* pixel size, which is what the block carries —
        not the patch grid in ``grid_thws`` and not any resized size. Verified against
        the processor for square, wide, tall and odd inputs.
        """
        pil = _load_pil_image(part)
        digest = _image_hash(pil)
        cached = self._image_cache.get(digest)
        if cached is not None:
            out, size = cached
            return out, size, digest
        image_processor = self._get_processor().image_processor
        out = image_processor.preprocess(
            [{"type": "image", "image": pil}], return_tensors="np"
        )
        size = (int(pil.width), int(pil.height))
        if len(self._image_cache) >= self.config.image_cache_max:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[digest] = (out, size)
        return out, size, digest

    @staticmethod
    def _media_prefix(width: int, height: int) -> str:
        """Everything before the pad in ``KimiK3VisionProcessor.make_image_prompt``."""
        return f"{MEDIA_BEGIN}image {width}x{height}{MEDIA_CONTENT}"

    # ----------------------------------------------------------------- ops

    def _content_ops(self, content: Any) -> list[Op]:
        if content is None:
            return []
        if isinstance(content, str):
            return [(_TEXT, content)]
        ops: list[Op] = []
        for part in content:
            if not isinstance(part, dict):
                ops.append((_TEXT, str(part)))
            elif _is_image_part(part):
                ops.append((_IMAGE, part))
            elif part.get("type") == "text":
                ops.append((_TEXT, part.get("text", "")))
            elif part.get("type") == "tool_result":
                ops.extend(self._content_ops(part.get("content")))
        return ops

    def _system_block_ops(self, message_type: str, body: str) -> list[Op]:
        return [
            (
                _TEXT,
                _open_tag(_MESSAGE_TAG, (("role", "system"), ("type", message_type)))
                + body.strip()
                + _close_tag(_MESSAGE_TAG)
                + END_OF_MSG_TOKEN,
            )
        ]

    def _preamble_ops(self, tools: list[ToolSpec] | None) -> list[Op]:
        ops: list[Op] = []
        if tools:
            ops.extend(
                self._system_block_ops(
                    "tool-declare",
                    _TOOL_DECLARE_BODY.format(tools=_json_compact(tools)),
                )
            )
        if self.config.thinking:
            effort = self.config.thinking_effort
            if effort not in VALID_THINKING_EFFORTS:
                raise ValueError(
                    f"Unsupported thinking_effort={effort!r}; "
                    f"supported values are {sorted(VALID_THINKING_EFFORTS)}."
                )
            ops.extend(
                self._system_block_ops(
                    "thinking-effort", _THINKING_EFFORT_BODY.format(effort=effort)
                )
            )
        return ops

    def _assistant_ops(self, message: Message) -> list[Op]:
        reasoning = message.get("reasoning_content") or ""
        if self.effective_thinking_retention == "none":
            reasoning = ""
        ops: list[Op] = [
            (_TEXT, _open_tag(_THINK_CHANNEL) + reasoning + _close_tag(_THINK_CHANNEL)),
            (_TEXT, _open_tag(_RESPONSE_CHANNEL)),
        ]
        ops.extend(self._content_ops(message.get("content")))
        ops.append((_TEXT, _close_tag(_RESPONSE_CHANNEL)))
        calls = message.get("tool_calls") or []
        if calls:
            rendered = _open_tag(_TOOLS_CHANNEL)
            for index, call in enumerate(calls, start=1):
                function = call.get("function", call)
                rendered += _open_tag(
                    _CALL_TAG, (("tool", function.get("name", "")), ("index", index))
                )
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (ValueError, TypeError):
                        pass
                if isinstance(arguments, dict):
                    for key, value in arguments.items():
                        rendered += (
                            _open_tag(
                                _ARGUMENT_TAG, (("key", key), ("type", _xtml_type(value)))
                            )
                            + _xtml_value(value)
                            + _close_tag(_ARGUMENT_TAG)
                        )
                elif arguments is not None:
                    rendered += (
                        _open_tag(_JSON_TAG, (("type", "object"),))
                        + _xtml_value(arguments)
                        + _close_tag(_JSON_TAG)
                    )
                rendered += _close_tag(_CALL_TAG)
            ops.append((_TEXT, rendered + _close_tag(_TOOLS_CHANNEL)))
        return ops

    def _message_ops(
        self,
        message: Message,
        tool_index: int = 0,
        pending_calls: list[dict[str, Any]] | None = None,
    ) -> list[Op]:
        role = message.get("role", "user")
        attrs: tuple[tuple[str, Any], ...] = (("role", role),)
        if role == "tool":
            name = message.get("tool") or message.get("name")
            if name is None and pending_calls and tool_index <= len(pending_calls):
                # OpenAI-shaped results carry only a tool_call_id, so fall back to the
                # position within the assistant turn, as K3's own encoder does.
                call = pending_calls[tool_index - 1]
                name = (call.get("function", call) or {}).get("name")
            if name is None:
                raise ValueError(
                    "Kimi K3 tool messages need a resolvable tool name: carry `tool` or "
                    "`name`, or match a preceding assistant tool_call by order"
                )
            # The result is bound to its call by position within the assistant turn, not by id.
            attrs = (("role", role), ("tool", name), ("index", tool_index))
        elif role in ("user", "system") and message.get("name"):
            attrs = (("role", role), ("name", message["name"]))
        ops: list[Op] = [(_TEXT, _open_tag(_MESSAGE_TAG, attrs))]
        if role == "assistant":
            ops.extend(self._assistant_ops(message))
        else:
            ops.extend(self._content_ops(message.get("content")))
        ops.append((_TEXT, _close_tag(_MESSAGE_TAG) + END_OF_MSG_TOKEN))
        return ops

    @staticmethod
    def _generation_prompt_ops() -> list[Op]:
        return [
            (
                _TEXT,
                _open_tag(_MESSAGE_TAG, (("role", "assistant"),))
                + _open_tag(_THINK_CHANNEL),
            )
        ]

    # --------------------------------------------------------------- emitting

    def _emit(
        self, plan: list[tuple[list[Op], int]]
    ) -> tuple[list[int], list[int], list[bool], MultiModalData]:
        token_ids: list[int] = []
        message_indices: list[int] = []
        is_content: list[bool] = []
        mm_hashes: dict[str, list[str]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        mm_items: dict[str, list[dict[str, Any]]] = {}

        def emit(text: str, message_index: int, *, content: bool) -> list[int]:
            ids = self._encode(text)
            token_ids.extend(ids)
            message_indices.extend([message_index] * len(ids))
            is_content.extend([content] * len(ids))
            return ids

        for ops, message_index in plan:
            for kind, payload in ops:
                if kind == _TEXT:
                    emit(payload, message_index, content=False)
                    continue
                out, (width, height), digest = self._process_image(payload)
                emit(self._media_prefix(width, height), message_index, content=False)
                offset = len(token_ids)
                pad_ids = emit(MEDIA_PAD, message_index, content=True)
                emit(MEDIA_END, message_index, content=False)
                mm_hashes.setdefault(_IMAGE, []).append(digest)
                mm_placeholders.setdefault(_IMAGE, []).append(
                    PlaceholderRange(offset=offset, length=len(pad_ids))
                )
                # Ship under Kimi's native ``grid_thws`` key so a generic packer can
                # route it straight into the model's forward kwargs.
                mm_items.setdefault(_IMAGE, []).append(
                    {"pixel_values": out["pixel_values"], "grid_thws": out["grid_thws"]}
                )

        return (
            token_ids,
            message_indices,
            is_content,
            MultiModalData(
                mm_hashes=mm_hashes, mm_placeholders=mm_placeholders, mm_items=mm_items
            ),
        )

    # --------------------------------------------------------------- protocol

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        plan: list[tuple[list[Op], int]] = []
        preamble = self._preamble_ops(tools)
        if preamble:
            plan.append((preamble, -1))
        tool_index = 0
        pending_calls: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            role = message.get("role", "user")
            if role == "tool":
                tool_index += 1
            elif role == "assistant":
                tool_index = 0
                pending_calls = message.get("tool_calls") or []
            plan.append(
                (self._message_ops(message, tool_index, pending_calls), index)
            )
        if add_generation_prompt:
            plan.append((self._generation_prompt_ops(), -1))

        token_ids, message_indices, is_content, mm_data = self._emit(plan)
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            sampled_mask=[False] * len(token_ids),
            is_content=is_content,
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
            multi_modal_data=mm_data,
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

    def get_stop_token_ids(self) -> list[int]:
        return [self._end_of_msg]

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        """Read the channels back off the ids.

        Decoded text is not a safe surface to match on: whether a revision renders
        ``<|sep|>`` with padding spaces is its own choice, and a mismatch there silently
        yields an empty parse rather than an error.
        """
        nodes = self._structure(token_ids)
        reasoning = self._channel_text(nodes, _THINK_CHANNEL)
        if reasoning is None and nodes and nodes[0].get("tag") is None:
            # The generation prompt already opened the think channel, so a sampled
            # completion starts inside it and its first run has no opening tag.
            reasoning = nodes[0]["text"]
        content = self._channel_text(nodes, _RESPONSE_CHANNEL)
        declared = {
            (tool.get("function", tool) or {}).get("name") for tool in tools or []
        }
        return ParsedResponse(
            content=content or "",
            reasoning_content=reasoning or None,
            tool_calls=self._collect_tool_calls(nodes, declared),
        )

    def _structure(self, token_ids: list[int]) -> list[dict[str, Any]]:
        """Ids -> a shallow tree of ``{tag, attrs, text, children}`` nodes.

        A tag is ``<|open|> header <|sep|>`` and ends at ``<|close|> tag <|sep|>``; anything
        else is body text belonging to the innermost open tag.
        """
        root: dict[str, Any] = {"tag": None, "attrs": "", "text": "", "children": []}
        stack = [root]
        pending: list[int] = []
        i = 0
        specials = {self._open_id, self._close_id, self._sep_id, self._end_of_msg}

        def flush() -> None:
            if pending:
                stack[-1]["text"] += self.tokenizer.decode(
                    pending, skip_special_tokens=False
                )
                pending.clear()

        while i < len(token_ids):
            tid = token_ids[i]
            if tid == self._open_id or tid == self._close_id:
                flush()
                header_ids = []
                i += 1
                while i < len(token_ids) and token_ids[i] not in specials:
                    header_ids.append(token_ids[i])
                    i += 1
                if i < len(token_ids) and token_ids[i] == self._sep_id:
                    i += 1
                header = self.tokenizer.decode(
                    header_ids, skip_special_tokens=False
                ).strip()
                tag = header.split(" ", 1)[0]
                if tid == self._open_id:
                    node = {
                        "tag": tag,
                        "attrs": header[len(tag) :].strip(),
                        "text": "",
                        "children": [],
                    }
                    stack[-1]["children"].append(node)
                    stack.append(node)
                elif len(stack) > 1:
                    # An unmatched close would otherwise pop the root and lose everything.
                    for depth in range(len(stack) - 1, 0, -1):
                        if stack[depth]["tag"] == tag:
                            del stack[depth:]
                            break
                    else:
                        stack.pop()
                continue
            if tid in specials:
                flush()
                i += 1
                continue
            pending.append(tid)
            i += 1
        flush()
        return [root, *root["children"]] if root["text"] else root["children"]

    @staticmethod
    def _walk(nodes: list[dict[str, Any]]):
        """Every node, depth first. A sampled completion carries its channels at the top
        level; a re-rendered history turn nests them under its own ``message`` tag."""
        for node in nodes:
            yield node
            yield from KimiK3Renderer._walk(node.get("children") or [])

    def _channel_text(
        self, nodes: list[dict[str, Any]], channel: str
    ) -> str | None:
        for node in self._walk(nodes):
            if node.get("tag") == channel:
                return node["text"]
        return None

    def _collect_tool_calls(
        self, nodes: list[dict[str, Any]], declared: set[str | None]
    ) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for node in self._walk(nodes):
            if node.get("tag") != _TOOLS_CHANNEL:
                continue
            for call in node["children"]:
                if call.get("tag") != _CALL_TAG:
                    continue
                name = _attr(call["attrs"], "tool")
                arguments: Any = {}
                for child in call["children"]:
                    if child["tag"] == _JSON_TAG:
                        try:
                            arguments = json.loads(child["text"])
                        except (ValueError, TypeError):
                            arguments = child["text"]
                    elif child["tag"] == _ARGUMENT_TAG:
                        arguments[_attr(child["attrs"], "key")] = _xtml_parse(
                            child["text"], _attr(child["attrs"], "type")
                        )
                if not name:
                    status = ToolCallParseStatus.MISSING_NAME
                elif isinstance(arguments, str):
                    status = ToolCallParseStatus.INVALID_JSON
                elif declared and name not in declared:
                    status = ToolCallParseStatus.UNKNOWN_TOOL
                else:
                    status = ToolCallParseStatus.OK
                calls.append(
                    ParsedToolCall(
                        raw=f'{_CALL_TAG} {call["attrs"]}',
                        name=name or None,
                        arguments=arguments,
                        status=status,
                        id=_attr(call["attrs"], "index") or None,
                    )
                )
        return calls

    @staticmethod
    def _between(text: str, start: str, end: str) -> str | None:
        if start not in text:
            return None
        rest = text.split(start, 1)[1]
        return rest.split(end, 1)[0] if end in rest else rest

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        previous_multi_modal_data: MultiModalData | None = None,
    ) -> RenderedTokens | None:
        if (
            not previous_prompt_ids
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
            {self._end_of_msg},
            synthesize_close=self._end_of_msg,
        )
        if previous_ids is None:
            return None

        plan: list[tuple[list[Op], int]] = [
            (self._message_ops(message), -1) for message in new_messages
        ]
        plan.append((self._generation_prompt_ops(), -1))
        ext_ids, _, ext_content, mm_data = self._emit(plan)

        # Placeholder offsets are relative to the extension; shift into the full stream.
        offset = len(previous_ids)
        for ranges in mm_data.mm_placeholders.values():
            ranges[:] = [
                PlaceholderRange(offset=r.offset + offset, length=r.length)
                for r in ranges
            ]

        # Carry earlier-turn images forward. Copy the per-modality lists so extending
        # them never mutates the caller's data.
        merged = MultiModalData(
            mm_hashes={
                k: list(v)
                for k, v in (
                    previous_multi_modal_data.mm_hashes
                    if previous_multi_modal_data
                    else {}
                ).items()
            },
            mm_placeholders={
                k: list(v)
                for k, v in (
                    previous_multi_modal_data.mm_placeholders
                    if previous_multi_modal_data
                    else {}
                ).items()
            },
            mm_items={
                k: list(v)
                for k, v in (
                    previous_multi_modal_data.mm_items
                    if previous_multi_modal_data
                    else {}
                ).items()
            },
        )
        for modality, values in mm_data.mm_hashes.items():
            merged.mm_hashes.setdefault(modality, []).extend(values)
        for modality, values in mm_data.mm_placeholders.items():
            merged.mm_placeholders.setdefault(modality, []).extend(values)
        for modality, values in mm_data.mm_items.items():
            merged.mm_items.setdefault(modality, []).extend(values)

        token_ids = [*previous_ids, *ext_ids]
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=[-1] * len(token_ids),
            sampled_mask=[False] * len(token_ids),
            is_content=[False] * offset + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
            multi_modal_data=merged,
        )
