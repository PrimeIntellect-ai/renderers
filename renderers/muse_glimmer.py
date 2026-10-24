"""Muse Glimmer renderer — hard-coded Python mirroring the ATEM chat template
pinned by ``meta-models/Muse-Glimmer-30B``.

Format (every tag is a single special token; recipients are plain text):

    <|begin_of_text|>
    <|start|>system<|message|>…system preamble…<|eot|>
    <|start|>user<|message|>…<|eot|>
    <|start|>assistant to=self<|message|>…reasoning…<|eom|>
    <|start|>assistant to=tool.fn<|message|><atem:function_calls>…</atem:function_calls><|eot|>
    <|start|>tool tool.fn<|message|><tool_output name="tool.fn">\\n…\\n</tool_output><|eot|>
    <|start|>assistant to=user<|message|>…answer…<|eot|>

- The generation prompt is ``<|start|>assistant``; the model samples the
  recipient (`` to=self`` / `` to=<tool>`` / `` to=user``) itself.
- ``<|eom|>`` ends a message but not the turn, so one sampled turn can hold
  reasoning, tool calls and the answer. Only ``<|eot|>`` and
  ``<|end_of_text|>`` stop generation.
- Historical reasoning is always re-rendered (the template never drops it).
- Images render as one ``<|patch|>`` placeholder per image (the logical
  prompt); the inference engine expands it from the raw image. Video and
  audio are not supported yet.
- Without a system message the template injects a default preamble with the
  knowledge cutoff and the current date. ``current_date=None`` uses today's
  date, fixed when the renderer is constructed; pin it for reproducible
  training renders.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from renderers.base import (
    Message,
    MultiModalData,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    Tokenizer,
    ToolCallParseStatus,
    ToolSpec,
    _get_offset_tokenizer,
    _infer_offsets_from_decode,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
)
from renderers.configs import MuseGlimmerRendererConfig

_IMAGE_TYPES = ("image", "image_url")
_UNSUPPORTED_MEDIA_TYPES = ("video", "video_url", "audio", "audio_url", "input_audio")

_DEFAULT_SYSTEM = "You are a helpful AI assistant."

_TOOLS_INTRO = (
    "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
    'You can invoke a function by writing a "<atem:function_calls>" block like the following:\n'
    "<atem:function_calls>\n"
    '<atem:invoke name="$FUNCTION_NAME">\n'
    '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n'
    "...\n"
    "</atem:invoke>\n"
    "</atem:function_calls>\n\n"
    "String and scalar parameters should be specified as is, while lists and objects should use JSON format. "
    "Note that spaces for string values are not stripped. The output is not expected to be valid XML and is "
    "parsed with regular expressions.\n"
    "Here are the functions available in JSONSchema format:\n"
    "// Tool metadata\n"
)

_TOOLS_EXAMPLE = (
    "\n\nHere's an example of how to call a function in the tool set:\n"
    "(If the tool namespace is not specified, invoke the function directly as `example_function_name` "
    "rather than `example_tool_name.example_function_name`)\n\n"
    "to=example_tool_name.example_function_name\n\n"
    "<atem:function_calls>\n"
    '<atem:invoke name="example_tool_name.example_function_name">\n'
    '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
    '<atem:parameter name="example_parameter_2">This is the value for the second parameter\n'
    'that can span\n"multiple" lines\n</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)

# A message header inside a sampled turn. The first message of a turn follows
# the ``<|start|>assistant`` generation prompt, so it appears as a bare
# `` to=<recipient><|message|>``; later ones repeat ``<|start|>assistant``.
_HEADER_RE = re.compile(
    r"(?P<start><\|start\|>[^\S\n]*(?:assistant)?)?(?P<ws>[^\S\n]*)"
    r"(?:to=(?P<rcpt>[A-Za-z0-9_.\-:/]+))?[^\S\n]*<\|message\|>"
)
_INVOKE_OPEN_RE = re.compile(r"<atem:invoke\b")
_INVOKE_RE = re.compile(
    r"<atem:invoke\b(?P<attrs>[^>]*)>(?P<body>.*?)</atem:invoke>", re.DOTALL
)
_INVOKE_NAME_RE = re.compile(r'\bname="(?P<name>[^"]*)"')
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)
_STRIP_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?to=self<\|message\|>.*?<\|eom\|>", re.DOTALL
)
_STRIP_OPEN_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?to=self<\|message\|>"
    r"(?:(?!<\|eom\|>)(?!to=[^\s<]+<\|message\|>).)*"
    r"(?=to=[^\s<]+<\|message\|>|$)",
    re.DOTALL,
)


def _json(value: Any) -> str:
    """HF's ``tojson`` filter: ``json.dumps(..., ensure_ascii=False)``."""
    return json.dumps(value, ensure_ascii=False)


def _function(tool: Any) -> Mapping[str, Any]:
    fn = tool.get("function") if isinstance(tool, Mapping) else None
    return fn if isinstance(fn, Mapping) else tool


def _namespace(name: str) -> str:
    return name.split(".")[0]


def _arguments(tool_call: Mapping[str, Any]) -> Mapping[str, Any]:
    fn = tool_call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        # The template requires a mapping; accept OpenAI-style JSON strings.
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Muse Glimmer tool_call.function.arguments must be a JSON object"
            ) from exc
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise ValueError(
            "Muse Glimmer tool_call.function.arguments must be a dict (mapping)"
        )
    return args


def _atem_value(value: Any) -> str:
    """One ``<atem:parameter>`` body, as the template prints it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (Mapping, list, tuple)):
        return _json(value)
    return str(value)


def _atem(tool_call: Mapping[str, Any]) -> str:
    name = (tool_call.get("function") or {}).get("name", "")
    out = f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
    for key, value in _arguments(tool_call).items():
        out += f'<atem:parameter name="{key}">{_atem_value(value)}</atem:parameter>\n'
    return out + "</atem:invoke>\n</atem:function_calls>"


class _TokenStream:
    """Accumulates tokens with per-token attribution.

    Text between special tokens is collected as segments and encoded in one
    BPE pass when the next special token (or the end) arrives, exactly like
    the HF tokenizer splitting the rendered template at its special tokens.
    Each token is attributed to the segment holding its first character.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self._offsets = _get_offset_tokenizer(tokenizer)
        self.tokens: list[int] = []
        self.indices: list[int] = []
        self.sampled: list[bool] = []
        self.content: list[bool] = []
        self._pending: list[tuple[str, int, bool, bool]] = []

    def special(
        self, token_id: int, msg_idx: int, *, sampled: bool, content: bool
    ) -> None:
        self.flush()
        self.tokens.append(token_id)
        self.indices.append(msg_idx)
        self.sampled.append(sampled)
        self.content.append(content)

    def text(self, text: str, msg_idx: int, *, sampled: bool, content: bool) -> None:
        if text:
            self._pending.append((text, msg_idx, sampled, content))

    def flush(self) -> None:
        segments, self._pending = self._pending, []
        if not segments:
            return
        full = "".join(s[0] for s in segments)
        if self._offsets is not None:
            enc = self._offsets(
                full, add_special_tokens=False, return_offsets_mapping=True
            )
            ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
        else:
            ids = self._tokenizer.encode(full, add_special_tokens=False)
            offsets = _infer_offsets_from_decode(self._tokenizer, ids, full)
        bounds, pos = [], 0
        for seg in segments:
            bounds.append((pos, pos + len(seg[0])))
            pos += len(seg[0])
        for k, tok in enumerate(ids):
            if offsets is not None:
                start = offsets[k][0]
                owner = next(
                    (j for j, (a, b) in enumerate(bounds) if a <= start < b),
                    len(segments) - 1,
                )
            else:
                # No character offsets: keep ids exact, attribute to the last segment.
                owner = len(segments) - 1
            _, msg_idx, sampled, content = segments[owner]
            self.tokens.append(tok)
            self.indices.append(msg_idx)
            self.sampled.append(sampled)
            self.content.append(content)

    def rendered(self, messages: list[Message], *, prefix: int = 0) -> RenderedTokens:
        self.flush()
        content = self.content if self._offsets is not None else []
        if content and prefix:
            content = [False] * prefix + content
        return RenderedTokens(
            token_ids=self.tokens,
            message_indices=self.indices,
            sampled_mask=self.sampled,
            is_content=content,
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
        )


class MuseGlimmerRenderer:
    """Deterministic message → token renderer for Muse Glimmer (ATEM)."""

    _config_cls = MuseGlimmerRendererConfig
    supports_process_multimodal = True

    @property
    def is_prefix_stable(self) -> bool:
        """Whether full renders preserve completed conversation prefixes."""
        # A tool message without ``name`` takes the function name of the last
        # assistant tool call with its ``tool_call_id`` anywhere in the
        # conversation, so a later reused id renames an earlier tool result.
        return False

    def __init__(
        self, tokenizer: Tokenizer, config: MuseGlimmerRendererConfig | None = None
    ):
        self._tokenizer = tokenizer
        self.config = config or MuseGlimmerRendererConfig()
        # The template re-renders every assistant's reasoning.
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )
        # ``current_date=None`` follows the template's ``strftime_now`` fallback;
        # resolve it once so renders from one instance agree.
        self._current_date = self.config.current_date or datetime.now().strftime(
            "%Y-%m-%d"
        )

        self._bos = self._token_id("<|begin_of_text|>")
        self._end_of_text = self._token_id("<|end_of_text|>")
        self._start = self._token_id("<|start|>")
        self._message = self._token_id("<|message|>")
        self._eot = self._token_id("<|eot|>")
        self._eom = self._token_id("<|eom|>")
        self._patch = self._token_id("<|patch|>")

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        return {self._patch: 1}

    # ── media ───────────────────────────────────────────────────────────

    @staticmethod
    def _validate_media(messages: list[Message], *, process_multimodal: bool) -> None:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            role = message.get("role")
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind in _UNSUPPORTED_MEDIA_TYPES:
                    raise NotImplementedError(
                        f"MuseGlimmerRenderer does not support {kind!r} content"
                    )
                if kind in _IMAGE_TYPES:
                    if role not in ("system", "user", "tool"):
                        raise NotImplementedError(
                            f"MuseGlimmerRenderer does not support images in {role!r} messages"
                        )
                    if process_multimodal:
                        raise NotImplementedError(
                            "MuseGlimmerRenderer renders images only with "
                            "process_multimodal=False (one placeholder per image)"
                        )

    def _emit_body(
        self, stream: _TokenStream, content: Any, msg_idx: int, *, sampled: bool
    ) -> None:
        """``render_content``: text, and one ``<|patch|>`` per image."""
        if content is None:
            return
        if isinstance(content, str):
            stream.text(content, msg_idx, sampled=sampled, content=True)
            return
        for item in content:
            if isinstance(item, str):
                stream.text(item, msg_idx, sampled=sampled, content=True)
            elif isinstance(item, dict) and item.get("type") == "text":
                stream.text(
                    item.get("text", ""), msg_idx, sampled=sampled, content=True
                )
            elif isinstance(item, dict) and item.get("type") in _IMAGE_TYPES:
                # The placeholder stands for caller-provided image data.
                stream.special(self._patch, msg_idx, sampled=False, content=True)

    @staticmethod
    def _text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item if isinstance(item, str) else item.get("text", "")
                for item in content
                if isinstance(item, str)
                or (isinstance(item, dict) and item.get("type") == "text")
            )
        return ""

    # ── system preamble ─────────────────────────────────────────────────

    def _reasoning_line(self) -> str:
        return f"Reasoning strength: {self.config.reasoning_strength or 'high'}."

    def _tool_defs(self, tools: list[ToolSpec]) -> str:
        namespaces: list[str] = []
        for tool in tools:
            ns = _namespace(_function(tool)["name"])
            if ns not in namespaces:
                namespaces.append(ns)
        text = _TOOLS_INTRO
        for ns in namespaces:
            text += '{"name": ' + _json(ns) + ', "description": ' + _json("") + "}\n"
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

    @staticmethod
    def _recipients(tools: list[ToolSpec] | None) -> str:
        recipients = ['"self"']
        seen: list[str] = []
        for tool in tools or []:
            ns = _namespace(_function(tool)["name"])
            if ns not in seen:
                seen.append(ns)
                recipients.append(f'"{ns}.*"')
        recipients.append('"user"')
        return "# Valid recipients: " + ", ".join(recipients) + "."

    def _preamble_tail(self, tools: list[ToolSpec] | None) -> str:
        tail = "\n\n" + self._tool_defs(tools) if tools else ""
        return tail + "\n\n" + self._recipients(tools)

    def _emit_default_system(
        self, stream: _TokenStream, tools: list[ToolSpec] | None
    ) -> None:
        stream.special(self._start, -1, sampled=False, content=False)
        stream.text("system", -1, sampled=False, content=False)
        stream.special(self._message, -1, sampled=False, content=False)
        text = (
            f"{_DEFAULT_SYSTEM}\nKnowledge cutoff: {self.config.knowledge_cutoff or '2026-01-04'}."
            f"\nCurrent date: {self._current_date}.\n\n{self._reasoning_line()}"
            + self._preamble_tail(tools)
        )
        stream.text(text, -1, sampled=False, content=False)
        stream.special(self._eot, -1, sampled=False, content=False)

    def _emit_system(
        self, stream: _TokenStream, msg: Message, i: int, tools: list[ToolSpec] | None
    ) -> None:
        stream.special(self._start, i, sampled=False, content=False)
        stream.text("system", i, sampled=False, content=False)
        stream.special(self._message, i, sampled=False, content=False)
        content = msg.get("content")
        # The template normalises the rendered body, joining adjacent text
        # parts but retaining image markers between otherwise adjacent words.
        parts: list[str | None] = [""]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in _IMAGE_TYPES:
                    parts.extend([None, ""])
                else:
                    parts[-1] = (parts[-1] or "") + self._text_of([item])
        else:
            parts = [self._text_of(content)]
        parts = [
            self._normalise_effort(part) if part is not None else None for part in parts
        ]
        for part in parts:
            if part is None:
                stream.special(self._patch, i, sampled=False, content=True)
            else:
                stream.text(part, i, sampled=False, content=True)
        system_text = "".join("<|patch|>" if part is None else part for part in parts)
        tail = ""
        if "reasoning strength" not in system_text.lower():
            tail = "\n\n" + self._reasoning_line()
        stream.text(tail + self._preamble_tail(tools), i, sampled=False, content=False)
        stream.special(self._eot, i, sampled=False, content=False)

    @staticmethod
    def _normalise_effort(text: str) -> str:
        for old, new in (
            ("Reasoning effort", "Reasoning strength"),
            ("Reasoning Effort", "Reasoning Strength"),
            ("reasoning effort", "reasoning strength"),
            ("REASONING EFFORT", "REASONING STRENGTH"),
        ):
            text = text.replace(old, new)
        return text

    # ── messages ────────────────────────────────────────────────────────

    def _emit_user(self, stream: _TokenStream, msg: Message, i: int) -> None:
        stream.special(self._start, i, sampled=False, content=False)
        stream.text("user", i, sampled=False, content=False)
        stream.special(self._message, i, sampled=False, content=False)
        self._emit_body(stream, msg.get("content"), i, sampled=False)
        stream.special(self._eot, i, sampled=False, content=False)

    def _emit_tool(self, stream: _TokenStream, msg: Message, i: int, name: str) -> None:
        stream.special(self._start, i, sampled=False, content=False)
        stream.text(f"tool {name}", i, sampled=False, content=False)
        stream.special(self._message, i, sampled=False, content=False)
        stream.text(f'<tool_output name="{name}">\n', i, sampled=False, content=False)
        self._emit_body(stream, msg.get("content"), i, sampled=False)
        stream.text("\n</tool_output>", i, sampled=False, content=False)
        stream.special(self._eot, i, sampled=False, content=False)

    @staticmethod
    def _tool_name(messages: list[Message], msg: Message) -> str:
        """The template's name lookup: ``name``, else the last tool call anywhere
        in the conversation with this ``tool_call_id``, else the id itself."""
        if msg.get("name"):
            return str(msg["name"])
        call_id = msg.get("tool_call_id")
        name = call_id or ""
        if call_id is not None:
            for m in messages:
                for tc in m.get("tool_calls") or []:
                    if tc.get("id") == call_id:
                        name = (tc.get("function") or {}).get("name", name)
        return str(name)

    def _emit_assistant(
        self,
        stream: _TokenStream,
        msg: Message,
        i: int,
        *,
        continues: bool,
        next_is_assistant: bool,
    ) -> int:
        """Emit one assistant message; return its final end token.

        ``continues`` means the previous assistant message ended with
        ``<|eom|>``, so the model sampled this message's ``<|start|>assistant``
        itself rather than receiving it as the generation prompt."""

        def header(recipient: str | None, opener_sampled: bool) -> None:
            stream.special(
                self._start, i, sampled=opener_sampled, content=opener_sampled
            )
            stream.text("assistant", i, sampled=opener_sampled, content=opener_sampled)
            if recipient:
                stream.text(f" to={recipient}", i, sampled=True, content=True)
            stream.special(self._message, i, sampled=True, content=True)

        opener_sampled = continues
        reasoning = msg.get("reasoning_content")
        if reasoning:
            header("self", opener_sampled)
            stream.text(reasoning, i, sampled=True, content=True)
            stream.special(self._eom, i, sampled=True, content=True)
            opener_sampled = True

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            end = self._eot
            for n, tc in enumerate(tool_calls):
                header((tc.get("function") or {}).get("name", ""), opener_sampled)
                stream.text(_atem(tc), i, sampled=True, content=True)
                last = n == len(tool_calls) - 1
                end = (
                    (self._eom if next_is_assistant else self._eot)
                    if last
                    else self._eom
                )
                stream.special(end, i, sampled=True, content=True)
                opener_sampled = True
            return end

        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and item.get("type") not in (None, "text")
            for item in content
        ):
            raise NotImplementedError(
                "MuseGlimmerRenderer renders only text in assistant messages"
            )
        recipient = msg.get("recipient") or "user"
        end_turn = msg.get("end_turn")
        if end_turn is None:
            end_turn = recipient == "user"
        header(recipient, opener_sampled)
        self._emit_body(stream, content, i, sampled=True)
        end = self._eot if end_turn else self._eom
        stream.special(end, i, sampled=True, content=True)
        return end

    # ── public API ──────────────────────────────────────────────────────

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
        process_multimodal: bool = True,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")
        self._validate_media(messages, process_multimodal=process_multimodal)
        stream = _TokenStream(self._tokenizer)
        stream.special(self._bos, -1, sampled=False, content=False)
        if not any(m.get("role") == "system" for m in messages):
            self._emit_default_system(stream, tools)
        last_end: int | None = None
        for i, msg in enumerate(messages):
            role = msg.get("role")
            prev_role = messages[i - 1].get("role") if i else None
            if role == "system":
                self._emit_system(stream, msg, i, tools)
            elif role == "user":
                self._emit_user(stream, msg, i)
            elif role == "tool":
                self._emit_tool(stream, msg, i, self._tool_name(messages, msg))
            elif role == "assistant":
                last_end = self._emit_assistant(
                    stream,
                    msg,
                    i,
                    continues=prev_role == "assistant" and last_end == self._eom,
                    next_is_assistant=i + 1 < len(messages)
                    and messages[i + 1].get("role") == "assistant",
                )
                continue
            last_end = None
        if add_generation_prompt:
            stream.special(self._start, -1, sampled=False, content=False)
            stream.text("assistant", -1, sampled=False, content=False)
        return stream.rendered(messages)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
        process_multimodal: bool = True,
    ) -> list[int]:
        return self.render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            process_multimodal=process_multimodal,
        ).token_ids

    def get_stop_token_ids(self) -> list[int]:
        return [self._eot, self._end_of_text]

    def _decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False) if ids else ""

    def _open_recipient(self, prompt_ids: list[int] | None) -> tuple[bool, str | None]:
        """Read the prompt tail after its last ``<|start|>``: whether the
        completion begins inside an open message, and that message's recipient."""
        if not prompt_ids or self._start not in prompt_ids:
            return False, None
        last = len(prompt_ids) - 1 - prompt_ids[::-1].index(self._start)
        tail = prompt_ids[last + 1 :]
        if self._eot in tail or self._eom in tail or self._message not in tail:
            return False, None
        head = self._decode(tail[: tail.index(self._message)])
        match = re.search(r"to=([A-Za-z0-9_.\-:/]+)", head)
        return True, match.group(1) if match else None

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        stops = set(self.get_stop_token_ids())
        ids = list(token_ids)
        for k, tok in enumerate(ids):
            if tok in stops:
                ids = ids[:k]
                break
        # Decode between structural token IDs, recording their text offsets.
        # Ordinary tokens can spell the same markers; those remain body text.
        pieces: list[str] = []
        positions: dict[int, set[int]] = {
            self._start: set(),
            self._message: set(),
            self._eom: set(),
        }
        offset = begin = 0
        for k, tok in enumerate(ids):
            if tok not in positions:
                continue
            chunk = self._decode(ids[begin:k])
            pieces.append(chunk)
            offset += len(chunk)
            positions[tok].add(offset)
            marker = self._decode([tok])
            pieces.append(marker)
            offset += len(marker)
            begin = k + 1
        pieces.append(self._decode(ids[begin:]))
        text = "".join(pieces)
        inside, open_recipient = self._open_recipient(prompt_ids)
        if inside:
            prefill = (f"to={open_recipient}" if open_recipient else "") + "<|message|>"
            positions = {
                tok: {pos + len(prefill) for pos in offsets}
                for tok, offsets in positions.items()
            }
            positions[self._message].add(len(prefill) - len("<|message|>"))
            text = prefill + text

        messages: list[tuple[str | None, str, bool]] = []
        headers: list[re.Match] = []
        for h in _HEADER_RE.finditer(text):
            if h.end() - len("<|message|>") not in positions[self._message]:
                continue
            if h.group("start") and h.start() not in positions[self._start]:
                # A literal opener cannot consume preceding body text. A real
                # message token can still open a bare recipient header after it.
                h = _HEADER_RE.search(text, h.end("start"), h.end())
                assert h is not None
            headers.append(h)

        def cut(h: re.Match) -> int:
            # A bare mid-message header (no ``<|eom|><|start|>``) leaves the
            # whitespace before ``to=`` in the preceding body, as vLLM does.
            return h.start() if h.group("start") or h.start() == 0 else h.end("ws")

        leading = text[: cut(headers[0])] if headers else text
        for n, h in enumerate(headers):
            stop = cut(headers[n + 1]) if n + 1 < len(headers) else len(text)
            end = next(
                (pos for pos in sorted(positions[self._eom]) if h.end() <= pos < stop),
                None,
            )
            messages.append(
                (
                    h.group("rcpt"),
                    text[h.end() : end if end is not None else stop],
                    end is not None,
                )
            )

        reasoning_parts: list[str] = []
        content_parts: list[str] = [leading.strip()] if leading.strip() else []
        tool_bodies: list[tuple[str, str, bool]] = []
        for recipient, body, closed in messages:
            if recipient == "self":
                reasoning_parts.append(body)
            elif recipient in (None, "user"):
                content_parts.append(body)
            else:
                tool_bodies.append((recipient, body, closed))

        # Unfinished means the last message is reasoning with no end marker.
        # Earlier, closed messages (answer text, tool calls) keep their output.
        reasoning_complete = not (
            messages and messages[-1][0] == "self" and not messages[-1][2]
        )
        # Separate reasoning messages are joined with a newline (vLLM's parser).
        # Empty bodies (e.g. a header the prompt already opened) add nothing.
        reasoning = (
            "\n".join(p for p in reasoning_parts if p) if reasoning_parts else None
        )
        registered = {_function(tool)["name"] for tool in tools or []}
        calls = [
            c
            for recipient, body, closed in tool_bodies
            for c in self._parse_calls(body, registered)
        ]
        content = "".join(content_parts)
        if (
            messages
            and messages[-1][0] == "user"
            and not messages[-1][2]
            and not any(call.name for call in calls)
            and content.endswith("\n")
            and "<atem:invoke" not in content
            and "<atem:function_calls>" not in content
        ):
            # vLLM's non-streaming content regex ends at `$`, before one final LF.
            content = content[:-1]
        if tool_bodies and not any(call.name for call in calls):
            # Stock vLLM preserves malformed tool markup as content when no
            # complete invocation can be dispatched; it does not invent a call.
            remainder = _STRIP_REASONING_RE.sub("", text)
            remainder = _STRIP_OPEN_REASONING_RE.sub("", remainder)
            if "<atem:invoke" in remainder or "<atem:function_calls>" in remainder:
                content = remainder
            elif not content:
                content = "" if "to=self<|message|>" in text else text
        return ParsedResponse(
            content=content,
            reasoning_content=reasoning,
            tool_calls=calls,
            reasoning_complete=reasoning_complete,
        )

    @staticmethod
    def _parse_calls(body: str, registered: set[str]) -> list[ParsedToolCall]:
        """Every ``<atem:invoke>`` in one tool-channel message."""
        calls: list[ParsedToolCall] = []
        last_end = 0
        for m in _INVOKE_RE.finditer(body):
            last_end = m.end()
            name_match = _INVOKE_NAME_RE.search(m.group("attrs"))
            name = name_match.group("name") if name_match else None
            if not name:
                calls.append(
                    ParsedToolCall(
                        raw=m.group(0), status=ToolCallParseStatus.MISSING_NAME
                    )
                )
                continue
            if name not in registered:
                head, sep, tail = name.partition(".")
                if sep and head == tail and head in registered:
                    name = head
            arguments: dict[str, Any] = {}
            for p in _PARAM_RE.finditer(m.group("body")):
                value = p.group("value")
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
                arguments[p.group("key")] = value
            calls.append(
                ParsedToolCall(
                    raw=m.group(0),
                    name=name,
                    arguments=arguments,
                    status=ToolCallParseStatus.OK,
                )
            )
        rest = body[last_end:]
        if _INVOKE_OPEN_RE.search(rest):
            # An invoke opened but never closed: generation stopped inside it.
            calls.append(
                ParsedToolCall(
                    raw=rest.strip(),
                    status=ToolCallParseStatus.UNCLOSED_BLOCK,
                )
            )
        elif not calls:
            calls.append(
                ParsedToolCall(
                    raw=body,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            )
        return calls

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        previous_multi_modal_data: MultiModalData | None = None,
        process_multimodal: bool = True,
    ) -> RenderedTokens | None:
        self._validate_media(new_messages, process_multimodal=process_multimodal)
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None
        # A new system message would remove the default preamble from the prefix.
        if any(m.get("role") == "system" for m in new_messages):
            return None
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention, new_messages
        ):
            return None

        stops = set(self.get_stop_token_ids())
        completion = list(previous_completion_ids)
        cut = next((k for k, tok in enumerate(completion) if tok in stops), None)
        parsed = self.parse_response(
            completion[:cut] if cut is not None else completion,
            tools=tools,
            prompt_ids=previous_prompt_ids,
        )
        if cut is not None:
            # A stop sampled inside reasoning has no canonical rendering.
            if parsed.reasoning_complete is False:
                return None
            completion = completion[: cut + 1]
        elif not completion:
            return None
        else:
            # Truncated: close the turn as a full render of the parsed message
            # would — end the open reasoning (``<|eom|>``) and add the empty
            # ``to=user`` answer, or end the open answer/tool message.
            if parsed.reasoning_complete is False:
                completion.append(self._eom)
            if completion[-1] == self._eom:
                completion += [
                    self._start,
                    *self._tokenizer.encode(
                        "assistant to=user", add_special_tokens=False
                    ),
                    self._message,
                ]
            completion.append(self._eot)

        # ATEM carries no call IDs, so a bridge cannot associate unnamed results
        # with distinct function names. Full rendering can use the history's IDs.
        unnamed = [
            m for m in new_messages if m.get("role") == "tool" and not m.get("name")
        ]
        names: list[str] = []
        if unnamed:
            parsed = self.parse_response(
                previous_completion_ids, tools=tools, prompt_ids=previous_prompt_ids
            )
            names = [c.name for c in parsed.tool_calls if c.name]
            if len(names) != len(unnamed) or len(set(names)) > 1:
                return None

        previous = list(previous_prompt_ids) + completion
        stream = _TokenStream(self._tokenizer)
        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            if role == "user":
                self._emit_user(stream, msg, i)
            elif role == "tool":
                name = msg.get("name") or names.pop(0)
                self._emit_tool(stream, msg, i, str(name))
            else:
                return None
        stream.special(self._start, -1, sampled=False, content=False)
        stream.text("assistant", -1, sampled=False, content=False)
        ext = stream.rendered(new_messages, prefix=len(previous))
        return RenderedTokens(
            token_ids=previous + ext.token_ids,
            message_indices=[-1] * len(previous) + ext.message_indices,
            sampled_mask=[False] * (len(previous) + len(ext.token_ids)),
            is_content=ext.is_content,
            message_roles=ext.message_roles,
            message_tool_names=ext.message_tool_names,
        )
