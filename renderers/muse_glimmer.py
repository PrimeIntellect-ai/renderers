"""MuseGlimmer Renderer — hard-coded Python mirroring the ATEM chat template.

Scope: ``meta-models/Muse-Glimmer-30B``.

ATEM is a channel protocol closer to gpt-oss Harmony than to the Qwen/GLM families:

* Every turn opens with ``<|start|>{role}<|message|>`` and closes with ``<|eot|>``
  (turn over) or ``<|eom|>`` (channel over, same speaker continues). Only ``<|eot|>``
  and ``<|end_of_text|>`` are stop tokens — ``<|eom|>`` is *not*, since generation
  continues into the next channel.
* Assistant turns carry a recipient: ``to=self`` is the reasoning channel,
  ``to=<tool>`` a tool call, and ``to=user`` (or no recipient) the user-facing reply.
* The generation prompt is bare ``<|start|>assistant`` — the model itself emits the
  ``to=...`` recipient, the ``<|message|>`` separator, the body and the closing token.
  So within an assistant turn only the *first* ``<|start|>assistant`` is scaffold;
  every later channel opener is model-sampled. Masks reflect that.
* The system block is emitted unconditionally, and always ends with a
  ``# Valid recipients:`` line listing ``"self"``, one entry per tool namespace, and
  ``"user"``. A default block is synthesised when the caller supplies no system message.
* Tool calls are XML-ish ``<atem:function_calls>`` blocks, not JSON.
"""

from __future__ import annotations

import json
from datetime import date as _date
from typing import Any

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    Tokenizer,
    ToolSpec,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import MuseGlimmerRendererConfig
from renderers.parsing import parse_muse_glimmer

# ---------------------------------------------------------------------------
# Constants — must match the Jinja chat template's literal strings exactly.
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_TEXT = "You are a helpful AI assistant."

_TOOL_DEFS_INTRO = (
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

_TOOL_DEFS_EXAMPLE = (
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

# The template normalises "Reasoning effort" to "Reasoning strength" in caller-supplied
# system prompts. Jinja has no case-insensitive replace, so it hard-codes four casings.
_EFFORT_REPLACEMENTS = (
    ("Reasoning effort", "Reasoning strength"),
    ("Reasoning Effort", "Reasoning Strength"),
    ("reasoning effort", "reasoning strength"),
    ("REASONING EFFORT", "REASONING STRENGTH"),
)


def _tojson(value: Any) -> str:
    """Match Jinja's ``tojson`` filter as transformers configures it."""
    return json.dumps(value, separators=(", ", ": "), ensure_ascii=False)


class MuseGlimmerRenderer:
    """Deterministic message → token renderer for Muse Glimmer."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: MuseGlimmerRendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or MuseGlimmerRendererConfig()
        # The template re-emits historical reasoning verbatim (there is no
        # drop-thinking knob), so full renders always retain it.
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )
        # Resolve once, not per render, so an instance stays deterministic for its
        # lifetime even when the caller leaves the date unpinned.
        self._current_date = self.config.current_date or _date.today().strftime(
            "%Y-%m-%d"
        )

        self._bos = self._token_id("<|begin_of_text|>")
        self._start = self._token_id("<|start|>")
        self._message = self._token_id("<|message|>")
        self._eom = self._token_id("<|eom|>")
        self._eot = self._token_id("<|eot|>")
        self._end_of_text = self._token_id("<|end_of_text|>")

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Template fragments
    # ------------------------------------------------------------------

    @staticmethod
    def _content_str(content: Any) -> str:
        """Mirror the template's ``render_content`` macro."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                kind = item.get("type")
                if kind in {"image", "image_url", "video", "video_url"}:
                    raise ValueError(
                        "Muse Glimmer rendering is text-only in PrimeRL; image and video content is not supported."
                    )
                if kind == "text":
                    parts.append(item["text"])
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    @staticmethod
    def _namespaces(tools: list[ToolSpec]) -> list[str]:
        """Unique tool namespaces in first-appearance order."""
        seen: list[str] = []
        for tool in tools:
            fn = tool.get("function", tool)
            ns = fn["name"].split(".")[0]
            if ns not in seen:
                seen.append(ns)
        return seen

    def _reasoning_line(self) -> str:
        return f"Reasoning strength: {self.config.reasoning_strength}."

    def _system_meta(self, tools: list[ToolSpec] | None) -> str:
        recipients = ['"self"']
        if tools:
            recipients += [f'"{ns}.*"' for ns in self._namespaces(tools)]
        recipients.append('"user"')
        return "# Valid recipients: " + ", ".join(recipients) + "."

    def _system_suffix(self, tools: list[ToolSpec] | None) -> str:
        """Tool definitions (if any) plus the valid-recipients line."""
        suffix = "\n\n" + self._tool_defs(tools) if tools else ""
        return suffix + "\n\n" + self._system_meta(tools)

    def _tool_defs(self, tools: list[ToolSpec]) -> str:
        # The template reads namespace descriptions from an optional
        # ``tool_namespace_descriptions`` kwarg and renders "" for anything missing.
        # No caller sets it, and a mapping field would break the config's frozen/
        # hashable contract, so descriptions are always empty here.
        out = _TOOL_DEFS_INTRO
        for ns in self._namespaces(tools):
            out += '{"name": ' + _tojson(ns) + ', "description": ' + _tojson("") + "}\n"
        out += "// Function schemas"
        for tool in tools:
            fn = tool.get("function", tool)
            out += (
                "\n{"
                + '"name": '
                + _tojson(fn.get("name"))
                + ', "description": '
                + _tojson(fn.get("description"))
                + ', "parameters": '
                + _tojson(fn.get("parameters"))
                + "}"
            )
        return out + _TOOL_DEFS_EXAMPLE

    @staticmethod
    def _atem_call(tool_call: dict) -> str:
        """Mirror the template's ``render_atem`` macro."""
        fn = tool_call.get("function", tool_call)
        args = fn.get("arguments")
        if not isinstance(args, dict):
            raise ValueError(
                "Muse Glimmer ATEM rendering requires tool_call.function.arguments to be a dict; "
                f"got {type(args).__name__}."
            )
        out = '<atem:function_calls>\n<atem:invoke name="' + fn.get("name", "") + '">\n'
        for key, value in args.items():
            out += f'<atem:parameter name="{key}">'
            if isinstance(value, bool):
                out += "true" if value else "false"
            elif value is None:
                out += "null"
            elif isinstance(value, (dict, list, tuple)):
                out += _tojson(value)
            else:
                out += str(value)
            out += "</atem:parameter>\n"
        return out + "</atem:invoke>\n</atem:function_calls>"

    @staticmethod
    def _tool_name(msg: Message, messages: list[Message]) -> str:
        """Resolve a tool response's name, falling back to tool_call_id lookup."""
        name = msg.get("name")
        if name:
            return name
        call_id = msg.get("tool_call_id")
        for other in messages:
            for tc in other.get("tool_calls") or []:
                if call_id is not None and tc.get("id") == call_id:
                    return tc.get("function", tc).get("name", "")
        return call_id or ""

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit_special(
            token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(
            text: str, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_segments(
            segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool
        ) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        def emit_assistant_segments(
            segments: list[tuple[str, bool]], msg_idx: int
        ) -> None:
            """One BPE pass with per-token attribution.

            On assistant tokens ``is_content == sampled_mask`` by the RenderedTokens
            contract, so a single per-segment flag drives both.
            """
            for tok_id, flag in attribute_text_segments(self._tokenizer, segments):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(flag)
                content_mask.append(flag)

        emit_special(self._bos, -1, is_sampled=False, is_content=False)
        assistant_run_open = False

        # ── Default system block, when the caller supplied none anywhere ──
        if not any(m.get("role") == "system" for m in messages):
            emit_special(self._start, -1, is_sampled=False, is_content=False)
            emit_text("system", -1, is_sampled=False, is_content=False)
            emit_special(self._message, -1, is_sampled=False, is_content=False)
            # One encode pass: splitting here would break BPE merges across the
            # boundaries (e.g. "." + "\n\n" must merge into a single ".\n\n" token).
            body = _DEFAULT_SYSTEM_TEXT
            body += f"\nKnowledge cutoff: {self.config.knowledge_cutoff}."
            body += f"\nCurrent date: {self._current_date}."
            body += "\n\n" + self._reasoning_line()
            body += self._system_suffix(tools)
            emit_text(body, -1, is_sampled=False, is_content=False)
            emit_special(self._eot, -1, is_sampled=False, is_content=False)

        # ── Message loop ──────────────────────────────────────────────
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role != "assistant":
                assistant_run_open = False

            if role == "system":
                sys_text = self._content_str(msg.get("content"))
                for old, new in _EFFORT_REPLACEMENTS:
                    sys_text = sys_text.replace(old, new)
                emit_special(self._start, idx, is_sampled=False, is_content=False)
                emit_text("system", idx, is_sampled=False, is_content=False)
                emit_special(self._message, idx, is_sampled=False, is_content=False)
                # Caller text is body, everything after is scaffold — but both go
                # through one BPE pass so the boundary can't shift merges.
                trailer = ""
                if "reasoning strength" not in sys_text.lower():
                    trailer += "\n\n" + self._reasoning_line()
                trailer += self._system_suffix(tools)
                segments: list[tuple[str, bool]] = []
                if sys_text:
                    segments.append((sys_text, True))
                segments.append((trailer, False))
                emit_segments(segments, idx, is_sampled=False)
                emit_special(self._eot, idx, is_sampled=False, is_content=False)

            elif role == "user":
                emit_special(self._start, idx, is_sampled=False, is_content=False)
                emit_text("user", idx, is_sampled=False, is_content=False)
                emit_special(self._message, idx, is_sampled=False, is_content=False)
                emit_text(
                    self._content_str(msg.get("content")),
                    idx,
                    is_sampled=False,
                    is_content=True,
                )
                emit_special(self._eot, idx, is_sampled=False, is_content=False)

            elif role == "tool":
                name = self._tool_name(msg, messages)
                emit_special(self._start, idx, is_sampled=False, is_content=False)
                emit_text(f"tool {name}", idx, is_sampled=False, is_content=False)
                emit_special(self._message, idx, is_sampled=False, is_content=False)
                body = self._content_str(msg.get("content"))
                emit_segments(
                    [
                        (f'<tool_output name="{name}">\n', False),
                        (body, True),
                        ("\n</tool_output>", False),
                    ],
                    idx,
                    is_sampled=False,
                )
                emit_special(self._eot, idx, is_sampled=False, is_content=False)

            elif role == "assistant":
                # ``end_token`` mirrors the template: a run of same-role messages keeps
                # the speaker open with <|eom|> until the last one.
                same_role_next = (
                    idx + 1 < len(messages) and messages[idx + 1].get("role") == role
                )
                end_token = self._eom if same_role_next else self._eot
                # Only the first channel opener of a turn is scaffold; the model emits
                # any subsequent <|start|>assistant itself.
                opened = assistant_run_open

                def open_channel(recipient: str | None, *, _idx: int = idx) -> None:
                    """Emit ``<|start|>assistant[ to=X]<|message|>``.

                    The first opener of a turn is the generation prompt, which the model
                    never samples; it emits every later channel opener itself.
                    """
                    nonlocal opened
                    emit_special(
                        self._start, _idx, is_sampled=opened, is_content=opened
                    )
                    header: list[tuple[str, bool]] = [("assistant", opened)]
                    if recipient:
                        header.append((f" to={recipient}", True))
                    emit_assistant_segments(header, _idx)
                    emit_special(self._message, _idx, is_sampled=True, is_content=True)
                    opened = True

                reasoning = msg.get("reasoning_content")
                if reasoning:
                    open_channel("self")
                    emit_text(reasoning, idx, is_sampled=True, is_content=True)
                    emit_special(self._eom, idx, is_sampled=True, is_content=True)

                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for n, tc in enumerate(tool_calls):
                        last = n == len(tool_calls) - 1
                        open_channel(tc.get("function", tc).get("name", ""))
                        emit_text(
                            self._atem_call(tc), idx, is_sampled=True, is_content=True
                        )
                        emit_special(
                            end_token if last else self._eom,
                            idx,
                            is_sampled=True,
                            is_content=True,
                        )
                    assistant_run_open = end_token == self._eom
                else:
                    recipient = msg.get("recipient") or "user"
                    end_turn = msg.get("end_turn")
                    if end_turn is None:
                        end_turn = not (recipient and recipient != "user")
                    open_channel(recipient)
                    emit_text(
                        self._content_str(msg.get("content")),
                        idx,
                        is_sampled=True,
                        is_content=True,
                    )
                    emit_special(
                        self._eot if end_turn else self._eom,
                        idx,
                        is_sampled=True,
                        is_content=True,
                    )
                    assistant_run_open = not end_turn

        if add_generation_prompt:
            emit_special(self._start, -1, is_sampled=False, is_content=False)
            emit_text("assistant", -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
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
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        return parse_muse_glimmer(
            self._tokenizer,
            token_ids,
            stop_ids={self._eot, self._end_of_text},
            tools=tools,
        )

    def get_stop_token_ids(self) -> list[int]:
        # <|eom|> deliberately excluded: it closes a channel, not the turn.
        return [self._eot, self._end_of_text]

    # ------------------------------------------------------------------
    # bridge_to_next_turn
    # ------------------------------------------------------------------

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
        ):
            return None
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention, new_messages
        ):
            return None
        # A new system message rewrites the whole system block, which sits at the
        # front of the prompt — it cannot be appended.
        if any(m.get("role") == "system" for m in new_messages):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._eot, self._end_of_text},
            synthesize_close=self._eot,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []
        ext_indices: list[int] = []
        ext_content: list[bool] = []

        def emit_special(token_id: int, msg_idx: int = -1) -> None:
            ext.append(token_id)
            ext_indices.append(msg_idx)
            ext_content.append(False)

        def emit_text(
            text: str, msg_idx: int = -1, *, is_content: bool = False
        ) -> None:
            ids = self._encode(text)
            ext.extend(ids)
            ext_indices.extend([msg_idx] * len(ids))
            ext_content.extend([is_content] * len(ids))

        def emit_segments(segments: list[tuple[str, bool]], msg_idx: int) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                ext.append(tok_id)
                ext_indices.append(msg_idx)
                ext_content.append(is_content)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            if role == "user":
                emit_special(self._start, i)
                emit_text("user", i)
                emit_special(self._message, i)
                emit_text(self._content_str(msg.get("content")), i, is_content=True)
                emit_special(self._eot, i)
            elif role == "tool":
                if not msg.get("name"):
                    return None
                name = self._tool_name(msg, new_messages)
                emit_special(self._start, i)
                emit_text(f"tool {name}", i)
                emit_special(self._message, i)
                emit_segments(
                    [
                        (f'<tool_output name="{name}">\n', False),
                        (self._content_str(msg.get("content")), True),
                        ("\n</tool_output>", False),
                    ],
                    i,
                )
                emit_special(self._eot, i)
            else:
                return None

        emit_special(self._start, -1)
        emit_text("assistant", -1)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )
