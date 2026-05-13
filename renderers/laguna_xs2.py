"""Laguna-XS.2 Renderer.

Main properties:
- Prefix is the single token ``〈|EOS|〉`` (also the EOS / stop token).
- Role markers are block-style: ``<system>...</system>``, ``<user>...</user>``,
  ``<assistant>...</assistant>``, ``<tool_response>...</tool_response>``. Of
  these, only ``<assistant>`` / ``</assistant>`` are single (added) tokens
  in the tokenizer; everything else is plain text and BPEs into multiple
  subwords.
- Assistant turn has an explicit close: ``</assistant>`` is the canonical
  stop token (alongside ``〈|EOS|〉``).
- Tool calls: ``<tool_call>`` / ``</tool_call>`` ARE single tokens, but the
  inner ``<arg_key>`` / ``</arg_key>`` / ``<arg_value>`` / ``</arg_value>``
  markers are plain text — parsed via regex on the decoded inner block.
- The template bakes in a default system prompt when ``messages[0]`` is not
  a system message. The system block also contains the tools section (under
  a ``### Tools`` header with an ``<available_tools>`` listing and prose
  format instructions that vary on ``enable_thinking``).
- Reasoning is rendered for every assistant message — no last-user-index
  gating. ``preserve_all_thinking`` and
  ``preserve_thinking_between_tool_calls`` are accepted for protocol
  uniformity but are effectively no-ops since past reasoning is preserved
  by default.
"""

from __future__ import annotations

import json

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Content,
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    reject_assistant_in_extension,
)
from renderers.parsing import parse_laguna_xs2

_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful, conversationally-fluent assistant made by Poolside. "
    "You are here to be helpful to users through natural language conversations."
)

_TOOLS_HEADER = (
    "\n\n### Tools\n\n"
    "You may call functions to assist with the user query.\n"
    "All available function signatures are listed below:\n"
    "<available_tools>\n"
)

_TOOLS_FOOTER_THINKING = (
    "</available_tools>\n\n"
    "Wrap your thinking in '<think>', '</think>' tags, followed by a function call. "
    "For each function call, return an unescaped XML-like object with function name "
    "and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n"
    "<think> your thoughts here </think>\n"
    "<tool_call>function-name\n"
    "<arg_key>argument-key</arg_key>\n"
    "<arg_value>value-of-argument-key</arg_value>\n"
    "</tool_call>"
)

_TOOLS_FOOTER_NO_THINKING = (
    "</available_tools>\n\n"
    "For each function call, return an unescaped XML-like object with function name "
    "and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n"
    "<tool_call>function-name\n"
    "<arg_key>argument-key</arg_key>\n"
    "<arg_value>value-of-argument-key</arg_value>\n"
    "</tool_call>"
)


class LagunaXS2Renderer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = False,
        preserve_all_thinking: bool = False,
        preserve_thinking_between_tool_calls: bool = False,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking
        # Accepted for protocol uniformity. The chat template renders
        # reasoning on every assistant message regardless, so flipping
        # these flags has no effect on the byte-level output.
        self._preserve_all_thinking = preserve_all_thinking
        self._preserve_thinking_between_tool_calls = (
            preserve_thinking_between_tool_calls
        )

        self._eos = self._token_id("〈|EOS|〉")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._assistant = self._token_id("<assistant>")
        self._assistant_end = self._token_id("</assistant>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")

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

    @staticmethod
    def _visible_text(content: Content | None) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return ""

    @staticmethod
    def _thinking_text(content: Content | None) -> str:
        """Concatenate ``ThinkingPart`` entries from list-form content.

        Used as a reasoning source in ``_render_assistant`` when neither
        ``reasoning`` nor ``reasoning_content`` is present on the message.
        Returns ``""`` for any non-list input.
        """
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "thinking":
                parts.append(item.get("thinking", ""))
        return "".join(parts)

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

        def emit_special(token_id: int, msg_idx: int) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)

        def emit_text(text: str, msg_idx: int) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))

        emit_special(self._eos, -1)

        # ── System header (absorbs messages[0] if it's a system message) ──
        system_content = _DEFAULT_SYSTEM_MESSAGE
        system_msg_idx = -1
        if messages and messages[0].get("role") == "system":
            system_content = self._visible_text(messages[0].get("content"))
            system_msg_idx = 0

        has_sys_content = bool(system_content and system_content.strip())
        # Mirrors the template's ``(system_message and system_message.strip()) or tools``
        # gate: when the caller passes an empty system message and no tools,
        # the whole ``<system>...</system>`` block is omitted.
        if has_sys_content or tools:
            # The template emits ``<system>\n`` then conditionally a second
            # ``\n``. Bundle those into one emit so BPE merges ``\n\n`` into
            # its single-token form (rather than two ``\n`` atoms).
            emit_text("<system>\n\n" if has_sys_content else "<system>\n", -1)
            if has_sys_content:
                emit_text(system_content.rstrip(), system_msg_idx)
            if tools:
                tool_text = _TOOLS_HEADER
                for tool in tools:
                    tool_text += json.dumps(tool, ensure_ascii=False) + "\n"
                tool_text += (
                    _TOOLS_FOOTER_THINKING
                    if self._enable_thinking
                    else _TOOLS_FOOTER_NO_THINKING
                )
                emit_text(tool_text, -1)
            emit_text("\n</system>\n", -1)

        # ── Per-message loop ──────────────────────────────────────────
        for i, msg in enumerate(messages):
            content = self._visible_text(msg.get("content"))

            match msg["role"]:
                case "system":
                    # Already consumed in the header block.
                    if i == 0:
                        continue
                    emit_text("<system>\n" + content + "\n</system>\n", i)
                case "user":
                    emit_text("<user>\n" + content + "\n</user>\n", i)
                case "assistant":
                    self._render_assistant(
                        msg, i, content, emit_special=emit_special, emit_text=emit_text
                    )
                case "tool":
                    emit_text("<tool_response>\n" + content + "\n</tool_response>\n", i)

        # ── Generation prompt ─────────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._assistant, -1)
            emit_text("\n", -1)
            if self._enable_thinking:
                emit_special(self._think, -1)
            else:
                emit_special(self._think_end, -1)

        return RenderedTokens(token_ids=tokens, message_indices=indices)

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

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        return parse_laguna_xs2(
            self._tokenizer,
            token_ids,
            stop_ids={self._assistant_end, self._eos},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._assistant_end, self._eos]

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

        # The canonical assistant-turn close is ``</assistant>``. ``〈|EOS|〉``
        # also stops generation; either being the final token means the turn
        # ended cleanly. Truncation (no stop token at the tail) synthesises
        # ``</assistant>\n`` — the same scaffold the template emits.
        previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
        stop_ids = {self._assistant_end, self._eos}
        if (
            not previous_ids[len(previous_prompt_ids) :]
            or previous_ids[-1] not in stop_ids
        ):
            previous_ids.append(self._assistant_end)
            previous_ids.extend(self._encode("\n"))

        ext: list[int] = []

        def emit_special(token_id: int, _msg_idx: int = -1) -> None:
            ext.append(token_id)

        def emit_text(text: str, _msg_idx: int = -1) -> None:
            ext.extend(self._encode(text))

        for msg in new_messages:
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))
            if role == "user":
                emit_text("<user>\n" + content + "\n</user>\n")
            elif role == "system":
                emit_text("<system>\n" + content + "\n</system>\n")
            elif role == "tool":
                emit_text("<tool_response>\n" + content + "\n</tool_response>\n")
            else:
                return None

        emit_special(self._assistant)
        emit_text("\n")
        if self._enable_thinking:
            emit_special(self._think)
        else:
            emit_special(self._think_end)

        return RenderedTokens(token_ids=previous_ids + ext)

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        else:
            # When the caller stores reasoning as a ``ThinkingPart`` inside
            # a list-form ``content`` (e.g. after parse_response →
            # reserialize), pull it out here so it survives the re-render.
            part_thinking = self._thinking_text(msg.get("content"))
            if part_thinking:
                reasoning_content = part_thinking

        emit_special(self._assistant, msg_idx)
        emit_text("\n", msg_idx)

        if reasoning_content:
            emit_special(self._think, msg_idx)
            emit_text("\n" + reasoning_content.strip() + "\n", msg_idx)
            emit_special(self._think_end, msg_idx)
        else:
            emit_special(self._think_end, msg_idx)

        # Combined newline-after-</think> with optional content. Bundling
        # preserves BPE merges across the boundary.
        post_think_text = "\n"
        if content.strip():
            post_think_text += content.strip() + "\n"
        emit_text(post_think_text, msg_idx)

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function") or tc
            name = func.get("name", "")
            arguments = func.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            emit_special(self._tool_call, msg_idx)
            inner = name + "\n"
            if isinstance(arguments, dict):
                for k, v in arguments.items():
                    inner += "<arg_key>" + k + "</arg_key>\n"
                    if isinstance(v, str):
                        val_text = v
                    else:
                        val_text = json.dumps(v, ensure_ascii=False)
                    inner += "<arg_value>" + val_text + "</arg_value>\n"
            emit_text(inner, msg_idx)
            emit_special(self._tool_call_end, msg_idx)
            emit_text("\n", msg_idx)

        emit_special(self._assistant_end, msg_idx)
        emit_text("\n", msg_idx)
