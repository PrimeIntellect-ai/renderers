"""Llama-3 Renderer — hard-coded Python mirroring Meta's Llama-3 chat template.

Initial scope: Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct (and the
unrestricted ``unsloth/Llama-3.2-{1B,3B}-Instruct`` mirror, which ships a
byte-identical chat template). Other Llama-3.x sizes ship slightly
different templates and are NOT covered by this renderer until parity is
verified.

Notable differences from the Qwen / GLM family renderers:

* No ``<think>`` / reasoning channel — Llama-3 doesn't ship a
  reasoning-content concept, so ``preserve_*_thinking`` flags don't
  apply.
* ``<|begin_of_text|>`` (BOS) is emitted at the very start of every
  render. The chat template never omits it.
* The system block is emitted **unconditionally** with a fixed
  ``Cutting Knowledge Date: December 2023\\nToday Date: <date>\\n\\n``
  preamble — even when no system message is supplied. Empty system
  message → block ends with ``\\n\\n<|eot_id|>``.
* Tools default to "first-user-message" mode (matching the chat
  template's default ``tools_in_user_message=True``): tool descriptions
  + JSON signatures are injected into the first user message rather
  than the system block. Set ``Llama3RendererConfig.tools_in_user_message
  = False`` to flip to system-block mode.
* ``Llama3RendererConfig.date_string`` is pinned at ``"26 Jul 2024"`` by
  default to match the chat template's ``strftime`` fallback (and keep
  output deterministic). Override per instance for production runs that
  want today's date.
* Tool calls: a single ``{"name": "...", "parameters": ...}`` JSON blob
  inside the assistant body. The chat template explicitly raises if
  ``message.tool_calls | length != 1``; this renderer matches that.
* Tool responses: rendered with role ``ipython`` regardless of whether
  the source message used ``role: "tool"`` or ``role: "ipython"``. The
  chat template runs ``content | tojson`` on any mapping/iterable
  content — and Jinja considers strings iterable, so plain string
  contents get JSON-quoted. We mirror that exactly.
"""

from __future__ import annotations

import json
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.configs import Llama3RendererConfig
from renderers.parsing import parse_llama_3

# ---------------------------------------------------------------------------
# Constants — must match the Jinja chat template's literal strings exactly.
# ---------------------------------------------------------------------------

_CUTTING_KNOWLEDGE_DATE = "December 2023"

# Tools-in-system intro: emitted into the system block when tools is set
# AND tools_in_user_message=False. Note: the chat template puts these
# three string literals back-to-back with NO newline between the second
# and third, so there's no space before "Do not use variables.".
_TOOLS_IN_SYSTEM_INTRO = (
    "You have access to the following functions. To call a function, "
    "please respond with JSON for a function call."
    'Respond in the format {"name": function name, "parameters": '
    "dictionary of argument name and its value}."
    "Do not use variables.\n\n"
)

# Tools-in-user intro: emitted into the first user message when tools is
# set AND tools_in_user_message=True (the default).
_TOOLS_IN_USER_INTRO = (
    "Given the following functions, please respond with a JSON for a "
    "function call with its proper arguments that best answers the given "
    "prompt.\n\n"
    'Respond in the format {"name": function name, "parameters": '
    "dictionary of argument name and its value}."
    "Do not use variables.\n\n"
)


class Llama3Renderer:
    """Deterministic message → token renderer for Llama-3.x Instruct models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Llama3RendererConfig | None = None,
    ):
        config = config or Llama3RendererConfig()
        if config.preserve_all_thinking or config.preserve_thinking_between_tool_calls:
            raise NotImplementedError(
                "Llama-3 doesn't ship a reasoning_content channel — the chat "
                "template has no <think> block to preserve or drop. "
                "preserve_*_thinking flags are not applicable."
            )
        self._tokenizer = tokenizer
        self.config = config

        self._bos = self._token_id("<|begin_of_text|>")
        self._start_header = self._token_id("<|start_header_id|>")
        self._end_header = self._token_id("<|end_header_id|>")
        self._eot = self._token_id("<|eot_id|>")
        self._end_of_text = self._token_id("<|end_of_text|>")
        # ``<|eom_id|>`` shows up in some Llama-3 tool-calling traces (the
        # "ipython" / python-tag flow) but the standard 3.2 chat template
        # closes turns with ``<|eot_id|>``. We still treat eom as a stop
        # token so models that emit it terminate cleanly.
        self._eom = self._token_id("<|eom_id|>")

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
    def _content_str(content: Any) -> str:
        """Render content to a plain string. Handles ``str``, list-of-text-parts,
        and ``None``. Matches the chat template's ``message.content | trim``
        callers, which expect a string in."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    raise ValueError(f"Unexpected content item: {item}")
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    @staticmethod
    def _tool_response_str(content: Any) -> str:
        """Mirror the chat template's tool-response branch:
        ``{% if message.content is mapping or message.content is iterable %}
        {{ message.content | tojson }} {% else %} {{ message.content }}``.

        In Jinja, **strings are iterable** — so plain-string tool contents
        also go through ``tojson`` (i.e. ``json.dumps``), wrapping them in
        quotes and escaping. Non-iterable scalars (numbers, bools, None)
        fall through to literal stringification.
        """
        if content is None:
            return ""
        if isinstance(content, (dict, list, str)):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

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

        def emit_ids(ids: list[int], msg_idx: int) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))

        def emit_special(token_id: int, msg_idx: int) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)

        def emit_text(text: str, msg_idx: int) -> None:
            emit_ids(self._encode(text), msg_idx)

        # ── 0. BOS ──────────────────────────────────────────────────
        emit_special(self._bos, -1)

        # ── 1. System block (always emitted) ────────────────────────
        first_is_system = messages[0].get("role") == "system"
        sys_idx = 0 if first_is_system else -1
        sys_text = (
            self._content_str(messages[0].get("content")).strip()
            if first_is_system
            else ""
        )

        emit_special(self._start_header, sys_idx)
        emit_text("system", sys_idx)
        emit_special(self._end_header, sys_idx)
        body = "\n\n"
        if tools is not None:
            body += "Environment: ipython\n"
        body += f"Cutting Knowledge Date: {_CUTTING_KNOWLEDGE_DATE}\n"
        body += f"Today Date: {self.config.date_string}\n\n"
        if tools is not None and not self.config.tools_in_user_message:
            body += _TOOLS_IN_SYSTEM_INTRO
            for t in tools:
                body += json.dumps(t, indent=4, ensure_ascii=False) + "\n\n"
        body += sys_text
        emit_text(body, sys_idx)
        emit_special(self._eot, sys_idx)

        # ── 2. Body messages ────────────────────────────────────────
        body_messages = messages[1:] if first_is_system else messages
        offset = 1 if first_is_system else 0

        i = 0
        # 2a. tools_in_user_message mode pulls the first user message
        #     into a special block with the tools description prepended.
        if tools is not None and self.config.tools_in_user_message:
            if i >= len(body_messages):
                raise ValueError(
                    "Cannot place tools in the first user message — no user "
                    "message was provided."
                )
            first_user = body_messages[i]
            if first_user.get("role") != "user":
                raise ValueError(
                    "tools_in_user_message=True requires the first non-system "
                    f"message to be 'user'; got {first_user.get('role')!r}."
                )
            user_idx = i + offset
            emit_special(self._start_header, user_idx)
            emit_text("user", user_idx)
            emit_special(self._end_header, user_idx)
            user_body = "\n\n" + _TOOLS_IN_USER_INTRO
            for t in tools:
                user_body += json.dumps(t, indent=4, ensure_ascii=False) + "\n\n"
            user_body += self._content_str(first_user.get("content")).strip()
            emit_text(user_body, user_idx)
            emit_special(self._eot, user_idx)
            i += 1

        # 2b. Remaining messages — plain user/assistant/tool/assistant-with-tool-calls.
        for j in range(i, len(body_messages)):
            msg = body_messages[j]
            msg_idx = j + offset
            role = msg.get("role")
            tool_calls = msg.get("tool_calls")

            if role in ("tool", "ipython"):
                emit_special(self._start_header, msg_idx)
                emit_text("ipython", msg_idx)
                emit_special(self._end_header, msg_idx)
                emit_text(
                    "\n\n" + self._tool_response_str(msg.get("content")),
                    msg_idx,
                )
                emit_special(self._eot, msg_idx)
            elif tool_calls:
                if len(tool_calls) != 1:
                    raise ValueError(
                        "Llama-3 chat template only supports a single tool call "
                        "per assistant message."
                    )
                tc = tool_calls[0]
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                if isinstance(arguments, str):
                    args_str = arguments
                else:
                    args_str = json.dumps(arguments, ensure_ascii=False)
                emit_special(self._start_header, msg_idx)
                emit_text("assistant", msg_idx)
                emit_special(self._end_header, msg_idx)
                emit_text(
                    '\n\n{"name": "' + name + '", "parameters": ' + args_str + "}",
                    msg_idx,
                )
                emit_special(self._eot, msg_idx)
            else:
                content = self._content_str(msg.get("content")).strip()
                emit_special(self._start_header, msg_idx)
                emit_text(role or "", msg_idx)
                emit_special(self._end_header, msg_idx)
                emit_text("\n\n" + content, msg_idx)
                emit_special(self._eot, msg_idx)

        # ── 3. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._start_header, -1)
            emit_text("assistant", -1)
            emit_special(self._end_header, -1)
            emit_text("\n\n", -1)

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
        return parse_llama_3(
            self._tokenizer,
            token_ids,
            stop_ids={self._eot, self._end_of_text, self._eom},
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._eot, self._end_of_text, self._eom]

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
    ) -> list[int] | None:
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._eot, self._end_of_text, self._eom},
            synthesize_close=self._eot,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []

        def emit_special(token_id: int, _msg_idx: int = -1) -> None:
            ext.append(token_id)

        def emit_text(text: str, _msg_idx: int = -1) -> None:
            ext.extend(self._encode(text))

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            if role == "system":
                emit_special(self._start_header, i)
                emit_text("system", i)
                emit_special(self._end_header, i)
                emit_text("\n\n" + self._content_str(msg.get("content")).strip(), i)
                emit_special(self._eot, i)
            elif role == "user":
                emit_special(self._start_header, i)
                emit_text("user", i)
                emit_special(self._end_header, i)
                emit_text("\n\n" + self._content_str(msg.get("content")).strip(), i)
                emit_special(self._eot, i)
            elif role in ("tool", "ipython"):
                emit_special(self._start_header, i)
                emit_text("ipython", i)
                emit_special(self._end_header, i)
                emit_text("\n\n" + self._tool_response_str(msg.get("content")), i)
                emit_special(self._eot, i)
            else:
                return None

        # Generation prompt — matches the gen-prompt branch of ``render()``.
        emit_special(self._start_header, -1)
        emit_text("assistant", -1)
        emit_special(self._end_header, -1)
        emit_text("\n\n", -1)

        return previous_ids + ext
