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
    attribute_text_segments,
    extract_message_tool_names,
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
        # ``preserve_*_thinking`` are accepted but no-ops: Llama-3 ships no
        # reasoning_content channel, so there's never any past-assistant
        # thinking to retain or drop. The flags are stored on ``self.config``
        # for cross-renderer uniformity but never change the token stream —
        # the same contract as Kimi-K2 / Qwen3-VL (see the never-preserves
        # renderers in tests/test_preserve_thinking.py).
        self._tokenizer = tokenizer
        self.config = config or Llama3RendererConfig()

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

        def emit_text_segments(
            segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool
        ) -> None:
            """Tokenize concatenated wrap + body as one BPE pass; per-token
            ``is_content`` follows each token's source segment. Lets the
            scaffold/body split stay attributed without splitting the
            encode call (which could shift BPE merges at the boundary)."""
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        # ── 0. BOS ──────────────────────────────────────────────────
        emit_special(self._bos, -1, is_sampled=False, is_content=False)

        # ── 1. System block (always emitted) ────────────────────────
        first_is_system = messages[0].get("role") == "system"
        sys_idx = 0 if first_is_system else -1
        sys_text = (
            self._content_str(messages[0].get("content")).strip()
            if first_is_system
            else ""
        )

        emit_special(self._start_header, sys_idx, is_sampled=False, is_content=False)
        emit_text("system", sys_idx, is_sampled=False, is_content=False)
        emit_special(self._end_header, sys_idx, is_sampled=False, is_content=False)
        # The Cutting Knowledge / Today Date preamble (and any tools-in-system
        # block) is template scaffold; only the caller's system content is
        # body. Route both through one BPE pass so the wrap/body boundary
        # can't shift merges.
        preamble = "\n\n"
        if tools is not None:
            preamble += "Environment: ipython\n"
        preamble += f"Cutting Knowledge Date: {_CUTTING_KNOWLEDGE_DATE}\n"
        preamble += f"Today Date: {self.config.date_string}\n\n"
        if tools is not None and not self.config.tools_in_user_message:
            preamble += _TOOLS_IN_SYSTEM_INTRO
            for t in tools:
                preamble += json.dumps(t, indent=4, ensure_ascii=False) + "\n\n"
        sys_segments: list[tuple[str, bool]] = [(preamble, False)]
        if sys_text:
            sys_segments.append((sys_text, True))
        emit_text_segments(sys_segments, sys_idx, is_sampled=False)
        emit_special(self._eot, sys_idx, is_sampled=False, is_content=False)

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
            emit_special(
                self._start_header, user_idx, is_sampled=False, is_content=False
            )
            emit_text("user", user_idx, is_sampled=False, is_content=False)
            emit_special(self._end_header, user_idx, is_sampled=False, is_content=False)
            user_preamble = "\n\n" + _TOOLS_IN_USER_INTRO
            for t in tools:
                user_preamble += json.dumps(t, indent=4, ensure_ascii=False) + "\n\n"
            user_content = self._content_str(first_user.get("content")).strip()
            user_segments: list[tuple[str, bool]] = [(user_preamble, False)]
            if user_content:
                user_segments.append((user_content, True))
            emit_text_segments(user_segments, user_idx, is_sampled=False)
            emit_special(self._eot, user_idx, is_sampled=False, is_content=False)
            i += 1

        # 2b. Remaining messages — plain user/assistant/tool/assistant-with-tool-calls.
        for j in range(i, len(body_messages)):
            msg = body_messages[j]
            msg_idx = j + offset
            role = msg.get("role")
            tool_calls = msg.get("tool_calls")

            if role in ("tool", "ipython"):
                # Tool responses are conversation history the model never
                # samples; the response body is caller content, the wrap is
                # scaffold.
                emit_special(
                    self._start_header, msg_idx, is_sampled=False, is_content=False
                )
                emit_text("ipython", msg_idx, is_sampled=False, is_content=False)
                emit_special(
                    self._end_header, msg_idx, is_sampled=False, is_content=False
                )
                tool_body = self._tool_response_str(msg.get("content"))
                tool_segments: list[tuple[str, bool]] = [("\n\n", False)]
                if tool_body:
                    tool_segments.append((tool_body, True))
                emit_text_segments(tool_segments, msg_idx, is_sampled=False)
                emit_special(self._eot, msg_idx, is_sampled=False, is_content=False)
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
                emit_special(
                    self._start_header, msg_idx, is_sampled=False, is_content=False
                )
                emit_text("assistant", msg_idx, is_sampled=False, is_content=False)
                emit_special(
                    self._end_header, msg_idx, is_sampled=False, is_content=False
                )
                # The ``\n\n`` after the header is gen-prompt scaffold the
                # model never samples; the JSON tool-call body and the
                # closing ``<|eot_id|>`` are the model's sampled emission.
                emit_text("\n\n", msg_idx, is_sampled=False, is_content=False)
                emit_text(
                    '{"name": "' + name + '", "parameters": ' + args_str + "}",
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
                emit_special(self._eot, msg_idx, is_sampled=True, is_content=True)
            elif role == "assistant":
                content = self._content_str(msg.get("content")).strip()
                emit_special(
                    self._start_header, msg_idx, is_sampled=False, is_content=False
                )
                emit_text("assistant", msg_idx, is_sampled=False, is_content=False)
                emit_special(
                    self._end_header, msg_idx, is_sampled=False, is_content=False
                )
                # ``\n\n`` separator is scaffold (it's the generation prompt);
                # the body and the closing ``<|eot_id|>`` are model-sampled.
                emit_text("\n\n", msg_idx, is_sampled=False, is_content=False)
                if content:
                    emit_text(content, msg_idx, is_sampled=True, is_content=True)
                emit_special(self._eot, msg_idx, is_sampled=True, is_content=True)
            else:
                # user / non-leading system: caller content, never sampled.
                content = self._content_str(msg.get("content")).strip()
                emit_special(
                    self._start_header, msg_idx, is_sampled=False, is_content=False
                )
                emit_text(role or "", msg_idx, is_sampled=False, is_content=False)
                emit_special(
                    self._end_header, msg_idx, is_sampled=False, is_content=False
                )
                segments: list[tuple[str, bool]] = [("\n\n", False)]
                if content:
                    segments.append((content, True))
                emit_text_segments(segments, msg_idx, is_sampled=False)
                emit_special(self._eot, msg_idx, is_sampled=False, is_content=False)

        # ── 3. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._start_header, -1, is_sampled=False, is_content=False)
            emit_text("assistant", -1, is_sampled=False, is_content=False)
            emit_special(self._end_header, -1, is_sampled=False, is_content=False)
            emit_text("\n\n", -1, is_sampled=False, is_content=False)

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
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
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
    ) -> RenderedTokens | None:
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
        ext_indices: list[int] = []
        ext_content: list[bool] = []

        # Every token the bridge emits is template scaffolding for the next
        # prompt — none of it is model-sampled — so ``sampled_mask`` is
        # uniformly ``False`` (applied over the whole sequence at return).
        # ``is_content`` follows the same rules as :meth:`render` so a
        # consumer can walk the trajectory and read each step's body mask.
        def emit_special(token_id: int, msg_idx: int = -1) -> None:
            ext.append(token_id)
            ext_indices.append(msg_idx)
            ext_content.append(False)

        def emit_text(text: str, msg_idx: int = -1) -> None:
            ids = self._encode(text)
            ext.extend(ids)
            ext_indices.extend([msg_idx] * len(ids))
            ext_content.extend([False] * len(ids))

        def emit_text_segments(
            segments: list[tuple[str, bool]], msg_idx: int = -1
        ) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                ext.append(tok_id)
                ext_indices.append(msg_idx)
                ext_content.append(is_content)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            if role in ("system", "user"):
                content = self._content_str(msg.get("content")).strip()
                emit_special(self._start_header, i)
                emit_text(role, i)
                emit_special(self._end_header, i)
                segs: list[tuple[str, bool]] = [("\n\n", False)]
                if content:
                    segs.append((content, True))
                emit_text_segments(segs, i)
                emit_special(self._eot, i)
            elif role in ("tool", "ipython"):
                tool_body = self._tool_response_str(msg.get("content"))
                emit_special(self._start_header, i)
                emit_text("ipython", i)
                emit_special(self._end_header, i)
                tool_segs: list[tuple[str, bool]] = [("\n\n", False)]
                if tool_body:
                    tool_segs.append((tool_body, True))
                emit_text_segments(tool_segs, i)
                emit_special(self._eot, i)
            else:
                return None

        # Generation prompt — matches the gen-prompt branch of ``render()``.
        emit_special(self._start_header, -1)
        emit_text("assistant", -1)
        emit_special(self._end_header, -1)
        emit_text("\n\n", -1)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )
