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
    attribute_text_segments,
    reject_assistant_in_extension,
)
from renderers.configs import LagunaXS2RendererConfig
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
        config: LagunaXS2RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or LagunaXS2RendererConfig()

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
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        emit_special(self._eos, -1, is_sampled=False, is_content=False)

        # ── System header (absorbs messages[0] if it's a system message) ──
        system_content = _DEFAULT_SYSTEM_MESSAGE
        system_msg_idx = -1
        caller_has_system = bool(messages and messages[0].get("role") == "system")
        if caller_has_system:
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
            emit_text(
                "<system>\n\n" if has_sys_content else "<system>\n",
                -1,
                is_sampled=False,
                is_content=False,
            )
            if has_sys_content:
                # If the caller provided system content, it's body bytes;
                # otherwise this is the default system prompt (scaffold).
                sys_is_content = caller_has_system
                emit_text(
                    system_content.rstrip(),
                    system_msg_idx,
                    is_sampled=False,
                    is_content=sys_is_content,
                )
            if tools:
                tool_text = _TOOLS_HEADER
                for tool in tools:
                    tool_text += json.dumps(tool, ensure_ascii=False) + "\n"
                tool_text += (
                    _TOOLS_FOOTER_THINKING
                    if self.config.enable_thinking
                    else _TOOLS_FOOTER_NO_THINKING
                )
                emit_text(tool_text, -1, is_sampled=False, is_content=False)
            emit_text("\n</system>\n", -1, is_sampled=False, is_content=False)

        # ── Per-message loop ──────────────────────────────────────────
        for i, msg in enumerate(messages):
            content = self._visible_text(msg.get("content"))

            match msg["role"]:
                case "system":
                    # Already consumed in the header block.
                    if i == 0:
                        continue
                    # Body = caller's content; the ``<system>...</system>``
                    # wrap and surrounding ``\n``s are scaffold.
                    sys_segs: list[tuple[str, bool]] = [("<system>\n", False)]
                    if content:
                        sys_segs.append((content, True))
                    sys_segs.append(("\n</system>\n", False))
                    emit_text_segments(sys_segs, i, is_sampled=False)
                case "user":
                    user_segs: list[tuple[str, bool]] = [("<user>\n", False)]
                    if content:
                        user_segs.append((content, True))
                    user_segs.append(("\n</user>\n", False))
                    emit_text_segments(user_segs, i, is_sampled=False)
                case "assistant":
                    self._render_assistant(
                        msg,
                        i,
                        content,
                        emit_special=emit_special,
                        emit_text=emit_text,
                        emit_text_segments=emit_text_segments,
                    )
                case "tool":
                    tool_segs: list[tuple[str, bool]] = [("<tool_response>\n", False)]
                    if content:
                        tool_segs.append((content, True))
                    tool_segs.append(("\n</tool_response>\n", False))
                    emit_text_segments(tool_segs, i, is_sampled=False)

        # ── Generation prompt ─────────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._assistant, -1, is_sampled=False, is_content=False)
            emit_text("\n", -1, is_sampled=False, is_content=False)
            if self.config.enable_thinking:
                emit_special(self._think, -1, is_sampled=False, is_content=False)
            else:
                emit_special(self._think_end, -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
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
        return parse_laguna_xs2(
            self._tokenizer,
            token_ids,
            stop_ids={self._assistant_end, self._eos},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
            tools=tools,
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
        ext_indices: list[int] = []
        ext_sampled: list[bool] = []
        ext_content: list[bool] = []

        # Bridge populates ``message_indices`` (relative to ``new_messages``)
        # and ``sampled_mask`` (uniformly ``False`` — every token the
        # bridge emits is template scaffolding for the next prompt, not
        # something the model sampled). ``is_content`` follows the same
        # rules as in :meth:`render` so consumers can walk the trajectory
        # and read each step's own body mask.
        def emit_special(
            token_id: int,
            msg_idx: int = -1,
            *,
            is_sampled: bool = False,
            is_content: bool = False,
        ) -> None:
            ext.append(token_id)
            ext_indices.append(msg_idx)
            ext_sampled.append(is_sampled)
            ext_content.append(is_content)

        def emit_text(
            text: str,
            msg_idx: int = -1,
            *,
            is_sampled: bool = False,
            is_content: bool = False,
        ) -> None:
            ids = self._encode(text)
            ext.extend(ids)
            ext_indices.extend([msg_idx] * len(ids))
            ext_sampled.extend([is_sampled] * len(ids))
            ext_content.extend([is_content] * len(ids))

        def emit_text_segments(
            segments: list[tuple[str, bool]],
            msg_idx: int = -1,
            *,
            is_sampled: bool = False,
        ) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                ext.append(tok_id)
                ext_indices.append(msg_idx)
                ext_sampled.append(is_sampled)
                ext_content.append(is_content)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))
            if role == "user":
                segs: list[tuple[str, bool]] = [("<user>\n", False)]
                if content:
                    segs.append((content, True))
                segs.append(("\n</user>\n", False))
                emit_text_segments(segs, i)
            elif role == "system":
                segs = [("<system>\n", False)]
                if content:
                    segs.append((content, True))
                segs.append(("\n</system>\n", False))
                emit_text_segments(segs, i)
            elif role == "tool":
                segs = [("<tool_response>\n", False)]
                if content:
                    segs.append((content, True))
                segs.append(("\n</tool_response>\n", False))
                emit_text_segments(segs, i)
            else:
                return None

        emit_special(self._assistant, -1)
        emit_text("\n", -1)
        if self.config.enable_thinking:
            emit_special(self._think, -1)
        else:
            emit_special(self._think_end, -1)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
        )

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        if self.config.render_assistant_messages_raw:
            self._render_assistant_raw(
                msg_idx,
                content,
                emit_special=emit_special,
                emit_text=emit_text,
            )
            return

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

        # ``<assistant>\n`` is template-injected scaffolding — the chat
        # template emits these as the generation prompt at inference and
        # the model never samples them. Marking the role tag as
        # ``is_sampled=False`` keeps the SFT loss mask aligned with what
        # the model would actually have produced. ``is_content`` is also
        # False on the role tag. On assistant the invariant
        # ``is_content == sampled_mask`` holds.
        emit_special(self._assistant, msg_idx, is_sampled=False, is_content=False)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

        if reasoning_content:
            emit_special(self._think, msg_idx, is_sampled=True, is_content=True)
            emit_text(
                "\n" + reasoning_content.strip() + "\n",
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
        else:
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)

        # Combined newline-after-</think> with optional content. Bundling
        # preserves BPE merges across the boundary.
        post_think_text = "\n"
        if content.strip():
            post_think_text += content.strip() + "\n"
        emit_text(post_think_text, msg_idx, is_sampled=True, is_content=True)

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

            emit_special(self._tool_call, msg_idx, is_sampled=True, is_content=True)
            inner = name + "\n"
            if isinstance(arguments, dict):
                for k, v in arguments.items():
                    inner += "<arg_key>" + k + "</arg_key>\n"
                    if isinstance(v, str):
                        val_text = v
                    else:
                        val_text = json.dumps(v, ensure_ascii=False)
                    inner += "<arg_value>" + val_text + "</arg_value>\n"
            emit_text(inner, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._tool_call_end, msg_idx, is_sampled=True, is_content=True)
            emit_text("\n", msg_idx, is_sampled=True, is_content=True)

        # ``</assistant>`` is the model's stop signal (alongside
        # ``〈|EOS|〉``) — it samples this to end its turn, so it's part of
        # the sampled stream. The trailing ``\n`` is template-appended
        # between turns and never sampled.
        emit_special(self._assistant_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

    def _render_assistant_raw(
        self,
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        """Passthrough assistant rendering matching the Jinja template's
        ``render_assistant_messages_raw`` branch.

        Three pieces, each conditional on the content's own bytes:

        - Open the assistant turn (``<assistant>\\n``) — always.
        - Prepend the gen-prompt prefix (``<think>`` if
          ``enable_thinking``, else ``</think>``) only when ``content``
          doesn't already start with it. This lets callers ship content
          that already includes the prefix (e.g. raw rollouts) without
          duplicating it.
        - Emit ``content`` verbatim. ``</think>`` and ``</assistant>``
          land inside the content as added-vocab specials via the
          tokenizer's default ``split_special_tokens=False`` behaviour,
          matching what ``apply_chat_template`` does when it tokenises
          the rendered string.
        - Append ``\\n</assistant>`` only when ``content`` doesn't end
          with ``</assistant>`` (or ``</assistant>\\n``), then always
          emit the inter-turn ``\\n``.

        Tool calls are deliberately ignored in raw mode — the template
        also ignores ``message.tool_calls`` here. Callers shipping raw
        content are expected to embed any tool-call payload in the
        content string themselves.
        """
        emit_special(self._assistant, msg_idx, is_sampled=False, is_content=False)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

        if self.config.enable_thinking:
            if not content.startswith("<think>"):
                emit_special(self._think, msg_idx, is_sampled=False, is_content=False)
        else:
            if not content.startswith("</think>"):
                emit_special(
                    self._think_end, msg_idx, is_sampled=False, is_content=False
                )

        emit_text(content, msg_idx, is_sampled=True, is_content=True)

        if not (content.endswith("</assistant>\n") or content.endswith("</assistant>")):
            emit_text("\n", msg_idx, is_sampled=False, is_content=False)
            emit_special(self._assistant_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)
