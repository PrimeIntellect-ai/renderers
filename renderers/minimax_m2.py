"""MiniMax M2.5 Renderer — hard-coded Python mirroring the MiniMax M2.5 Jinja chat template.

Unique characteristics:
- Token format: ]~!b[ (BOS), ]~b] (role prefix), [e~[ (EOS)
- Role "assistant" rendered as "ai"
- Default system message injected if none provided
- Tool calls use <minimax:tool_call>/<invoke>/<parameter> XML format
- Tool responses wrapped in <response> tags (regular text, not special tokens)
- Thinking only for assistant messages after last user turn
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
    reject_assistant_in_extension,
    should_preserve_past_thinking,
    trim_to_turn_close,
)
from renderers.parsing import parse_minimax

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax."
)

_TOOLS_HEADER = (
    "\n\n# Tools\n"
    "You may call one or more tools to assist with the user query.\n"
    "Here are the tools available in JSONSchema format:\n"
    "\n<tools>\n"
)

_TOOLS_FOOTER_PREFIX = "</tools>\n\n"

_TOOLS_INSTRUCTIONS = (
    "When making tool calls, use XML format to invoke tools and pass parameters:\n"
    "\n<minimax:tool_call>\n"
    '<invoke name="tool-name-1">\n'
    '<parameter name="param-key-1">param-value-1</parameter>\n'
    '<parameter name="param-key-2">param-value-2</parameter>\n'
    "...\n"
    "</invoke>\n"
    "</minimax:tool_call>"
)


class MiniMaxM2Renderer:
    """Deterministic message → token renderer for MiniMax M2 / M2.5 models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        default_system: str = _DEFAULT_SYSTEM,
        preserve_all_thinking: bool = False,
        preserve_thinking_between_tool_calls: bool = False,
    ):
        self._tokenizer = tokenizer
        self._default_system = default_system
        self._preserve_all_thinking = preserve_all_thinking
        self._preserve_thinking_between_tool_calls = (
            preserve_thinking_between_tool_calls
        )

        self._bos = self._token_id("]~!b[")
        self._role = self._token_id("]~b]")
        self._eos = self._token_id("[e~[")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tool_call_tok = self._token_id("<minimax:tool_call>")
        self._tool_call_end_tok = self._token_id("</minimax:tool_call>")

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
    def _visible_text(content: Any) -> str:
        if content is None:
            return "None"
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content)

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

        def emit_token_overlap_body(
            full_text: str,
            body_start: int,
            body_end: int,
            msg_idx: int,
            *,
            is_sampled: bool,
        ) -> None:
            """Tokenize ``full_text`` and mark tokens that overlap the body
            char span as ``is_content=True``.

            Differs from :func:`attribute_text_segments` only in the
            boundary-token rule: a token straddling scaffold→body gets
            ``True`` if any of its bytes are body bytes (overlap rule),
            rather than being attributed to whichever segment its first
            char belongs to. The body's first byte is preserved even when
            BPE merges it with the wrap's trailing byte (``>The`` →
            single token).
            """
            from renderers.base import _get_offset_tokenizer

            offset_tok = _get_offset_tokenizer(self._tokenizer)
            encoding = offset_tok(
                full_text, add_special_tokens=False, return_offsets_mapping=True
            )
            for tok_id, (start, end) in zip(
                encoding["input_ids"], encoding["offset_mapping"]
            ):
                overlaps = start < body_end and end > body_start
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(overlaps)

        # ── Extract system message ──────────────────────────────────
        first_is_system = messages[0].get("role") == "system"
        sys_idx = 0 if first_is_system else -1
        conversation: list[Message] = messages[1:] if first_is_system else messages

        # ── System block (always present) ───────────────────────────
        emit_special(self._bos, sys_idx, is_sampled=False, is_content=False)
        emit_special(self._role, sys_idx, is_sampled=False, is_content=False)

        sys_content = (
            self._visible_text(messages[0].get("content")) if first_is_system else ""
        )
        # Body = caller's system content (if any). Default system message
        # is template-injected scaffold; tools header / per-tool JSON /
        # footer / instructions are scaffold too (the tools dict is
        # recoverable from the ``tools`` arg).
        sys_segments: list[tuple[str, bool]] = [("system\n", False)]
        if sys_content:
            sys_segments.append((sys_content, True))
        else:
            sys_segments.append((self._default_system, False))

        if tools:
            sys_segments.append((_TOOLS_HEADER, False))
            for tool in tools:
                func = tool.get("function", tool)
                sys_segments.append(
                    (
                        "<tool>" + json.dumps(func, ensure_ascii=False) + "</tool>\n",
                        False,
                    )
                )
            sys_segments.append((_TOOLS_FOOTER_PREFIX, False))
            sys_segments.append((_TOOLS_INSTRUCTIONS, False))

        emit_text_segments(sys_segments, sys_idx, is_sampled=False)
        emit_special(self._eos, sys_idx, is_sampled=False, is_content=False)
        emit_text("\n", sys_idx, is_sampled=False, is_content=False)

        # ── Compute last_user_index (relative to conversation) ──────
        last_ui = -1
        for ci, msg in enumerate(conversation):
            if msg.get("role") == "user":
                last_ui = ci

        # ── Iterate conversation messages ───────────────────────────
        for ci, msg in enumerate(conversation):
            role = msg["role"]
            # Map back to original message index for attribution
            orig_idx = ci + (1 if first_is_system else 0)

            if role == "user":
                emit_special(self._role, orig_idx, is_sampled=False, is_content=False)
                user_content = self._visible_text(msg.get("content"))
                user_segments: list[tuple[str, bool]] = [("user\n", False)]
                if user_content:
                    user_segments.append((user_content, True))
                emit_text_segments(user_segments, orig_idx, is_sampled=False)
                emit_special(self._eos, orig_idx, is_sampled=False, is_content=False)
                emit_text("\n", orig_idx, is_sampled=False, is_content=False)

            elif role == "assistant":
                preserve_thinking = should_preserve_past_thinking(
                    messages,
                    orig_idx,
                    preserve_all_thinking=self._preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self._preserve_thinking_between_tool_calls,
                )
                self._render_assistant(
                    msg,
                    orig_idx,
                    ci,
                    last_ui,
                    preserve_thinking=preserve_thinking,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

            elif role == "tool":
                self._render_tool(
                    conversation,
                    ci,
                    orig_idx,
                    msg,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                    emit_token_overlap_body=emit_token_overlap_body,
                )

        # ── Generation prompt ───────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._role, -1, is_sampled=False, is_content=False)
            emit_text("ai\n", -1, is_sampled=False, is_content=False)
            emit_special(self._think, -1, is_sampled=False, is_content=False)
            emit_text("\n", -1, is_sampled=False, is_content=False)

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
        return parse_minimax(
            self._tokenizer,
            token_ids,
            stop_ids={self._eos},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call_tok,
            tool_call_end_id=self._tool_call_end_tok,
            tools=tools,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._eos]

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
            {self._eos},
            synthesize_close=self._eos,
        )
        if previous_ids is None:
            return None

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

        def emit_token_overlap_body(
            full_text: str,
            body_start: int,
            body_end: int,
            msg_idx: int,
            *,
            is_sampled: bool,
        ) -> None:
            from renderers.base import _get_offset_tokenizer

            offset_tok = _get_offset_tokenizer(self._tokenizer)
            encoding = offset_tok(
                full_text, add_special_tokens=False, return_offsets_mapping=True
            )
            for tok_id, (start, end) in zip(
                encoding["input_ids"], encoding["offset_mapping"]
            ):
                overlaps = start < body_end and end > body_start
                ext.append(tok_id)
                ext_indices.append(msg_idx)
                ext_sampled.append(is_sampled)
                ext_content.append(overlaps)

        # Trailing ``\n`` after the ``[e~[`` turn close — see ``render()``.
        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))
            if role == "user":
                emit_special(self._role, i)
                user_segments: list[tuple[str, bool]] = [("user\n", False)]
                if content:
                    user_segments.append((content, True))
                emit_text_segments(user_segments, i)
                emit_special(self._eos, i)
                emit_text("\n", i)
            elif role == "system":
                emit_special(self._role, i)
                sys_segments: list[tuple[str, bool]] = [("system\n", False)]
                if content:
                    sys_segments.append((content, True))
                emit_text_segments(sys_segments, i)
                emit_special(self._eos, i)
                emit_text("\n", i)
            elif role == "tool":
                self._render_tool(
                    new_messages,
                    i,
                    i,
                    msg,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                    emit_token_overlap_body=emit_token_overlap_body,
                )
            else:
                return None

        # Generation prompt.
        emit_special(self._role, -1)
        emit_text("ai\n", -1)
        emit_special(self._think, -1)
        emit_text("\n", -1)

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
        msg,
        orig_idx,
        conv_idx,
        last_user_index,
        *,
        preserve_thinking: bool = False,
        emit_special,
        emit_text,
        emit_text_segments,
    ):
        content = self._visible_text(msg.get("content"))

        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        elif "</think>" in content:
            before, after = content.split("</think>", 1)
            if "<think>" in before:
                reasoning_content = before.split("<think>")[-1].strip("\n")
            else:
                reasoning_content = before.strip("\n")
            content = after.strip("\n")

        # ``]~b]ai\n`` is template-injected scaffolding — at inference
        # the chat template emits these as the generation prompt and the
        # model never samples them. Marking the role marker and tag as
        # ``is_sampled=False`` keeps the SFT loss mask aligned with what
        # the model would actually have produced. ``is_content`` is also
        # False here — the role tag isn't part of any message's body.
        emit_special(self._role, orig_idx, is_sampled=False, is_content=False)

        # Build the model-sampled portion (think block + content + tool
        # calls). For assistant messages the invariant
        # ``is_content == sampled_mask`` holds — every sampled token is
        # body, every scaffold token isn't.
        tool_calls = msg.get("tool_calls") or []
        emit_thinking = reasoning_content and (
            conv_idx > last_user_index or preserve_thinking
        )

        if emit_thinking:
            # The thinking branch has the ``<think>`` special token
            # immediately after ``ai\n``, which forces a tokenizer
            # boundary — splitting ``ai\n`` (not_sampled) from the
            # ``<think>``-led body (sampled) is BPE-safe.
            emit_text("ai\n", orig_idx, is_sampled=False, is_content=False)
            emit_special(self._think, orig_idx, is_sampled=True, is_content=True)
            emit_text(
                "\n" + reasoning_content + "\n",
                orig_idx,
                is_sampled=True,
                is_content=True,
            )
            emit_special(self._think_end, orig_idx, is_sampled=True, is_content=True)
            # \n\n + content must be contiguous for BPE
            body = "\n\n" + content if content else "\n\n"
        else:
            body = content
            # Empty body + tool_calls would emit ``"\n"`` next, and
            # ``ai\n`` + ``\n`` BPE-merges into a single ``\n\n`` token
            # in this tokenizer. Fold the boundary ``\n`` into the
            # role-tag emission so the merged token stays whole. The
            # combined token is is_sampled=False — the conservative
            # choice for SFT (don't train on a token whose first byte
            # is template scaffolding).
            if tool_calls and not body:
                emit_text("ai\n\n", orig_idx, is_sampled=False, is_content=False)
            else:
                emit_text("ai\n", orig_idx, is_sampled=False, is_content=False)

        if tool_calls:
            # \n before <minimax:tool_call> must be contiguous with preceding text.
            # The empty-body / non-thinking case folded the leading \n
            # into the role-tag emission above; skip it here.
            if emit_thinking or body:
                emit_text(body + "\n", orig_idx, is_sampled=True, is_content=True)
            emit_special(
                self._tool_call_tok, orig_idx, is_sampled=True, is_content=True
            )

            invoke_block = "\n"
            for tc in tool_calls:
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})

                invoke_block += '<invoke name="' + name + '">\n'
                # OpenAI canonical form: arguments is a JSON string. Parse it so the
                # per-argument rendering below still works.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if isinstance(arguments, dict):
                    for arg_name, arg_value in arguments.items():
                        val_str = (
                            arg_value
                            if isinstance(arg_value, str)
                            else json.dumps(arg_value, ensure_ascii=False)
                        )
                        invoke_block += (
                            '<parameter name="'
                            + arg_name
                            + '">'
                            + val_str
                            + "</parameter>\n"
                        )
                invoke_block += "</invoke>\n"

            emit_text(invoke_block, orig_idx, is_sampled=True, is_content=True)
            emit_special(
                self._tool_call_end_tok, orig_idx, is_sampled=True, is_content=True
            )
        elif body:
            emit_text(body, orig_idx, is_sampled=True, is_content=True)

        # ``[e~[`` is the model's stop signal — it samples this to end
        # its turn, so it is part of the sampled stream. The trailing
        # ``\n`` is template-appended between turns and never sampled.
        emit_special(self._eos, orig_idx, is_sampled=True, is_content=True)
        emit_text("\n", orig_idx, is_sampled=False, is_content=False)

    def _render_tool(
        self,
        conversation: list[Message],
        conv_idx: int,
        orig_idx: int,
        msg: Message,
        *,
        emit_special,
        emit_text,
        emit_text_segments,
        emit_token_overlap_body=None,
    ) -> None:
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False. The ``content``
        # body bytes get ``is_content=True``; the surrounding ``<response>``
        # wrap, role tag and separators are scaffold so an SFT mask over
        # tool body never trains the model to emit those.
        prev_is_tool = conv_idx > 0 and conversation[conv_idx - 1]["role"] == "tool"
        next_is_tool = (
            conv_idx + 1 < len(conversation)
            and conversation[conv_idx + 1]["role"] == "tool"
        )

        if not prev_is_tool:
            emit_special(self._role, orig_idx, is_sampled=False, is_content=False)
            emit_text("tool", orig_idx, is_sampled=False, is_content=False)

        content = self._visible_text(msg.get("content"))
        # Leading ``\n`` before ``<response>`` only on the first of a
        # consecutive run — subsequent ones piggyback on the trailing ``\n``
        # emitted below, so BPE can merge ``</response>\n<response>``
        # through a single emit_text call instead of splitting the merge.
        prefix = "" if prev_is_tool else "\n"
        suffix = "\n" if next_is_tool else ""

        # ``<response>`` is plain text with no separator between the
        # closing ``>`` and ``content``'s first byte, so BPE can merge
        # them into a single token (e.g., ``>The``). The shared
        # ``attribute_text_segments`` helper picks the segment of a
        # boundary-spanning token by its *first* char (here scaffold),
        # which would drop the body's leading letter out of the body
        # run. We instead use an "intersects body" rule: any token whose
        # ``[start, end)`` char range overlaps the body span gets
        # ``is_content=True``. A few scaffold bytes (the leading ``>``
        # or trailing ``<``) bleed into the body run, but body bytes are
        # recoverable as a substring of the decoded body span.
        body_text = prefix + "<response>" + content + "</response>" + suffix
        body_start = len(prefix) + len("<response>")
        body_end = body_start + len(content)
        if content and emit_token_overlap_body is not None:
            emit_token_overlap_body(
                body_text, body_start, body_end, orig_idx, is_sampled=False
            )
        else:
            # Empty body or no overlap-aware emitter available — fall back
            # to the standard segments path.
            tool_segments: list[tuple[str, bool]] = []
            if prefix:
                tool_segments.append((prefix, False))
            tool_segments.append(("<response>", False))
            if content:
                tool_segments.append((content, True))
            tool_segments.append(("</response>", False))
            if suffix:
                tool_segments.append((suffix, False))
            emit_text_segments(tool_segments, orig_idx, is_sampled=False)

        if not next_is_tool:
            emit_special(self._eos, orig_idx, is_sampled=False, is_content=False)
            emit_text("\n", orig_idx, is_sampled=False, is_content=False)
