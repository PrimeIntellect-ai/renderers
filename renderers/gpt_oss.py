"""GptOssRenderer — adapter over OpenAI's harmony reference encoder.

Wire format: harmony (channel-based, no BOS). The official renderer ships
in the ``openai-harmony`` package — vLLM uses it directly, so matching
its output guarantees byte-identical tokens with vLLM's serving path
(and with what the model was trained on).

This renderer is a thin adapter that:

  - converts each ``RendererMessage`` to one or more ``openai_harmony``
    messages (assistant tool_calls split into per-call ASSISTANT messages
    with channel=commentary and recipient=functions.<name>; tool results
    routed via TOOL author + recipient=assistant)
  - prepends the canonical SystemContent preamble (model identity,
    knowledge cutoff, current date, reasoning_effort, channel config)
    when ``use_system_prompt`` is set — matches HF's apply_chat_template
    behaviour, which the model was trained against
  - renders one harmony message at a time via ``enc.render(m)`` so each
    token can be attributed to its source caller index. ``enc.render``
    is byte-identical to ``enc.render_conversation`` when concatenated;
    verified empirically.

Special tokens
--------------
<|start|>      message start
<|end|>        message end (non-terminal)
<|return|>     message end (terminal — last assistant turn)
<|call|>       tool call end (terminal)
<|channel|>    followed by channel name
<|message|>    content start
<|constrain|>  followed by constraint (e.g. "json")
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    extract_message_tool_names,
    reject_assistant_in_extension,
    should_preserve_past_thinking,
    trim_to_turn_close,
)
from renderers.configs import GptOssRendererConfig
from renderers.parsing import parse_gpt_oss


def _reasoning_effort(effort: str | None) -> ReasoningEffort:
    if effort is None:
        return ReasoningEffort.MEDIUM
    e = effort.lower()
    if e == "low":
        return ReasoningEffort.LOW
    if e == "medium":
        return ReasoningEffort.MEDIUM
    if e == "high":
        return ReasoningEffort.HIGH
    raise ValueError(f"Unknown reasoning_effort: {effort!r}")


def _content_text(content: Any) -> str:
    """Flatten content (string or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text":
                out.append(p.get("text", ""))
            elif ptype == "thinking":
                out.append(p.get("thinking", ""))
        return "".join(out)
    return str(content)


def _tool_to_description(tool: ToolSpec) -> ToolDescription:
    """Convert an OpenAI-format tool spec to a harmony ToolDescription."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
    else:
        fn = tool
    return ToolDescription.new(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=fn.get("parameters") or {},
    )


def _arguments_to_str(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


class GptOssRenderer:
    """Deterministic message → token renderer for OpenAI gpt-oss (harmony)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: GptOssRendererConfig | None = None,
    ):
        """Initialise the renderer.

        Args:
            tokenizer: HuggingFace tokenizer.
            config: Typed renderer config (see
                :class:`renderers.GptOssRendererConfig`).
        """
        self._tokenizer = tokenizer
        self.config = config or GptOssRendererConfig()
        self._enc: HarmonyEncoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS
        )
        # Materialised harmony-enum form of reasoning_effort.
        self._reasoning_effort_enum = _reasoning_effort(self.config.reasoning_effort)
        # ``conversation_start_date=None`` defers to today's date —
        # materialise once at construction so renders within the same
        # instance use a stable date.
        self._conversation_start_date_resolved = (
            self.config.conversation_start_date or datetime.now().strftime("%Y-%m-%d")
        )

        # Cache special-token IDs for the bridge / generation-prompt path.
        self._start = self._token_id("<|start|>")
        self._end = self._token_id("<|end|>")
        self._return = self._token_id("<|return|>")
        self._call = self._token_id("<|call|>")
        self._channel = self._token_id("<|channel|>")
        self._message = self._token_id("<|message|>")
        self._constrain = self._token_id("<|constrain|>")

    # ── token utilities ──────────────────────────────────────────────────────

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

    def _prefix_content_mask(
        self,
        prefix_tokens: list[int],
        first_system_idx: int | None,
        messages: list[Message],
        tools: list[ToolSpec] | None,
    ) -> list[bool]:
        """Per-token is_content mask over the rendered system+developer prefix.

        Harmony's prefix is one opaque block. The caller's system content
        lands inside the developer message as ``# Instructions\\n\\n{content}``.
        To attribute body bytes back, we render the same prefix with empty
        instructions and diff: the unique-to-with-instructions span is the
        body region. Falls back to all-False (whole prefix scaffold) if
        the caller didn't supply a system message — there's no body to
        attribute in that case.
        """
        n = len(prefix_tokens)
        mask = [False] * n
        if first_system_idx is None:
            return mask
        instructions = _content_text(messages[first_system_idx].get("content"))
        if not instructions:
            return mask

        # Build the same prefix with empty instructions.
        empty_prefix_msgs: list[HarmonyMessage] = []
        if self.config.use_system_prompt:
            sys_content = SystemContent.new().with_reasoning_effort(
                self._reasoning_effort_enum
            )
            sys_content = sys_content.with_conversation_start_date(
                self._conversation_start_date_resolved
            )
            if self.config.knowledge_cutoff is not None:
                sys_content = sys_content.with_knowledge_cutoff(
                    self.config.knowledge_cutoff
                )
            if self.config.model_identity is not None:
                sys_content = sys_content.with_model_identity(
                    self.config.model_identity
                )
            empty_prefix_msgs.append(
                HarmonyMessage.from_role_and_content(Role.SYSTEM, sys_content)
            )
        dev = DeveloperContent.new()
        if tools:
            dev = dev.with_function_tools([_tool_to_description(t) for t in tools])
        empty_prefix_msgs.append(
            HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev)
        )
        try:
            empty_tokens = self._enc.render_conversation(
                Conversation.from_messages(empty_prefix_msgs)
            )
        except Exception:
            return mask

        # Longest common prefix.
        i_start = 0
        n_empty = len(empty_tokens)
        while (
            i_start < min(n, n_empty)
            and prefix_tokens[i_start] == empty_tokens[i_start]
        ):
            i_start += 1
        # Longest common suffix.
        j_full = n
        j_empty = n_empty
        while (
            j_full > i_start
            and j_empty > i_start
            and prefix_tokens[j_full - 1] == empty_tokens[j_empty - 1]
        ):
            j_full -= 1
            j_empty -= 1
        # Tokens [i_start:j_full] in prefix_tokens are unique to the
        # with-instructions render — that's the body span (includes the
        # ``# Instructions\n\n`` scaffolding header, which the substring
        # match in the body-decode test ignores).
        for k in range(i_start, j_full):
            mask[k] = True
        return mask

    # ── public interface ─────────────────────────────────────────────────────

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

        def emit(
            ids: list[int], msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_harmony_message(
            hm_ids: list[int], msg_idx: int, *, is_assistant: bool
        ) -> None:
            """Emit one harmony-rendered message, splitting its tokens into
            template scaffolding vs model-sampled content.

            Harmony per-message layout:
                <|start|> role [' to=' recipient] [<|channel|> name] <|message|>
                <body...>
                <|end|> | <|return|> | <|call|>

            Everything up to and including ``<|message|>`` is the
            generation-prompt header the template injects before the
            model starts sampling. For assistant turns the body and the
            terminator (``<|end|>`` / ``<|return|>`` / ``<|call|>``) are
            what the model actually produces, so they are
            ``is_sampled=True``. For non-assistant turns (user, system,
            developer, tool) the whole message is conversation history
            the model never samples — every token is
            ``is_sampled=False``.

            ``is_content`` further splits the body: the trailing
            terminator (``<|end|>`` / ``<|return|>`` / ``<|call|>``) is
            scaffold on non-assistant turns; on assistant turns it's the
            model's stop signal, so ``is_content=True`` mirrors
            ``sampled_mask`` as the invariant on assistant requires.
            The body bytes between ``<|message|>`` and the terminator
            are body (``is_content=True``) on every role — that's the
            caller-provided content (or, for assistant, the model's
            sampled emission). The header (``<|start|>`` ... ``<|message|>``)
            — including any ``functions.{name}`` recipient text on tool
            results, which comes from a prior assistant's tool_calls
            rather than this tool message's own content — is scaffold.
            """
            try:
                msg_marker = hm_ids.index(self._message)
            except ValueError:
                # Defensive: a harmony message without <|message|> is
                # malformed. Treat the whole thing as scaffolding.
                emit(hm_ids, msg_idx, is_sampled=False, is_content=False)
                return
            header = hm_ids[: msg_marker + 1]
            body = hm_ids[msg_marker + 1 :]
            emit(header, msg_idx, is_sampled=False, is_content=False)
            # Split body into content + terminator. The terminator (if
            # present) is the last token of the body and is one of the
            # three harmony stop tokens.
            terminator_ids = {self._end, self._return, self._call}
            if body and body[-1] in terminator_ids:
                body_content = body[:-1]
                terminator = body[-1:]
            else:
                body_content = body
                terminator = []
            emit(
                body_content,
                msg_idx,
                is_sampled=is_assistant,
                is_content=True,
            )
            if terminator:
                emit(
                    terminator,
                    msg_idx,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )

        # ── Build harmony prefix (system + developer) ───────────────────
        # When tools are present, harmony's conversation-level renderer
        # injects a channel-routing line into the SystemContent block
        # (``Calls to these tools must go to the commentary channel:
        # 'functions'.``). Per-message ``enc.render`` doesn't know about
        # the surrounding tool namespaces, so we render the prefix via
        # ``render_conversation`` to pick that up correctly.
        first_system_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") == "system"),
            None,
        )
        prefix_msgs: list[HarmonyMessage] = []
        if self.config.use_system_prompt:
            sys_content = SystemContent.new().with_reasoning_effort(
                self._reasoning_effort_enum
            )
            sys_content = sys_content.with_conversation_start_date(
                self._conversation_start_date_resolved
            )
            if self.config.knowledge_cutoff is not None:
                sys_content = sys_content.with_knowledge_cutoff(
                    self.config.knowledge_cutoff
                )
            if self.config.model_identity is not None:
                sys_content = sys_content.with_model_identity(
                    self.config.model_identity
                )
            prefix_msgs.append(
                HarmonyMessage.from_role_and_content(Role.SYSTEM, sys_content)
            )
        if first_system_idx is not None or tools:
            dev = DeveloperContent.new()
            if first_system_idx is not None:
                instructions = _content_text(messages[first_system_idx].get("content"))
                if instructions:
                    dev = dev.with_instructions(instructions)
            if tools:
                dev = dev.with_function_tools([_tool_to_description(t) for t in tools])
            prefix_msgs.append(
                HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev)
            )

        if prefix_msgs:
            prefix_tokens = self._enc.render_conversation(
                Conversation.from_messages(prefix_msgs)
            )
            # Attribute the whole prefix block to first_system_idx if the
            # caller supplied a system message (so its content has *some*
            # caller-relative attribution); otherwise to -1 (pure scaffolding).
            # The whole prefix is pure template scaffolding — never sampled.
            prefix_origin = first_system_idx if first_system_idx is not None else -1
            # Compute the body-token span inside the prefix by diffing
            # against the same render with empty developer instructions.
            # Tokens unique to the with-instructions render are the body
            # span (``# Instructions\n\n{caller_system_content}``). Marking
            # those is_content=True so the caller's system text is
            # recoverable from ``content_token_spans_by_role()["system"]``.
            # The scaffolding ``# Instructions\n\n`` prefix bleeds into
            # the body run; consumers reading the body do a substring
            # check rather than expecting an exact match.
            prefix_content_mask = self._prefix_content_mask(
                prefix_tokens, first_system_idx, messages, tools
            )
            for tid, is_content in zip(prefix_tokens, prefix_content_mask):
                tokens.append(tid)
                indices.append(prefix_origin)
                sampled.append(False)
                content_mask.append(is_content)

        # ── Iterate the rest of the messages ────────────────────────────
        last_idx = len(messages) - 1
        for i, msg in enumerate(messages):
            if i == first_system_idx:
                continue  # already emitted as developer
            is_assistant = msg.get("role") == "assistant"
            preserve_thinking = is_assistant and (
                should_preserve_past_thinking(
                    messages,
                    i,
                    preserve_all_thinking=self.config.preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self.config.preserve_thinking_between_tool_calls,
                )
            )
            for hm in self._to_harmony_messages(
                msg, preserve_thinking=preserve_thinking
            ):
                emit_harmony_message(self._enc.render(hm), i, is_assistant=is_assistant)

        # When the conversation ends on an assistant final-channel turn,
        # ``apply_chat_template`` (and ``render_conversation_for_training``)
        # close it with ``<|return|>`` instead of ``<|end|>`` to mark the
        # final assistant turn as terminal. Per-message rendering always
        # uses ``<|end|>``, so patch it here when applicable. The sampled
        # bit on the terminator slot stays True — both ``<|end|>`` and
        # ``<|return|>`` are stop signals the model emits.
        if (
            not add_generation_prompt
            and last_idx >= 0
            and messages[last_idx].get("role") == "assistant"
            and tokens
            and tokens[-1] == self._end
            and not (messages[last_idx].get("tool_calls"))
        ):
            tokens[-1] = self._return

        # ── Generation prompt: <|start|>assistant<|channel|>analysis<|message|>
        # Pure template scaffolding the model continues from — never sampled.
        if add_generation_prompt:
            emit([self._start], -1, is_sampled=False, is_content=False)
            emit(self._encode("assistant"), -1, is_sampled=False, is_content=False)
            emit([self._channel], -1, is_sampled=False, is_content=False)
            emit(self._encode("analysis"), -1, is_sampled=False, is_content=False)
            emit([self._message], -1, is_sampled=False, is_content=False)

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
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — harmony args land in a JSON object, schema not needed
    ) -> ParsedResponse:
        return parse_gpt_oss(
            self._tokenizer,
            token_ids,
            return_id=self._return,
            call_id=self._call,
            start_id=self._start,
            end_id=self._end,
            channel_id=self._channel,
            message_id=self._message,
            constrain_id=self._constrain,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._return, self._call]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        """Per-message harmony bridge.

        Each new message is rendered in isolation via ``enc.render(m)`` —
        same primitive ``render()`` uses — so the bridge can never drift
        from the full re-render.
        """
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._return, self._call},
            synthesize_close=self._end,
        )
        if previous_ids is None:
            return None

        # Bridge populates ``message_indices`` (relative to ``new_messages``)
        # and ``sampled_mask`` (uniformly ``False``). The harmony encoder
        # renders each ``new_messages[i]`` as a single block, so every
        # token in that block carries index ``i``; the trailing
        # generation prompt uses ``-1``. ``is_content`` follows the same
        # rules as :meth:`render`'s ``emit_harmony_message``: header is
        # scaffold, body bytes are body, terminator scaffold (the bridge
        # never carries assistant turns, so terminators are always
        # scaffold on the non-assistant roles the bridge accepts).
        terminator_ids = {self._end, self._return, self._call}
        ext: list[int] = []
        ext_indices: list[int] = []
        ext_content: list[bool] = []
        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            if role not in ("tool", "user", "system", "developer"):
                return None
            for hm in self._to_harmony_messages(msg):
                ids = self._enc.render(hm)
                try:
                    msg_marker = ids.index(self._message)
                except ValueError:
                    # Defensive: treat as scaffolding.
                    ext.extend(ids)
                    ext_indices.extend([i] * len(ids))
                    ext_content.extend([False] * len(ids))
                    continue
                header = ids[: msg_marker + 1]
                body = ids[msg_marker + 1 :]
                ext.extend(header)
                ext_indices.extend([i] * len(header))
                ext_content.extend([False] * len(header))
                if body and body[-1] in terminator_ids:
                    body_content = body[:-1]
                    terminator = body[-1:]
                else:
                    body_content = body
                    terminator = []
                ext.extend(body_content)
                ext_indices.extend([i] * len(body_content))
                ext_content.extend([True] * len(body_content))
                if terminator:
                    ext.extend(terminator)
                    ext_indices.extend([i] * len(terminator))
                    ext_content.extend([False] * len(terminator))

        # Generation prompt: <|start|>assistant<|channel|>analysis<|message|>
        gen_before = len(ext)
        ext.append(self._start)
        ext.extend(self._encode("assistant"))
        ext.append(self._channel)
        ext.extend(self._encode("analysis"))
        ext.append(self._message)
        ext_indices.extend([-1] * (len(ext) - gen_before))
        ext_content.extend([False] * (len(ext) - gen_before))

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )

    # ── message conversion ───────────────────────────────────────────────────

    def _to_harmony_messages(
        self, msg: Message, *, preserve_thinking: bool = False
    ) -> list[HarmonyMessage]:
        """Convert a single RendererMessage to one or more harmony messages.

        Splits in two cases:
          - assistant with reasoning_content + content → analysis message
            then final message
          - assistant with multiple tool_calls → one ASSISTANT message per
            tool_call, with recipient=functions.<name>, channel=commentary
        """
        role = msg.get("role", "")

        if role == "user":
            return [
                HarmonyMessage.from_role_and_content(
                    Role.USER, _content_text(msg.get("content"))
                )
            ]

        if role == "system" or role == "developer":
            # Caller's system message is normally pulled out as developer
            # Instructions in render(); reaching here means it's a
            # secondary system or an explicit developer message. Render
            # as developer with the content as instructions.
            dev = DeveloperContent.new().with_instructions(
                _content_text(msg.get("content"))
            )
            return [HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev)]

        if role == "tool":
            # Tool result. Author needs the function name; we recover it
            # from `name` (set client-side via _attach_tool_call_names) or
            # from the message dict.
            tool_name = msg.get("name") or "unknown"
            if not tool_name.startswith("functions."):
                tool_name = f"functions.{tool_name}"
            content = _content_text(msg.get("content"))
            tm = HarmonyMessage.from_author_and_content(
                {"role": "tool", "name": tool_name}, content
            )
            tm = tm.with_recipient("assistant").with_channel("commentary")
            return [tm]

        if role == "assistant":
            return self._assistant_to_harmony(msg, preserve_thinking=preserve_thinking)

        # Unknown role: render as developer with the raw content.
        dev = DeveloperContent.new().with_instructions(
            _content_text(msg.get("content"))
        )
        return [HarmonyMessage.from_role_and_content(Role.DEVELOPER, dev)]

    def _assistant_to_harmony(
        self, msg: Message, *, preserve_thinking: bool = False
    ) -> list[HarmonyMessage]:
        """Convert an assistant message to harmony messages.

        Layout:
          - text content (if any)      → final channel
          - each tool_call             → commentary channel,
                                         recipient=functions.<name>

        Default: ``reasoning_content`` is NOT emitted — harmony strips
        analysis-channel messages from history-style rendering. Per-turn
        thinking is only relevant for live generation; once a turn closes,
        its analysis block is dropped from context.

        ``preserve_thinking=True``: prepend an analysis-channel message
        carrying ``reasoning_content`` so callers that want the trace in
        history (e.g. tool-call-chain training) see it surface.
        """
        out: list[HarmonyMessage] = []

        if preserve_thinking:
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                analysis = HarmonyMessage.from_role_and_content(
                    Role.ASSISTANT, reasoning
                )
                out.append(analysis.with_channel("analysis"))

        content = msg.get("content")
        text_parts: list[str] = []
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text_parts.append(p.get("text", ""))
                # Thinking parts are dropped (see docstring).
        elif isinstance(content, str):
            text_parts.append(content)
        text = "".join(text_parts)

        if text:
            m = HarmonyMessage.from_role_and_content(Role.ASSISTANT, text)
            m = m.with_channel("final")
            out.append(m)

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function") or tc
            name = fn.get("name", "")
            args = _arguments_to_str(fn.get("arguments", {}))
            recipient = name if name.startswith("functions.") else f"functions.{name}"
            m = HarmonyMessage.from_role_and_content(Role.ASSISTANT, args)
            m = m.with_channel("commentary").with_recipient(recipient)
            out.append(m)

        # Empty assistant (no text and no tool_calls) — emit an empty
        # final-channel message so message_indices stays non-empty.
        if not out:
            m = HarmonyMessage.from_role_and_content(Role.ASSISTANT, "")
            m = m.with_channel("final")
            out.append(m)

        return out
