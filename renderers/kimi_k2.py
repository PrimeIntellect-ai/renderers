"""Kimi K2 Renderer — hard-coded Python mirroring the Kimi K2 Jinja chat template.

Key characteristics:
- Role tokens: <|im_user|>, <|im_assistant|>, <|im_system|>
- Separator token: <|im_middle|> between role name and content
- Terminator token: <|im_end|>
- Tool calls wrapped in <|tool_calls_section_begin|>...<|tool_calls_section_end|>
  with individual calls in <|tool_call_begin|>id<|tool_call_argument_begin|>args<|tool_call_end|>
- Tool declaration messages use role="tool_declare" with <|im_system|>tool_declare<|im_middle|>
- Tool results use role="tool" with <|im_system|>{name}<|im_middle|>## Return of {id}\\n
- Thinking uses text tags <think>...</think>; historical messages strip to <think></think>
- Default system message injected if none present
"""

from __future__ import annotations

import json

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import KimiK2RendererConfig
from renderers.parsing import parse_kimi_k2

_DEFAULT_SYSTEM = "You are Kimi, an AI assistant created by Moonshot AI."


class KimiK2Renderer:
    """Deterministic message → token renderer for Kimi K2 models.

    Kimi K2's chat template doesn't read any thinking-related variable —
    ``content`` renders verbatim with no reasoning branch. The
    ``enable_thinking`` / ``thinking_retention`` fields on the config are
    stored for protocol uniformity with the rest of the renderer family
    but have no effect on the byte-level output.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: KimiK2RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or KimiK2RendererConfig()
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config,
            "all",
        )

        self._im_user = self._token_id("<|im_user|>")
        self._im_assistant = self._token_id("<|im_assistant|>")
        self._im_system = self._token_id("<|im_system|>")
        self._im_middle = self._token_id("<|im_middle|>")
        self._im_end = self._token_id("<|im_end|>")
        self._tool_calls_section_begin = self._token_id("<|tool_calls_section_begin|>")
        self._tool_calls_section_end = self._token_id("<|tool_calls_section_end|>")
        self._tool_call_begin = self._token_id("<|tool_call_begin|>")
        self._tool_call_argument_begin = self._token_id("<|tool_call_argument_begin|>")
        self._tool_call_end = self._token_id("<|tool_call_end|>")

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

    def _ensure_system_message(
        self, messages: list[Message]
    ) -> tuple[list[Message], int]:
        """Prepend default system message if none present.

        Returns ``(messages, auto_injected_idx)``. ``auto_injected_idx`` is
        the index of the auto-injected system message in the returned list,
        or ``-1`` if no injection happened. The Jinja template emits a
        literal ``\\n`` after the auto-injected system's ``<|im_end|>``
        (but not after a user-supplied system), so the caller uses this
        index to replicate that.

        - If messages is empty: return list with just the default system message.
        - If first message is tool_declare and no system follows: insert default
          system after tool_declare.
        - If first message is not system (and not tool_declare): prepend default.
        - Otherwise: return unchanged.
        """
        if not messages:
            return [{"role": "system", "content": _DEFAULT_SYSTEM}], 0

        first_role = messages[0].get("role")
        if first_role == "tool_declare":
            if len(messages) >= 2 and messages[1].get("role") == "system":
                return messages, -1
            default_sys: Message = {"role": "system", "content": _DEFAULT_SYSTEM}
            return [messages[0], default_sys] + list(messages[1:]), 1
        elif first_role != "system":
            default_sys = {"role": "system", "content": _DEFAULT_SYSTEM}
            return [default_sys] + list(messages), 0

        return messages, -1

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        # Preserve the caller's list — ``message_roles`` and per-token
        # attribution refer to this frame (not the post-normalisation
        # list that includes auto-injected system / tool_declare).
        caller_messages = list(messages)

        # Inject tools as tool_declare message + ensure system message.
        # The Jinja template emits the tools list directly (no
        # ``{"type":"function","function":...}`` wrapper) using
        # ``tojson(separators=(',', ':'))``, which produces compact JSON
        # with sorted keys — match both to stay byte-identical.
        if tools:
            tools_json = json.dumps(
                list(tools), separators=(",", ":"), sort_keys=True, ensure_ascii=False
            )
            tool_declare_msg: Message = {
                "role": "tool_declare",
                "content": tools_json,
            }
            # Prepend tool_declare if not already present
            if messages[0].get("role") != "tool_declare":
                messages = [tool_declare_msg] + list(messages)
                tool_declare_injected = True
            else:
                tool_declare_injected = False
            # else leave as-is (caller already included tool_declare)
        else:
            tool_declare_injected = False

        messages, auto_system_idx = self._ensure_system_message(messages)

        # Map indices in the (possibly-normalised) ``messages`` list back to
        # the caller's original list. Injected system / tool_declare get
        # msg_idx=-1 (sentinel) so build_training_sample can't dereference
        # past the caller's input length.
        injected_positions = set()
        if tool_declare_injected:
            injected_positions.add(0)
        if auto_system_idx >= 0:
            injected_positions.add(auto_system_idx)
        n_injected_before = [0] * (len(messages) + 1)
        for k in range(len(messages)):
            n_injected_before[k + 1] = n_injected_before[k] + (
                1 if k in injected_positions else 0
            )

        def orig_idx(i: int) -> int:
            if i in injected_positions:
                return -1
            return i - n_injected_before[i]

        token_ids: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit_ids(
            ids: list[int], msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            token_ids.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_special(
            token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            token_ids.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(
            text: str, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            emit_ids(
                self._encode(text),
                msg_idx,
                is_sampled=is_sampled,
                is_content=is_content,
            )

        # Compute last non-tool-call assistant index to determine thinking preservation
        last_plain_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant" and not messages[i].get("tool_calls"):
                last_plain_assistant_idx = i
                break

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if not isinstance(content, str):
                # Flatten list content to text
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif part.get("type") == "thinking":
                            parts.append(
                                "<think>" + part.get("thinking", "") + "</think>"
                            )
                    elif isinstance(part, str):
                        parts.append(part)
                content = "".join(parts)

            oi = orig_idx(i)
            # Auto-injected system / tool_declare messages have ``oi == -1``.
            # Their text isn't from the caller's input, so we treat the
            # whole emission as scaffold (``is_content=False`` everywhere).
            # The test contract is that ``msg_idx == -1`` runs are
            # template-only and ``is_content=False``.
            body_is_content = oi >= 0

            if role == "system":
                emit_special(self._im_system, oi, is_sampled=False, is_content=False)
                emit_text("system", oi, is_sampled=False, is_content=False)
                emit_special(self._im_middle, oi, is_sampled=False, is_content=False)
                emit_text(content, oi, is_sampled=False, is_content=body_is_content)
                emit_special(self._im_end, oi, is_sampled=False, is_content=False)
                # Jinja emits a literal newline only after the auto-injected
                # system's <|im_end|> (see _ensure_system_message's contract).
                if i == auto_system_idx:
                    emit_text("\n", oi, is_sampled=False, is_content=False)

            elif role == "tool_declare":
                # The tool_declare body is the tools JSON — recoverable
                # from the caller's ``tools`` argument, so we treat it as
                # scaffold (consistent with Qwen3's tools-header block).
                emit_special(self._im_system, oi, is_sampled=False, is_content=False)
                emit_text("tool_declare", oi, is_sampled=False, is_content=False)
                emit_special(self._im_middle, oi, is_sampled=False, is_content=False)
                emit_text(content, oi, is_sampled=False, is_content=False)
                emit_special(self._im_end, oi, is_sampled=False, is_content=False)

            elif role == "user":
                emit_special(self._im_user, oi, is_sampled=False, is_content=False)
                emit_text("user", oi, is_sampled=False, is_content=False)
                emit_special(self._im_middle, oi, is_sampled=False, is_content=False)
                emit_text(content, oi, is_sampled=False, is_content=body_is_content)
                emit_special(self._im_end, oi, is_sampled=False, is_content=False)

            elif role == "assistant":
                # Kimi strips reasoning from historical assistant turns and
                # only keeps it for the most-recent plain assistant. Off-by-one
                # here would drop reasoning from the last turn too.
                is_last_turn = (
                    last_plain_assistant_idx == -1 or i >= last_plain_assistant_idx
                )
                self._render_assistant(
                    msg,
                    oi,
                    content,
                    is_last_turn=is_last_turn,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                self._render_tool(
                    msg,
                    oi,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            else:
                # Unknown role: use system-style formatting. Not a sampled
                # assistant turn — every token is template-injected from the
                # caller's POV, so is_sampled=False across the whole emission.
                emit_special(self._im_system, oi, is_sampled=False, is_content=False)
                emit_text(role, oi, is_sampled=False, is_content=False)
                emit_special(self._im_middle, oi, is_sampled=False, is_content=False)
                emit_text(content, oi, is_sampled=False, is_content=body_is_content)
                emit_special(self._im_end, oi, is_sampled=False, is_content=False)

        # Generation prompt
        if add_generation_prompt:
            emit_special(self._im_assistant, -1, is_sampled=False, is_content=False)
            emit_text("assistant", -1, is_sampled=False, is_content=False)
            emit_special(self._im_middle, -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=token_ids,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in caller_messages],
            message_tool_names=extract_message_tool_names(caller_messages),
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
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — section-JSON wire format quotes strings, schema not needed
    ) -> ParsedResponse:
        return parse_kimi_k2(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end},
            tool_calls_section_begin_id=self._tool_calls_section_begin,
            tool_calls_section_end_id=self._tool_calls_section_end,
            tool_call_begin_id=self._tool_call_begin,
            tool_call_argument_begin_id=self._tool_call_argument_begin,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end]

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
            self.effective_thinking_retention,
            new_messages,
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._im_end},
            synthesize_close=self._im_end,
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
        # and read each step's own body mask. Downstream consumers can
        # run :meth:`RenderedTokens.tokens_per_message` on the bridge
        # output to get per-new-message token counts without re-rendering.
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

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = msg.get("content") or ""
            if not isinstance(content, str):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif part.get("type") == "thinking":
                            parts.append(
                                "<think>" + part.get("thinking", "") + "</think>"
                            )
                    elif isinstance(part, str):
                        parts.append(part)
                content = "".join(parts)

            if role == "user":
                emit_special(self._im_user, i)
                emit_text("user", i)
                emit_special(self._im_middle, i)
                emit_text(content, i, is_content=True)
                emit_special(self._im_end, i)
            elif role == "system":
                emit_special(self._im_system, i)
                emit_text("system", i)
                emit_special(self._im_middle, i)
                emit_text(content, i, is_content=True)
                emit_special(self._im_end, i)
            elif role == "tool":
                self._render_tool(
                    msg,
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )
            else:
                return None

        # Generation prompt.
        emit_special(self._im_assistant, -1)
        emit_text("assistant", -1)
        emit_special(self._im_middle, -1)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        is_last_turn: bool,
        emit_special,
        emit_text,
    ) -> None:
        # ``<|im_assistant|>assistant<|im_middle|>`` is template-injected
        # scaffolding — at inference the chat template emits these as the
        # generation prompt and the model never samples them. Marking the
        # role tag as ``is_sampled=False`` keeps the SFT loss mask aligned
        # with what the model would actually have produced. ``is_content``
        # is also False here — the role tag isn't part of any message's
        # body, on any role.
        emit_special(self._im_assistant, msg_idx, is_sampled=False, is_content=False)
        emit_text("assistant", msg_idx, is_sampled=False, is_content=False)
        emit_special(self._im_middle, msg_idx, is_sampled=False, is_content=False)

        # Kimi K2's Jinja template has no reasoning-content support: the
        # assistant turn renders its ``content`` verbatim, including any
        # inline ``<think>...</think>`` tags. The separate
        # ``reasoning_content`` field is dropped (the template never reads
        # it). ``is_last_turn`` is unused here for the same reason.
        # On assistant tokens, ``is_content == sampled_mask`` by construction
        # — every sampled token is body, every scaffold token isn't.
        _ = is_last_turn
        emit_text(content, msg_idx, is_sampled=True, is_content=True)

        # Tool calls — model-sampled markup carrying caller / model body.
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            emit_special(
                self._tool_calls_section_begin,
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
            for tc in tool_calls:
                func = tc.get("function") or tc
                arguments = func.get("arguments", {})
                args_str = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                # The Jinja template emits ``tool_call['id']`` verbatim —
                # empty when missing. Round-trip parseability requires the
                # caller to provide an id in ``functions.{name}:{idx}`` form
                # (that's where the Kimi parser recovers the function name).
                tc_id = tc.get("id") or ""
                emit_special(
                    self._tool_call_begin,
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
                emit_text(tc_id, msg_idx, is_sampled=True, is_content=True)
                emit_special(
                    self._tool_call_argument_begin,
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
                emit_text(args_str, msg_idx, is_sampled=True, is_content=True)
                emit_special(
                    self._tool_call_end,
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
            emit_special(
                self._tool_calls_section_end,
                msg_idx,
                is_sampled=True,
                is_content=True,
            )

        # ``<|im_end|>`` is the model's stop signal — it samples this to
        # end its turn, so it is part of the sampled stream (and the
        # assistant's body).
        emit_special(self._im_end, msg_idx, is_sampled=True, is_content=True)

    def _render_tool(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False. The ``content``
        # field's body bytes get ``is_content=True``; everything else —
        # the ``<|im_system|>name<|im_middle|>`` wrap, the ``## Return of
        # …\n`` header (template-synthesised, not part of the body) —
        # is scaffold so the SFT mask for tool body never trains the
        # model to emit them.
        #
        # We keep the original kimi_k2 emit boundaries — the header and
        # the content are encoded separately, which preserves the
        # template's byte-identity since the original code also emitted
        # them as separate ``encode`` calls.
        name = msg.get("name") or "tool"
        tool_call_id = msg.get("tool_call_id") or ""

        emit_special(self._im_system, msg_idx, is_sampled=False, is_content=False)
        emit_text(name, msg_idx, is_sampled=False, is_content=False)
        emit_special(self._im_middle, msg_idx, is_sampled=False, is_content=False)
        emit_text(
            f"## Return of {tool_call_id}\n",
            msg_idx,
            is_sampled=False,
            is_content=False,
        )
        emit_text(content, msg_idx, is_sampled=False, is_content=True)
        emit_special(self._im_end, msg_idx, is_sampled=False, is_content=False)
