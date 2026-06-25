"""DeepSeek V3 Renderer — hard-coded Python mirroring the DeepSeek V3 Jinja chat template.

Special tokens use fullwidth Unicode vertical bar (｜ = U+FF5C) and underscores
rendered as ▁ (U+2581), e.g. <｜begin▁of▁sentence｜>.

Format:
    <｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>{assistant}<｜end▁of▁sentence｜>

Thinking uses plain text tags <think>...</think> (NOT special tokens).
When enable_thinking=True the generation prompt prefills <think>\\n to trigger reasoning.
"""

from __future__ import annotations

import json

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import DeepSeekV3RendererConfig
from renderers.parsing import parse_deepseek_v3

# Fullwidth vertical bar used in DeepSeek special token names.
_SEP = "\uff5c"  # ｜  (U+FF5C)
# Fullwidth underscore substitute used in DeepSeek special token names.
_US = "\u2581"  # ▁  (U+2581)


def _ds_token(name: str) -> str:
    """Build a DeepSeek special-token string: <｜{name}｜>."""
    return f"<{_SEP}{name}{_SEP}>"


class DeepSeekV3Renderer:
    """Deterministic message → token renderer for DeepSeek-V3 models.

    DeepSeek-V3 is non-reasoning: its chat template has no ``<think>``
    concept — the generation prompt is a bare ``<｜Assistant｜>`` and past
    assistant content is emitted verbatim. The reasoning variant
    (``<think>``-prefilled prompt, history reasoning stripped) lives in
    :class:`renderers.deepseek_r1.DeepSeekR1Renderer`, which subclasses
    this one. ``thinking_retention`` is a no-op here (no reasoning channel),
    stored for protocol uniformity.
    """

    #: Default typed config; the R1 subclass overrides this.
    _config_cls: type = DeepSeekV3RendererConfig
    _implied_thinking_retention = "all"
    #: Generation-prompt reasoning prefill. Empty for V3 (bare
    #: ``<｜Assistant｜>``); the R1 subclass overrides to ``"<think>\n"``.
    _GEN_THINK_PREFILL: str = ""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DeepSeekV3RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or type(self)._config_cls()
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config,
            self._implied_thinking_retention,
        )

        # ── BOS / EOS ────────────────────────────────────────────────
        self._bos = self._get_special_token(f"begin{_US}of{_US}sentence")
        self._eos = self._get_special_token(f"end{_US}of{_US}sentence")

        # ── Role tokens ───────────────────────────────────────────────
        self._user_token = self._get_special_token("User")
        self._assistant_token = self._get_special_token("Assistant")

        # ── Tool call section tokens ──────────────────────────────────
        self._tool_calls_begin = self._get_special_token(f"tool{_US}calls{_US}begin")
        self._tool_calls_end = self._get_special_token(f"tool{_US}calls{_US}end")
        self._tool_call_begin = self._get_special_token(f"tool{_US}call{_US}begin")
        self._tool_call_end = self._get_special_token(f"tool{_US}call{_US}end")
        self._tool_sep = self._get_special_token(f"tool{_US}sep")

        # ── Tool output section tokens ────────────────────────────────
        self._tool_outputs_begin = self._get_special_token(
            f"tool{_US}outputs{_US}begin"
        )
        self._tool_outputs_end = self._get_special_token(f"tool{_US}outputs{_US}end")
        self._tool_output_begin = self._get_special_token(f"tool{_US}output{_US}begin")
        self._tool_output_end = self._get_special_token(f"tool{_US}output{_US}end")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_special_token(self, name: str) -> int:
        """Encode <｜{name}｜> and assert it maps to exactly one token."""
        token_str = _ds_token(name)
        ids = self._tokenizer.encode(token_str, add_special_tokens=False)
        assert len(ids) == 1, f"Expected single token for {token_str!r}, got {ids}"
        return ids[0]

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Public API
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

        def emit_ids(
            ids: list[int], msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

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
            emit_ids(
                self._encode(text),
                msg_idx,
                is_sampled=is_sampled,
                is_content=is_content,
            )

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

        # ── 1. BOS token ─────────────────────────────────────────────
        emit_special(self._bos, -1, is_sampled=False, is_content=False)

        # ── 2. Collect system messages at the start ───────────────────
        # All leading system messages are concatenated with "\n\n" and emitted
        # before the first non-system message (no role token), matching the HF
        # chat template behaviour.
        sys_parts: list[str] = []
        first_non_sys = 0
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                sys_parts.append(str(content))
                first_non_sys += 1
            else:
                break

        if sys_parts:
            # Attribute the concatenated system text to the first system message (index 0).
            # The system content is the caller's body — mark is_content=True.
            emit_text("\n\n".join(sys_parts), 0, is_sampled=False, is_content=True)

        # ── 3. Render non-system messages ─────────────────────────────
        num_messages = len(messages)
        for i in range(first_non_sys, num_messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                # System messages after the initial block — treat as user turns.
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                emit_special(self._user_token, i, is_sampled=False, is_content=False)
                emit_text(str(content), i, is_sampled=False, is_content=True)

            elif role == "user":
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "")
                        if isinstance(p, dict) and p.get("type") == "text"
                        else p.get("text", "")
                        if isinstance(p, dict)
                        else ""
                        for p in content
                    )
                emit_special(self._user_token, i, is_sampled=False, is_content=False)
                emit_text(str(content), i, is_sampled=False, is_content=True)

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    messages,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

            elif role == "tool":
                self._render_tool(
                    messages,
                    i,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

        # ── 4. Generation prompt ──────────────────────────────────────
        if add_generation_prompt:
            # Don't add <｜Assistant｜> after tool outputs — content flows directly.
            last_role = messages[-1]["role"] if messages else None
            if last_role != "tool":
                emit_special(
                    self._assistant_token, -1, is_sampled=False, is_content=False
                )
            if self._GEN_THINK_PREFILL:
                emit_text(
                    self._GEN_THINK_PREFILL, -1, is_sampled=False, is_content=False
                )

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
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — args land in a ```json fence, schema not needed
    ) -> ParsedResponse:
        return parse_deepseek_v3(
            self._tokenizer,
            token_ids,
            stop_ids={self._eos},
            tool_calls_begin_id=self._tool_calls_begin,
            tool_calls_end_id=self._tool_calls_end,
            tool_call_begin_id=self._tool_call_begin,
            tool_call_end_id=self._tool_call_end,
            tool_sep_id=self._tool_sep,
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
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention,
            new_messages,
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

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            content = str(content)

            if role == "user":
                emit_special(self._user_token, i)
                emit_text(content, i, is_content=True)
            elif role == "system":
                # Post-initial system messages render as user turns.
                emit_special(self._user_token, i)
                emit_text(content, i, is_content=True)
            elif role == "tool":
                prev_is_tool = i > 0 and new_messages[i - 1].get("role") == "tool"
                next_is_tool = (
                    i + 1 < len(new_messages)
                    and new_messages[i + 1].get("role") == "tool"
                )
                if not prev_is_tool:
                    emit_special(self._tool_outputs_begin, i)
                emit_special(self._tool_output_begin, i)
                emit_text(content, i, is_content=True)
                emit_special(self._tool_output_end, i)
                if not next_is_tool:
                    emit_special(self._tool_outputs_end, i)
            else:
                return None

        # Generation prompt — skip ``<｜Assistant｜>`` when the prior new
        # message was a tool response (matches render()'s behaviour: tool
        # output flows directly into assistant content).
        last_role = new_messages[-1].get("role") if new_messages else None
        if last_role != "tool":
            emit_special(self._assistant_token, -1)
        if self._GEN_THINK_PREFILL:
            emit_text(self._GEN_THINK_PREFILL, -1)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )

    # ------------------------------------------------------------------
    # Assistant rendering
    # ------------------------------------------------------------------

    def _prepare_assistant_content(self, msg: Message) -> str:
        """Assistant content as the V3 template would emit it: verbatim.

        V3 is non-reasoning — its template emits ``message['content']`` as-is
        and never reads ``reasoning_content``. A structured content list is
        flattened to its ``text`` parts. The R1 subclass overrides this to
        strip ``</think>`` from history.
        """
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return content

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        messages: list[Message],
        *,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        # Determine whether this message follows a tool output sequence.
        # The HF template emits <｜tool▁outputs▁end｜> before the assistant content
        # without a new <｜Assistant｜> token in that case.
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"

        content = self._prepare_assistant_content(msg)
        tool_calls = msg.get("tool_calls") or []

        # ``<｜Assistant｜>`` is template-injected scaffolding — at
        # inference the chat template emits it as the generation prompt
        # and the model never samples it. Marking it ``is_sampled=False``
        # keeps the SFT loss mask aligned with what the model would
        # actually have produced. When the previous message is a tool
        # response, the template skips this token entirely (content
        # flows directly out of ``<｜tool▁outputs▁end｜>``). On assistant
        # the invariant ``is_content == sampled_mask`` holds.
        if not prev_is_tool:
            emit_special(
                self._assistant_token, msg_idx, is_sampled=False, is_content=False
            )

        if not tool_calls:
            emit_text(content, msg_idx, is_sampled=True, is_content=True)
        else:
            # Emit any pre-tool-call content first.
            emit_text(content, msg_idx, is_sampled=True, is_content=True)

            # Tool call section.
            emit_special(
                self._tool_calls_begin, msg_idx, is_sampled=True, is_content=True
            )
            for tc in tool_calls:
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                # Format: <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{args}\n```<｜tool▁call▁end｜>
                # tool_sep is a special token; type ("function") and name+args are plain text.
                emit_special(
                    self._tool_call_begin, msg_idx, is_sampled=True, is_content=True
                )
                emit_text("function", msg_idx, is_sampled=True, is_content=True)
                emit_special(self._tool_sep, msg_idx, is_sampled=True, is_content=True)
                emit_text(
                    f"{name}\n```json\n{args_str}\n```",
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
                emit_special(
                    self._tool_call_end, msg_idx, is_sampled=True, is_content=True
                )
            emit_special(
                self._tool_calls_end, msg_idx, is_sampled=True, is_content=True
            )

        # ``<｜end▁of▁sentence｜>`` is the model's stop signal — it
        # samples this to end its turn, so it is part of the sampled
        # stream.
        emit_special(self._eos, msg_idx, is_sampled=True, is_content=True)

    # ------------------------------------------------------------------
    # Tool (tool-response) rendering
    # ------------------------------------------------------------------

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        *,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False. The ``content``
        # body bytes get ``is_content=True``; the surrounding section
        # specials are scaffold.
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )

        content = messages[msg_idx].get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

        if not prev_is_tool:
            emit_special(
                self._tool_outputs_begin, msg_idx, is_sampled=False, is_content=False
            )

        emit_special(
            self._tool_output_begin, msg_idx, is_sampled=False, is_content=False
        )
        emit_text(str(content), msg_idx, is_sampled=False, is_content=True)
        emit_special(self._tool_output_end, msg_idx, is_sampled=False, is_content=False)

        if not next_is_tool:
            emit_special(
                self._tool_outputs_end, msg_idx, is_sampled=False, is_content=False
            )
