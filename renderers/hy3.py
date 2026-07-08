"""Hy3 renderer — hard-coded Python mirroring the Tencent Hunyuan Hy3 Jinja
chat template (``tencent/Hy3`` / ``Hy3-FP8``; ``Hy3-preview`` uses an older,
incompatible template and is not supported).

Shape of the Hy3 template, distinct from the GLM / Qwen families:

- No per-message ``<|system|>`` marker: every system message's content is
  concatenated into a single blob emitted right after ``<｜hy_begin_of_sentence｜>``.
- Reasoning is gated by ``reasoning_effort`` (``no_think`` / ``low`` / ``high``),
  not a boolean. Without tools a ``<｜reasoning_mode｜>reasoning_effort:{effort}``
  marker is appended to the system blob; with tools it rides at the end of the
  tool-instruction block instead.
- The generation prompt prefills ``<think></think>`` in ``no_think`` mode (the
  model answers directly) and only ``<think>`` in ``low`` / ``high`` mode (the
  model streams reasoning up to a ``</think>`` it emits itself).
- Each assistant turn closes with an explicit ``<｜hy_eos｜>`` — the sole stop
  token — unlike GLM where the next role marker doubles as the close.
- Tool calls: ``<tool_calls>`` wraps one or more ``<tool_call>name<tool_sep>``
  blocks, each carrying ``<arg_key>``/``<arg_value>`` pairs (single special
  tokens, as in GLM), and the block closes ``</tool_calls><｜hy_eos｜>``.
- Tool responses: ``<tool_responses>`` wraps one or more
  ``<tool_response>…</tool_response>`` blocks.
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
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
)
from renderers.configs import Hy3RendererConfig, ResolvedThinkingRetention
from renderers.parsing import parse_hy3

# Special-token strings, constructed exactly as the Jinja template does
# (``'<｜hy_eos{}｜>'.format(':opensource')`` etc.) so ``convert_tokens_to_ids``
# resolves each to its single vocabulary id.
_HYTK = ":opensource"
_BOS = f"<｜hy_begin_of_sentence{_HYTK}｜>"
_EOS = f"<｜hy_eos{_HYTK}｜>"
_USER = f"<｜hy_User{_HYTK}｜>"
_ASSISTANT = f"<｜hy_Assistant{_HYTK}｜>"
_REASONING_MODE = f"<｜reasoning_mode{_HYTK}｜>"
_THINK = f"<think{_HYTK}>"
_THINK_END = f"</think{_HYTK}>"
_TOOL_CALLS = f"<tool_calls{_HYTK}>"
_TOOL_CALLS_END = f"</tool_calls{_HYTK}>"
_TOOL_CALL = f"<tool_call{_HYTK}>"
_TOOL_CALL_END = f"</tool_call{_HYTK}>"
_TOOL_SEP = f"<tool_sep{_HYTK}>"
_ARG_KEY = f"<arg_key{_HYTK}>"
_ARG_KEY_END = f"</arg_key{_HYTK}>"
_ARG_VALUE = f"<arg_value{_HYTK}>"
_ARG_VALUE_END = f"</arg_value{_HYTK}>"
_TOOL_RESPONSES = f"<tool_responses{_HYTK}>"
_TOOL_RESPONSES_END = f"</tool_responses{_HYTK}>"
_TOOL_RESPONSE = f"<tool_response{_HYTK}>"
_TOOL_RESPONSE_END = f"</tool_response{_HYTK}>"


class Hy3Renderer:
    """Deterministic message → token renderer for Tencent Hy3 models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Hy3RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or Hy3RendererConfig()
        self._is_training = self.config.is_training
        self._raw_last_assistant = self.config.raw_last_assistant
        # ``fallback_strategy="reasoning_toolcall_retry"`` forces high effort and
        # suppresses the generation prompt (template lines 50-53), resolved here
        # so the reasoning-mode marker and gen-prompt polarity see the override.
        self._force_no_gen_prompt = (
            self.config.fallback_strategy == "reasoning_toolcall_retry"
        )
        self._reasoning_effort = (
            "high" if self._force_no_gen_prompt else self.config.reasoning_effort
        )
        # ``<think>`` (and, in no_think mode, the matching ``</think>``) are
        # prefilled by the generation prompt, so the model never samples them.
        # Only in low/high mode does the model itself emit the ``</think>``
        # that closes its reasoning.
        self._think_is_sampled = self._reasoning_effort in ("low", "high")
        # Derived bridge policy: when the template keeps reasoning on every
        # historical turn it is safe to extend across a user query ("all");
        # otherwise past-cycle reasoning is stripped once a new user query
        # arrives, so the faithful policy declines there ("tool_cycle").
        # ``preserved_thinking=None`` follows the template's tools-dependent
        # default, so the bridge resolves the policy per call; this attribute
        # holds the no-tools resolution.
        self.effective_thinking_retention = self._thinking_retention_for(None)

        self._bos = self._token_id(_BOS)
        self._eos = self._token_id(_EOS)
        self._user = self._token_id(_USER)
        self._assistant = self._token_id(_ASSISTANT)
        self._reasoning_mode = self._token_id(_REASONING_MODE)
        self._think = self._token_id(_THINK)
        self._think_end = self._token_id(_THINK_END)
        self._tool_calls = self._token_id(_TOOL_CALLS)
        self._tool_calls_end = self._token_id(_TOOL_CALLS_END)
        self._tool_call = self._token_id(_TOOL_CALL)
        self._tool_call_end = self._token_id(_TOOL_CALL_END)
        self._tool_sep = self._token_id(_TOOL_SEP)
        self._arg_key = self._token_id(_ARG_KEY)
        self._arg_key_end = self._token_id(_ARG_KEY_END)
        self._arg_value = self._token_id(_ARG_VALUE)
        self._arg_value_end = self._token_id(_ARG_VALUE_END)
        self._tool_responses = self._token_id(_TOOL_RESPONSES)
        self._tool_responses_end = self._token_id(_TOOL_RESPONSES_END)
        self._tool_response = self._token_id(_TOOL_RESPONSE)
        self._tool_response_end = self._token_id(_TOOL_RESPONSE_END)

    # ── helpers ──────────────────────────────────────────────────────

    def _preserved_thinking_for(self, tools: list[ToolSpec] | None) -> bool:
        """Mirror the template's ``preserved_thinking`` default: the kwarg
        when set, else True iff tools are present."""
        if self.config.preserved_thinking is None:
            return bool(tools)
        return self.config.preserved_thinking

    def _thinking_retention_for(
        self, tools: list[ToolSpec] | None
    ) -> ResolvedThinkingRetention:
        implied = "all" if self._preserved_thinking_for(tools) else "tool_cycle"
        return resolve_thinking_retention(self.config, implied)

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
        """Mirror the template's ``visible_text`` macro: string verbatim, list
        of text parts concatenated, ``None`` → empty string."""
        if content is None:
            return ""
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

    @staticmethod
    def _last_user_index(messages: list[Message]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    def _reasoning_content(self, msg: Message) -> str | None:
        rc = msg.get("reasoning_content")
        if isinstance(rc, str):
            return rc
        rc = msg.get("reasoning")
        if isinstance(rc, str):
            return rc
        return None

    def _tools_instruction_block(self, tools: list[ToolSpec], has_system: bool) -> str:
        """Build the ``# Tools`` instruction blob (all scaffold).

        Embedded special-token strings (``<tool_calls>`` etc.) tokenize to
        their single ids just as ``apply_chat_template`` produces them.
        """
        intro = (
            "# Tools\n\nYou may call one or more functions to assist with the "
            "user query."
        )
        s = ("\n\n" + intro) if has_system else intro
        s += (
            "\n\nYou are provided with function signatures within "
            "<tools></tools> XML tags:\n<tools>\n"
        )
        for idx, tool in enumerate(tools):
            if idx > 0:
                s += "\n"
            s += json.dumps(tool, ensure_ascii=False)
        s += "\n</tools>\n\n"
        s += "For function call returns, you should first print " + _TOOL_CALLS + "\n"
        s += "For each function call, you should return object like:\n"
        s += _TOOL_CALL + "{function-name}" + _TOOL_SEP + "\n"
        s += _ARG_KEY + "{arg-key-1}" + _ARG_KEY_END + "\n"
        s += _ARG_VALUE + "{arg-value-1}" + _ARG_VALUE_END + "\n"
        s += _ARG_KEY + "{arg-key-2}" + _ARG_KEY_END + "\n"
        s += _ARG_VALUE + "{arg-value-2}" + _ARG_VALUE_END + "\n"
        s += "...\n"
        s += _TOOL_CALL_END + "\n"
        # ``reasoning_effort`` is always a non-empty string in our config, so
        # the marker is always appended (matching the template's truthy path).
        s += (
            "At the end of function call returns, you should print "
            + _TOOL_CALLS_END
            + _REASONING_MODE
            + "reasoning_effort:"
            + self._reasoning_effort
        )
        return s

    def _attribute_segments(
        self, segments: list[tuple[str, bool, int]]
    ) -> list[tuple[int, bool, int]]:
        """Tokenize concatenated ``(text, is_content, msg_idx)`` segments as one
        BPE pass, attributing each token to its source segment via offset
        mapping. Generalises :func:`attribute_text_segments` with a message
        index — needed where a system blob abuts the tools block with no
        special-token boundary between them.
        """
        segments = [s for s in segments if s[0]]
        if not segments:
            return []
        full_text = "".join(text for text, _, _ in segments)
        encoding = self._tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoding["input_ids"])
        offsets = list(encoding["offset_mapping"])

        spans: list[tuple[int, int, bool, int]] = []
        pos = 0
        for text, is_content, msg_idx in segments:
            spans.append((pos, pos + len(text), is_content, msg_idx))
            pos += len(text)
        total_len = pos

        out: list[tuple[int, bool, int]] = []
        last = (spans[-1][2], spans[-1][3])
        for tok_id, (start, _end) in zip(token_ids, offsets):
            attr = last
            if start < total_len:
                for seg_start, seg_end, seg_is_content, seg_idx in spans:
                    if seg_start <= start < seg_end:
                        attr = (seg_is_content, seg_idx)
                        break
            out.append((tok_id, attr[0], attr[1]))
        return out

    # ── render ───────────────────────────────────────────────────────

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        # fallback_strategy="reasoning_toolcall_retry" suppresses the gen prompt.
        if self._force_no_gen_prompt:
            add_generation_prompt = False

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

        def emit_attributed(segments: list[tuple[str, bool, int]]) -> None:
            for tok_id, is_content, msg_idx in self._attribute_segments(segments):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(False)
                content_mask.append(is_content)

        preserved = self._preserved_thinking_for(tools)

        # ── Header: bos + aggregated system + reasoning marker / tools ──
        emit_special(self._bos, -1, is_sampled=False, is_content=False)

        system_segments = [
            (self._visible_text(m.get("content")), True, i)
            for i, m in enumerate(messages)
            if m.get("role") == "system"
        ]
        has_system = any(text for text, _, _ in system_segments)

        if tools:
            tools_text = self._tools_instruction_block(tools, has_system)
            emit_attributed(system_segments + [(tools_text, False, -1)])
        else:
            emit_attributed(system_segments)
            emit_special(self._reasoning_mode, -1, is_sampled=False, is_content=False)
            emit_text(
                "reasoning_effort:" + self._reasoning_effort,
                -1,
                is_sampled=False,
                is_content=False,
            )

        last_ui = self._last_user_index(messages)
        n = len(messages)
        prev_is_tool = False
        is_tool_first = True

        # ── Message loop (system handled in the header) ─────────────────
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "system":
                continue

            if role == "user":
                if prev_is_tool:
                    emit_special(
                        self._tool_responses_end, i, is_sampled=False, is_content=False
                    )
                emit_special(self._user, i, is_sampled=False, is_content=False)
                emit_text(
                    self._visible_text(msg.get("content")),
                    i,
                    is_sampled=False,
                    is_content=True,
                )
                prev_is_tool = False

            elif role == "assistant":
                if prev_is_tool:
                    emit_special(
                        self._tool_responses_end, i, is_sampled=False, is_content=False
                    )
                self._render_assistant(
                    msg,
                    i,
                    is_last=(i == n - 1),
                    retain_thinking=self._is_training or preserved or i > last_ui,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )
                prev_is_tool = False
                # The template re-opens a <tool_responses> group only after an
                # assistant that made tool calls; a plain assistant leaves the
                # flag as it was.
                if msg.get("tool_calls"):
                    is_tool_first = True

            elif role == "tool":
                if is_tool_first:
                    emit_special(
                        self._tool_responses, i, is_sampled=False, is_content=False
                    )
                    emit_text("\n", i, is_sampled=False, is_content=False)
                    is_tool_first = False
                emit_special(self._tool_response, i, is_sampled=False, is_content=False)
                emit_text_segments(
                    [
                        ("\n", False),
                        (self._visible_text(msg.get("content")), True),
                        ("\n", False),
                    ],
                    i,
                    is_sampled=False,
                )
                emit_special(
                    self._tool_response_end, i, is_sampled=False, is_content=False
                )
                emit_text("\n", i, is_sampled=False, is_content=False)
                prev_is_tool = True

        if prev_is_tool:
            emit_special(
                self._tool_responses_end, -1, is_sampled=False, is_content=False
            )

        # ── Generation prompt ───────────────────────────────────────────
        last_is_assistant = messages[-1].get("role") == "assistant"
        if add_generation_prompt and not last_is_assistant:
            emit_special(self._assistant, -1, is_sampled=False, is_content=False)
            emit_special(self._think, -1, is_sampled=False, is_content=False)
            if self._reasoning_effort == "no_think":
                emit_special(self._think_end, -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
        )

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        *,
        is_last: bool,
        retain_thinking: bool,
        emit_special,
        emit_text,
    ) -> None:
        # Invariant on assistant tokens: ``is_content == sampled_mask``.
        # The ``<｜hy_Assistant｜>`` opener and the ``<think>`` opener are both
        # generation-prompt scaffold (never sampled). ``</think>`` and any
        # reasoning body are sampled only in low/high mode; content, tool
        # calls and the closing ``<｜hy_eos｜>`` are always sampled.
        emit_special(self._assistant, msg_idx, is_sampled=False, is_content=False)

        visible = self._visible_text(msg.get("content"))
        tool_calls = msg.get("tool_calls") or []

        # Raw passthrough for a trailing non-tool assistant (prefill /
        # continuation): no ``<think>`` wrap, no ``<｜hy_eos｜>`` — just the
        # bare visible content (template lines 186-187).
        if self._raw_last_assistant and is_last and not tool_calls:
            emit_text(visible, msg_idx, is_sampled=True, is_content=True)
            return

        emit_special(self._think, msg_idx, is_sampled=False, is_content=False)

        rc = self._reasoning_content(msg)
        if retain_thinking and rc is not None:
            emit_text(
                rc,
                msg_idx,
                is_sampled=self._think_is_sampled,
                is_content=self._think_is_sampled,
            )
        emit_special(
            self._think_end,
            msg_idx,
            is_sampled=self._think_is_sampled,
            is_content=self._think_is_sampled,
        )

        if tool_calls:
            emit_text(visible, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._tool_calls, msg_idx, is_sampled=True, is_content=True)
            emit_text("\n", msg_idx, is_sampled=True, is_content=True)
            for tc in tool_calls:
                self._emit_tool_call(tc, msg_idx, emit_special, emit_text)
            emit_special(
                self._tool_calls_end, msg_idx, is_sampled=True, is_content=True
            )
            emit_special(self._eos, msg_idx, is_sampled=True, is_content=True)
        else:
            emit_text(visible, msg_idx, is_sampled=True, is_content=True)
            # Final assistant keeps its close only under is_training; otherwise
            # a terminal assistant is left open (template lines 188-192).
            if not is_last or self._is_training:
                emit_special(self._eos, msg_idx, is_sampled=True, is_content=True)

    def _emit_tool_call(self, tc: Any, msg_idx: int, emit_special, emit_text) -> None:
        func = tc.get("function") or tc
        name = func.get("name", "")
        arguments = func.get("arguments", {})

        emit_special(self._tool_call, msg_idx, is_sampled=True, is_content=True)
        emit_text(name, msg_idx, is_sampled=True, is_content=True)
        emit_special(self._tool_sep, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=True, is_content=True)

        # OpenAI canonical form serialises ``arguments`` as a JSON string;
        # parse it so per-argument rendering still fires.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                emit_special(self._arg_key, msg_idx, is_sampled=True, is_content=True)
                emit_text(key, msg_idx, is_sampled=True, is_content=True)
                emit_special(
                    self._arg_key_end, msg_idx, is_sampled=True, is_content=True
                )
                emit_text("\n", msg_idx, is_sampled=True, is_content=True)
                emit_special(self._arg_value, msg_idx, is_sampled=True, is_content=True)
                if isinstance(value, str):
                    emit_text(value, msg_idx, is_sampled=True, is_content=True)
                else:
                    emit_text(
                        json.dumps(value, ensure_ascii=False),
                        msg_idx,
                        is_sampled=True,
                        is_content=True,
                    )
                emit_special(
                    self._arg_value_end, msg_idx, is_sampled=True, is_content=True
                )
                emit_text("\n", msg_idx, is_sampled=True, is_content=True)
        emit_special(self._tool_call_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=True, is_content=True)

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

    # ── parse ────────────────────────────────────────────────────────

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        return parse_hy3(
            self._tokenizer,
            token_ids,
            stop_ids={self._eos},
            assistant_id=self._assistant,
            think_id=self._think,
            think_end_id=self._think_end,
            tool_calls_id=self._tool_calls,
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
            tool_sep_id=self._tool_sep,
            arg_key_id=self._arg_key,
            arg_key_end_id=self._arg_key_end,
            arg_value_id=self._arg_value,
            arg_value_end_id=self._arg_value_end,
            tools=tools,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._eos]

    # ── bridge ───────────────────────────────────────────────────────

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
            self._thinking_retention_for(tools),
            new_messages,
        ):
            return None

        # A bridge extends a *sampled* assistant turn; with no completion there
        # is no turn to extend (and no way to tell a pending assistant turn from
        # a closed tool section), so decline and let the caller re-render.
        if not previous_completion_ids:
            return None

        # Anchor on the canonical turn close (``<｜hy_eos｜>``). The model only
        # ever ends a turn on eos, so a completion that stops elsewhere was
        # truncated mid-turn — synthesise the close as non-loss prompt context.
        # Never synthesise after a boundary that is already a valid
        # continuation point: eos (turn closed) or ``</tool_responses>`` (tool
        # section closed, reachable when the prior prompt suppressed the
        # generation prompt) — an unconditional eos there would wedge a
        # spurious stop token before the extension.
        previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
        if previous_ids[-1] not in (self._eos, self._tool_responses_end):
            previous_ids.append(self._eos)

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

        def emit_text_segments(segments: list[tuple[str, bool]], msg_idx: int) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                ext.append(tok_id)
                ext_indices.append(msg_idx)
                ext_content.append(is_content)

        # The stream above ends on eos or </tool_responses> — never inside an
        # open tool group. ``is_tool_first`` mirrors the template's state
        # machine: it resets only on an assistant turn that made tool calls,
        # and a tool extension always follows exactly such a turn (that is
        # what produced the tool results), so it enters the extension True and
        # is never reset again — assistants are rejected above. The first tool
        # group opens <tool_responses>; a later group (after a user turn) does
        # not, matching a full render byte-for-byte.
        prev_is_tool = False
        is_tool_first = True
        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))
            if role == "user":
                if prev_is_tool:
                    emit_special(self._tool_responses_end, i)
                emit_special(self._user, i)
                emit_text(content, i, is_content=True)
                prev_is_tool = False
            elif role == "system":
                # Hy3 folds system content into the header, which the bridge
                # cannot rewrite without re-rendering the prior turn.
                return None
            elif role == "tool":
                if is_tool_first:
                    emit_special(self._tool_responses, i)
                    emit_text("\n", i)
                    is_tool_first = False
                emit_special(self._tool_response, i)
                emit_text_segments([("\n", False), (content, True), ("\n", False)], i)
                emit_special(self._tool_response_end, i)
                emit_text("\n", i)
                prev_is_tool = True
            else:
                return None

        if prev_is_tool:
            emit_special(self._tool_responses_end, -1)

        # Generation prompt — suppressed under the fallback retry strategy, so
        # the extension matches a full render (which forces it off too).
        if not self._force_no_gen_prompt:
            emit_special(self._assistant, -1)
            emit_special(self._think, -1)
            if self._reasoning_effort == "no_think":
                emit_special(self._think_end, -1)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )
