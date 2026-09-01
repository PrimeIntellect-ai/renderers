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

import numpy as np

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    Tokenizer,
    _get_offset_tokenizer,
    _infer_offsets_from_decode,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
)
from renderers.configs import Hy3RendererConfig, ResolvedThinkingRetention
from renderers.parsing import parse_hy3
from renderers.token_arrays import (
    MASK_DTYPE,
    MESSAGE_INDICES_DTYPE,
    OFFSETS_DTYPE,
    TOKEN_IDS_DTYPE,
    FixedWidthArrayBuilder,
    RenderedTokenBuilder,
    TextSegmentBuilder,
    encode_token_ids,
    owned_offsets_from_array,
    owned_token_ids_from_array,
    require_1d_array,
)

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

    def __init__(self, tokenizer: Tokenizer, config: Hy3RendererConfig | None = None):
        self._tokenizer = tokenizer
        self.config = config or Hy3RendererConfig()
        self._is_training = self.config.is_training
        self._raw_last_assistant = self.config.raw_last_assistant
        # ``fallback_strategy="reasoning_toolcall_retry"`` forces high effort and
        # suppresses the generation prompt (template lines 50-53), resolved here
        # so the reasoning-mode marker and gen-prompt polarity see the override.
        self._force_no_gen_prompt = self.config.fallback_strategy == "reasoning_toolcall_retry"
        self._reasoning_effort = "high" if self._force_no_gen_prompt else self.config.reasoning_effort
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
        self._offset_tokenizer = _get_offset_tokenizer(tokenizer)

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

    def _thinking_retention_for(self, tools: list[ToolSpec] | None) -> ResolvedThinkingRetention:
        implied = "all" if self._preserved_thinking_for(tools) else "tool_cycle"
        return resolve_thinking_retention(self.config, implied)

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

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
        intro = "# Tools\n\nYou may call one or more functions to assist with the user query."
        s = ("\n\n" + intro) if has_system else intro
        s += "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
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
        self, texts: list[str], segment_content: np.ndarray, segment_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Tokenize joined text and vectorize token attribution to each segment."""
        require_1d_array("segment_content", segment_content, dtype=MASK_DTYPE)
        require_1d_array("segment_indices", segment_indices, dtype=MESSAGE_INDICES_DTYPE, minimum=-1)
        if len(texts) != segment_content.size or len(texts) != segment_indices.size:
            raise ValueError("segment text and attribution arrays must have equal lengths")
        char_lengths = np.fromiter((len(text) for text in texts), dtype=OFFSETS_DTYPE, count=len(texts))
        nonempty = char_lengths > 0
        texts = [text for text in texts if text]
        if not texts:
            empty_tokens = np.empty(0, dtype=TOKEN_IDS_DTYPE)
            empty_indices = np.empty(0, dtype=MESSAGE_INDICES_DTYPE)
            empty_content = np.empty(0, dtype=MASK_DTYPE)
            for values in (empty_tokens, empty_indices, empty_content):
                values.flags.writeable = False
            return empty_tokens, empty_indices, empty_content, True
        char_lengths = char_lengths[nonempty]
        segment_content = segment_content[nonempty]
        segment_indices = segment_indices[nonempty]
        full_text = "".join(texts)
        offset_tokenizer = self._offset_tokenizer
        if offset_tokenizer is None:
            token_ids = encode_token_ids(self._tokenizer, full_text)
            offsets = _infer_offsets_from_decode(self._tokenizer, token_ids, full_text)
            if offsets is None:
                # Token IDs remain exact even when a lossy decoder prevents
                # reconstructing boundaries. Associate the opaque joined run
                # with a contributing caller system message rather than
                # silently classifying its body as global scaffold.
                candidates = np.flatnonzero(segment_indices >= 0)
                fallback_idx = int(segment_indices[candidates[0]]) if candidates.size else -1
                message_indices = np.full(token_ids.size, fallback_idx, dtype=MESSAGE_INDICES_DTYPE)
                is_content = np.zeros(token_ids.size, dtype=MASK_DTYPE)
                message_indices.flags.writeable = False
                is_content.flags.writeable = False
                return token_ids, message_indices, is_content, False
            has_content_attribution = False
        else:
            encoding = offset_tokenizer(
                full_text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="np"
            )
            token_ids = owned_token_ids_from_array(type(offset_tokenizer).__name__, encoding["input_ids"])
            offsets = owned_offsets_from_array(
                type(offset_tokenizer).__name__, encoding["offset_mapping"], token_count=token_ids.size
            )
            has_content_attribution = True
        segment_ends = np.cumsum(char_lengths, dtype=OFFSETS_DTYPE)
        token_segments = np.searchsorted(segment_ends, offsets[:, 0], side="right")
        np.minimum(token_segments, segment_ends.size - 1, out=token_segments)
        message_indices = np.array(segment_indices[token_segments], dtype=MESSAGE_INDICES_DTYPE, copy=True)
        is_content = (
            np.array(segment_content[token_segments], dtype=MASK_DTYPE, copy=True)
            if has_content_attribution
            else np.zeros(token_ids.size, dtype=MASK_DTYPE)
        )
        for values in (token_ids, message_indices, is_content):
            values.flags.writeable = False
        return token_ids, message_indices, is_content, has_content_attribution

    # ── render ───────────────────────────────────────────────────────

    def render(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, add_generation_prompt: bool = False
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        # fallback_strategy="reasoning_toolcall_retry" suppresses the gen prompt.
        if self._force_no_gen_prompt:
            add_generation_prompt = False

        builder = RenderedTokenBuilder(self._tokenizer, offset_tokenizer=self._offset_tokenizer)
        emit_special = builder.emit_special
        emit_text = builder.emit_text
        emit_text_segments = builder.emit_text_segments

        def emit_attributed(texts: list[str], segment_content: np.ndarray, segment_indices: np.ndarray) -> None:
            token_ids, message_indices, is_content, _ = self._attribute_segments(
                texts, segment_content, segment_indices
            )
            builder.emit_aligned(token_ids, message_indices, np.zeros(token_ids.size, dtype=MASK_DTYPE), is_content)

        preserved = self._preserved_thinking_for(tools)

        # ── Header: bos + aggregated system + reasoning marker / tools ──
        emit_special(self._bos, -1, is_sampled=False, is_content=False)

        system_texts: list[str] = []
        system_content = FixedWidthArrayBuilder(MASK_DTYPE, initial_capacity=len(messages))
        system_indices = FixedWidthArrayBuilder(MESSAGE_INDICES_DTYPE, initial_capacity=len(messages))
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                system_texts.append(self._visible_text(message.get("content")))
                system_content.append(True)
                system_indices.append(i)
        has_system = any(system_texts)

        if tools:
            tools_text = self._tools_instruction_block(tools, has_system)
            system_texts.append(tools_text)
            system_content.append(False)
            system_indices.append(-1)
            emit_attributed(system_texts, system_content.finish(), system_indices.finish())
        else:
            emit_attributed(system_texts, system_content.finish(), system_indices.finish())
            emit_special(self._reasoning_mode, -1, is_sampled=False, is_content=False)
            emit_text("reasoning_effort:" + self._reasoning_effort, -1, is_sampled=False, is_content=False)

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
                    emit_special(self._tool_responses_end, i, is_sampled=False, is_content=False)
                emit_special(self._user, i, is_sampled=False, is_content=False)
                emit_text(self._visible_text(msg.get("content")), i, is_sampled=False, is_content=True)
                prev_is_tool = False

            elif role == "assistant":
                if prev_is_tool:
                    emit_special(self._tool_responses_end, i, is_sampled=False, is_content=False)
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
                    emit_special(self._tool_responses, i, is_sampled=False, is_content=False)
                    emit_text("\n", i, is_sampled=False, is_content=False)
                    is_tool_first = False
                emit_special(self._tool_response, i, is_sampled=False, is_content=False)
                tool_segments = TextSegmentBuilder()
                tool_segments.append("\n", is_content=False)
                tool_segments.append(self._visible_text(msg.get("content")), is_content=True)
                tool_segments.append("\n", is_content=False)
                emit_text_segments(tool_segments, i, is_sampled=False)
                emit_special(self._tool_response_end, i, is_sampled=False, is_content=False)
                emit_text("\n", i, is_sampled=False, is_content=False)
                prev_is_tool = True

        if prev_is_tool:
            emit_special(self._tool_responses_end, -1, is_sampled=False, is_content=False)

        # ── Generation prompt ───────────────────────────────────────────
        last_is_assistant = messages[-1].get("role") == "assistant"
        if add_generation_prompt and not last_is_assistant:
            emit_special(self._assistant, -1, is_sampled=False, is_content=False)
            emit_special(self._think, -1, is_sampled=False, is_content=False)
            if self._reasoning_effort == "no_think":
                emit_special(self._think_end, -1, is_sampled=False, is_content=False)

        return builder.finish(
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
            content_available=self._offset_tokenizer is not None,
        )

    def _render_assistant(
        self, msg: Message, msg_idx: int, *, is_last: bool, retain_thinking: bool, emit_special, emit_text
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
            emit_text(rc, msg_idx, is_sampled=self._think_is_sampled, is_content=self._think_is_sampled)
        emit_special(self._think_end, msg_idx, is_sampled=self._think_is_sampled, is_content=self._think_is_sampled)

        if tool_calls:
            emit_text(visible, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._tool_calls, msg_idx, is_sampled=True, is_content=True)
            emit_text("\n", msg_idx, is_sampled=True, is_content=True)
            for tc in tool_calls:
                self._emit_tool_call(tc, msg_idx, emit_special, emit_text)
            emit_special(self._tool_calls_end, msg_idx, is_sampled=True, is_content=True)
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
                emit_special(self._arg_key_end, msg_idx, is_sampled=True, is_content=True)
                emit_text("\n", msg_idx, is_sampled=True, is_content=True)
                emit_special(self._arg_value, msg_idx, is_sampled=True, is_content=True)
                if isinstance(value, str):
                    emit_text(value, msg_idx, is_sampled=True, is_content=True)
                else:
                    emit_text(json.dumps(value, ensure_ascii=False), msg_idx, is_sampled=True, is_content=True)
                emit_special(self._arg_value_end, msg_idx, is_sampled=True, is_content=True)
                emit_text("\n", msg_idx, is_sampled=True, is_content=True)
        emit_special(self._tool_call_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=True, is_content=True)

    def render_ids(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, add_generation_prompt: bool = False
    ) -> np.ndarray:
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    # ── parse ────────────────────────────────────────────────────────

    def parse_response(self, token_ids: np.ndarray, *, tools: list[ToolSpec] | None = None) -> ParsedResponse:
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
        previous_prompt_ids: np.ndarray,
        previous_completion_ids: np.ndarray,
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if previous_prompt_ids.size == 0 or not new_messages or reject_assistant_in_extension(new_messages):
            return None

        if should_rerender_for_thinking_retention(self._thinking_retention_for(tools), new_messages):
            return None

        # A bridge extends a *sampled* assistant turn; with no completion there
        # is no turn to extend (and no way to tell a pending assistant turn from
        # a closed tool section), so decline and let the caller re-render.
        if previous_completion_ids.size == 0:
            return None

        # Anchor on the canonical turn close (``<｜hy_eos｜>``). The model only
        # ever ends a turn on eos, so a completion that stops elsewhere was
        # truncated mid-turn — synthesise the close as non-loss prompt context.
        # Never synthesise after a boundary that is already a valid
        # continuation point: eos (turn closed) or ``</tool_responses>`` (tool
        # section closed, reachable when the prior prompt suppressed the
        # generation prompt) — an unconditional eos there would wedge a
        # spurious stop token before the extension.
        previous = FixedWidthArrayBuilder(
            TOKEN_IDS_DTYPE, initial_capacity=previous_prompt_ids.size + previous_completion_ids.size + 1
        )
        previous.extend(previous_prompt_ids)
        previous.extend(previous_completion_ids)
        final_token = int(previous_completion_ids[-1])
        if final_token not in (self._eos, self._tool_responses_end):
            previous.append(self._eos)
        previous_ids = previous.finish()

        builder = RenderedTokenBuilder(self._tokenizer, offset_tokenizer=self._offset_tokenizer)
        builder.prepend_prior(previous_ids)

        def emit_special(token_id: int, msg_idx: int = -1) -> None:
            builder.emit_special(token_id, msg_idx)

        def emit_text(text: str, msg_idx: int = -1, *, is_content: bool = False) -> None:
            builder.emit_text(text, msg_idx, is_content=is_content)

        def emit_text_segments(segments: TextSegmentBuilder, msg_idx: int) -> None:
            builder.emit_text_segments(segments, msg_idx)

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
                tool_segments = TextSegmentBuilder()
                tool_segments.append("\n", is_content=False)
                tool_segments.append(content, is_content=True)
                tool_segments.append("\n", is_content=False)
                emit_text_segments(tool_segments, i)
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

        return builder.finish(
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
            content_available=self._offset_tokenizer is not None,
        )
