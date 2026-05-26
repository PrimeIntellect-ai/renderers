"""GLM-4.5 Air Renderer — hard-coded Python mirroring the GLM-4.5 Jinja chat template.

Key differences from GLM-5:
- \\n after every role marker (<|user|>\\n, <|assistant|>\\n)
- <think></think>\\n separator (vs bare </think> in GLM-5)
- Tool calls have \\n between arg tags
- Thinking disabled via /nothink appended to user content
- Gen prompt (thinking=True): just <|assistant|> (no <think>)
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
)
from renderers.configs import GLM45RendererConfig
from renderers.parsing import parse_glm

_TOOLS_HEADER = (
    "\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)

_TOOLS_FOOTER = (
    "</tools>\n\n"
    "For each function call, output the function name and arguments "
    "within the following XML format:\n"
    "<tool_call>{function-name}\n"
    "<arg_key>{arg-key-1}</arg_key>\n"
    "<arg_value>{arg-value-1}</arg_value>\n"
    "<arg_key>{arg-key-2}</arg_key>\n"
    "<arg_value>{arg-value-2}</arg_value>\n"
    "...\n"
    "</tool_call>"
)


class GLM45Renderer:
    """Deterministic message → token renderer for GLM-4.5 Air models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: GLM45RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or GLM45RendererConfig()

        self._gmask = self._token_id("[gMASK]")
        self._sop = self._token_id("<sop>")
        self._system = self._token_id("<|system|>")
        self._user = self._token_id("<|user|>")
        self._assistant = self._token_id("<|assistant|>")
        self._observation = self._token_id("<|observation|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tool_call_tok = self._token_id("<tool_call>")
        self._tool_call_end_tok = self._token_id("</tool_call>")
        self._arg_key = self._token_id("<arg_key>")
        self._arg_key_end = self._token_id("</arg_key>")
        self._arg_value = self._token_id("<arg_value>")
        self._arg_value_end = self._token_id("</arg_value>")

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

    @staticmethod
    def _last_user_index(messages: list[Message]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

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
            """Tokenize concatenated segments as one BPE pass; per-token
            ``is_content`` follows each token's source segment.

            Lets call sites express "this wrap + this body, joined the
            same way as the chat template, but attributed separately"
            without splitting the encode call (which could shift BPE
            merges at the boundary)."""
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        # ── Prefix ──────────────────────────────────────────────────
        emit_special(self._gmask, -1, is_sampled=False, is_content=False)
        emit_special(self._sop, -1, is_sampled=False, is_content=False)

        # ── Tools in system prompt ──────────────────────────────────
        # The tools-header block is all scaffold by design — the tools
        # dict is recoverable from the ``tools`` argument; don't
        # re-attribute the embedded JSON specs as message body.
        if tools:
            emit_special(self._system, -1, is_sampled=False, is_content=False)
            tool_text = _TOOLS_HEADER
            for tool in tools:
                tool_text += json.dumps(tool, ensure_ascii=False) + "\n"
            tool_text += _TOOLS_FOOTER
            emit_text(tool_text, -1, is_sampled=False, is_content=False)

        # ── Compute last_user_index ─────────────────────────────────
        last_ui = self._last_user_index(messages)

        # ── Iterate messages ────────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = self._visible_text(msg.get("content"))

            # When the previous message is an assistant, this message's
            # role-opening token (``<|user|>`` / ``<|observation|>``) is
            # the inference-time stop signal that closes the assistant's
            # turn (see ``get_stop_token_ids``). Mark it
            # ``is_sampled=True`` so the loss-mask pipeline trains the
            # model to emit it after ``</tool_call>`` (instead of
            # continuing with another ``<tool_call>`` block). The token
            # stays attributed to this message (msg_idx=i) and remains
            # ``is_content=False`` — it's a role-marker / scaffold, not
            # body bytes, so ``content_mask_for_roles({"tool"})`` and
            # ``content_token_spans_by_role()`` correctly exclude it
            # from "tool body" views. Byte stream is unchanged.
            # ``system`` only appears at the start of a GLM conversation,
            # so its opener is never the closer of an assistant turn.
            closes_assistant_turn = i > 0 and messages[i - 1]["role"] == "assistant"

            if role == "system":
                emit_special(self._system, i, is_sampled=False, is_content=False)
                # ``\n`` is the scaffold separator after the role tag;
                # the body proper is the caller-provided content.
                emit_text_segments(
                    [("\n", False), (content, True)], i, is_sampled=False
                )

            elif role == "user":
                emit_special(
                    self._user,
                    i,
                    is_sampled=closes_assistant_turn,
                    is_content=False,
                )
                # ``\n`` is scaffold; ``content`` is body; the optional
                # ``/nothink`` suffix is scaffold the renderer injects
                # when ``enable_thinking=False``.
                user_segments: list[tuple[str, bool]] = [("\n", False), (content, True)]
                if not self.config.enable_thinking and not content.endswith("/nothink"):
                    user_segments.append(("/nothink", False))
                emit_text_segments(user_segments, i, is_sampled=False)

            elif role == "assistant":
                preserve_thinking = should_preserve_past_thinking(
                    messages,
                    i,
                    preserve_all_thinking=self.config.preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self.config.preserve_thinking_between_tool_calls,
                )
                self._render_assistant(
                    msg,
                    i,
                    content,
                    last_ui,
                    preserve_thinking=preserve_thinking,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

            elif role == "tool":
                self._render_tool(
                    messages,
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

        # ── Generation prompt ───────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._assistant, -1, is_sampled=False, is_content=False)
            if not self.config.enable_thinking:
                emit_text("\n", -1, is_sampled=False, is_content=False)
                emit_special(self._think, -1, is_sampled=False, is_content=False)
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
        return parse_glm(
            self._tokenizer,
            token_ids,
            stop_ids={self._endoftext, self._user, self._observation},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call_tok,
            tool_call_end_id=self._tool_call_end_tok,
            arg_key_id=self._arg_key,
            arg_key_end_id=self._arg_key_end,
            arg_value_id=self._arg_value,
            arg_value_end_id=self._arg_value_end,
            tools=tools,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._endoftext, self._user, self._observation]

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

        # Same next-turn-marker scheme as GLM-5, but role markers are
        # followed by a literal ``\n`` in the prompt text.
        previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
        stop_ids = {self._endoftext, self._user, self._observation}
        if (
            not previous_ids[len(previous_prompt_ids) :]
            or previous_ids[-1] not in stop_ids
        ):
            previous_ids.append(self._endoftext)

        last_prev = previous_ids[-1]

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

        # The opener-token of the first new_message may also serve as
        # the close of the previous assistant turn (when the model
        # failed to sample the stop token itself and the bridge has to
        # synthesize the boundary above). Unlike :meth:`render`, the
        # bridge emits these with ``is_sampled=False, is_content=False``
        # — they are template scaffolding for the *next* step's prompt,
        # not tokens the model produced *in this* step. The RL loss
        # operates on ``previous_completion_ids`` (what the model
        # actually sampled this round); bridge tokens belong to the
        # subsequent prompt and must not be counted as "model output"
        # by downstream mask consumers. This deliberate disagreement
        # with ``render()`` reflects the SFT vs RL semantics: render's
        # masks describe what the model *should* produce given a
        # complete conversation; bridge's masks describe what it
        # *actually* produced this step.
        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._visible_text(msg.get("content"))
            if role == "user":
                if not (i == 0 and last_prev == self._user):
                    emit_special(self._user, i)
                user_segments: list[tuple[str, bool]] = [
                    ("\n", False),
                    (content, True),
                ]
                if not self.config.enable_thinking and not content.endswith("/nothink"):
                    user_segments.append(("/nothink", False))
                emit_text_segments(user_segments, i)
            elif role == "system":
                emit_special(self._system, i)
                emit_text_segments([("\n", False), (content, True)], i)
            elif role == "tool":
                prev_is_tool = i > 0 and new_messages[i - 1].get("role") == "tool"
                if i == 0 and last_prev == self._observation:
                    pass
                elif not prev_is_tool:
                    emit_special(self._observation, i)
                emit_text_segments(
                    [
                        ("\n<tool_response>\n", False),
                        (content, True),
                        ("\n</tool_response>", False),
                    ],
                    i,
                )
            else:
                return None

        # Generation prompt.
        emit_special(self._assistant, -1)
        if not self.config.enable_thinking:
            emit_text("\n", -1)
            emit_special(self._think, -1)
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
        msg,
        msg_idx,
        content,
        last_user_index,
        *,
        preserve_thinking: bool = False,
        emit_special,
        emit_text,
        emit_text_segments,
    ):
        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        elif "</think>" in content:
            before, after = content.split("</think>", 1)
            if "<think>" in before:
                reasoning_content = before.split("<think>")[-1].lstrip("\n")
            else:
                reasoning_content = before.lstrip("\n")
            reasoning_content = reasoning_content.rstrip("\n")
            content = after.lstrip("\n")

        # ``<|assistant|>\n`` is template-injected scaffolding — at
        # inference the chat template emits these as the generation
        # prompt and the model never samples them. Everything after
        # (think block + content + tool calls) is the model-sampled
        # portion.
        #
        # GLM-4.5 does NOT emit an explicit per-turn close token inside
        # the assistant message; the next message's role marker
        # (``<|user|>`` / ``<|observation|>`` / ``<|endoftext|>``) acts
        # as the stop signal at inference, and those tokens are
        # attributed to the *next* message (or are absent on the final
        # turn). So no sampled stop-signal token lives inside this
        # assistant span — content / think / tool_calls carry the
        # is_sampled=True signal.
        #
        # Invariant on assistant tokens: ``is_content == sampled_mask``.
        # Every scaffold token here gets ``is_sampled=False/is_content=False``;
        # every model-sampled emit gets both True.
        emit_special(self._assistant, msg_idx, is_sampled=False, is_content=False)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

        if (msg_idx > last_user_index or preserve_thinking) and reasoning_content:
            emit_special(self._think, msg_idx, is_sampled=True, is_content=True)
            emit_text(
                reasoning_content.strip(), msg_idx, is_sampled=True, is_content=True
            )
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
        else:
            emit_special(self._think, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)

        # Tool calls — keep content + \n contiguous to preserve BPE merges
        tool_calls = msg.get("tool_calls") or []
        if content.strip() and tool_calls:
            emit_text(
                "\n" + content.strip() + "\n", msg_idx, is_sampled=True, is_content=True
            )
        elif content.strip():
            emit_text("\n" + content.strip(), msg_idx, is_sampled=True, is_content=True)

        for tc in tool_calls:
            func = tc.get("function") or tc
            name = func.get("name", "")
            arguments = func.get("arguments", {})

            if not content.strip():
                emit_text("\n", msg_idx, is_sampled=True, is_content=True)
            emit_special(self._tool_call_tok, msg_idx, is_sampled=True, is_content=True)
            emit_text(name + "\n", msg_idx, is_sampled=True, is_content=True)
            # OpenAI canonical form: arguments is a JSON string. Parse it so the
            # per-argument rendering below still works.
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, dict):
                for arg_name, arg_value in arguments.items():
                    emit_special(
                        self._arg_key, msg_idx, is_sampled=True, is_content=True
                    )
                    emit_text(arg_name, msg_idx, is_sampled=True, is_content=True)
                    emit_special(
                        self._arg_key_end, msg_idx, is_sampled=True, is_content=True
                    )
                    emit_text("\n", msg_idx, is_sampled=True, is_content=True)
                    emit_special(
                        self._arg_value, msg_idx, is_sampled=True, is_content=True
                    )
                    if isinstance(arg_value, str):
                        emit_text(arg_value, msg_idx, is_sampled=True, is_content=True)
                    else:
                        emit_text(
                            json.dumps(arg_value, ensure_ascii=False),
                            msg_idx,
                            is_sampled=True,
                            is_content=True,
                        )
                    emit_special(
                        self._arg_value_end, msg_idx, is_sampled=True, is_content=True
                    )
                    emit_text("\n", msg_idx, is_sampled=True, is_content=True)
            emit_special(
                self._tool_call_end_tok, msg_idx, is_sampled=True, is_content=True
            )

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        # Tool body bytes get ``is_content=True``; the wraps are
        # scaffold. The ``<|observation|>`` role tag is scaffold too
        # (``is_content=False`` so ``content_mask_for_roles({"tool"})``
        # excludes it). When the previous message is an assistant it
        # doubles as the inference stop signal for that assistant's
        # turn — mark it ``is_sampled=True`` so SFT trains the model to
        # emit it after ``</tool_call>``. The token stays attributed to
        # this tool message; byte stream is unchanged.
        prev_role = messages[msg_idx - 1]["role"] if msg_idx > 0 else None
        closes_assistant_turn = prev_role == "assistant"

        if prev_role != "tool":
            emit_special(
                self._observation,
                msg_idx,
                is_sampled=closes_assistant_turn,
                is_content=False,
            )

        emit_text_segments(
            [
                ("\n<tool_response>\n", False),
                (content, True),
                ("\n</tool_response>", False),
            ],
            msg_idx,
            is_sampled=False,
        )
