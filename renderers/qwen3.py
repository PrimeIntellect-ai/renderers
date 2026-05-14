"""Qwen3 Renderer — hard-coded Python mirroring the Qwen3 Jinja chat template.

Key differences from Qwen3.5:
- Content is always string (no list/multimodal support)
- Tool calls use JSON format: {"name": "...", "arguments": ...}
- Thinking blocks only inserted when loop.last OR reasoning_content present
- Generation prompt does NOT add <think> by default
"""

from __future__ import annotations

import json

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    reject_assistant_in_extension,
    should_preserve_past_thinking,
    trim_to_turn_close,
)
from renderers.parsing import parse_qwen3

_TOOLS_HEADER = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)

_TOOLS_FOOTER = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


class Qwen3Renderer:
    """Deterministic message → token renderer for Qwen3 models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
        preserve_all_thinking: bool = False,
        preserve_thinking_between_tool_calls: bool = False,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking
        self._preserve_all_thinking = preserve_all_thinking
        self._preserve_thinking_between_tool_calls = (
            preserve_thinking_between_tool_calls
        )

        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")

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
    def _last_query_index(messages: list[Message]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if not (
                content.startswith("<tool_response>")
                and content.endswith("</tool_response>")
            ):
                return i
        return len(messages) - 1

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

        def emit_ids(ids: list[int], msg_idx: int, *, is_sampled: bool) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))

        def emit_special(token_id: int, msg_idx: int, *, is_sampled: bool) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)

        def emit_text(text: str, msg_idx: int, *, is_sampled: bool) -> None:
            emit_ids(self._encode(text), msg_idx, is_sampled=is_sampled)

        # ── 1. System + tools ───────────────────────────────────────
        first_is_system = messages[0].get("role") == "system"

        if tools:
            sys_idx = 0 if first_is_system else -1
            emit_special(self._im_start, sys_idx, is_sampled=False)
            tool_text = "system\n"
            if first_is_system:
                tool_text += (messages[0].get("content") or "") + "\n\n"
            tool_text += _TOOLS_HEADER
            for tool in tools:
                tool_text += "\n" + json.dumps(tool, ensure_ascii=False)
            tool_text += _TOOLS_FOOTER
            emit_text(tool_text, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False)
            emit_text("\n", sys_idx, is_sampled=False)
        elif first_is_system:
            emit_special(self._im_start, 0, is_sampled=False)
            emit_text(
                "system\n" + (messages[0].get("content") or ""), 0, is_sampled=False
            )
            emit_special(self._im_end, 0, is_sampled=False)
            emit_text("\n", 0, is_sampled=False)

        # ── 2. Compute last_query_index ─────────────────────────────
        last_qi = self._last_query_index(messages)

        # ── 3. Iterate messages ─────────────────────────────────────
        num_messages = len(messages)
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""

            if role == "system":
                if i == 0:
                    continue
                emit_special(self._im_start, i, is_sampled=False)
                emit_text(role + "\n" + content, i, is_sampled=False)
                emit_special(self._im_end, i, is_sampled=False)
                emit_text("\n", i, is_sampled=False)

            elif role == "user":
                emit_special(self._im_start, i, is_sampled=False)
                emit_text("user\n" + content, i, is_sampled=False)
                emit_special(self._im_end, i, is_sampled=False)
                emit_text("\n", i, is_sampled=False)

            elif role == "assistant":
                preserve_thinking = should_preserve_past_thinking(
                    messages,
                    i,
                    preserve_all_thinking=self._preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self._preserve_thinking_between_tool_calls,
                )
                self._render_assistant(
                    msg,
                    i,
                    content,
                    last_qi,
                    i == num_messages - 1,
                    preserve_thinking=preserve_thinking,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                self._render_tool(
                    messages, i, content, emit_special=emit_special, emit_text=emit_text
                )

        # ── 4. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_start, -1, is_sampled=False)
            emit_text("assistant\n", -1, is_sampled=False)
            if not self._enable_thinking:
                emit_text("<think>\n\n</think>\n\n", -1, is_sampled=False)

        return RenderedTokens(
            token_ids=tokens, message_indices=indices, sampled_mask=sampled
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
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — hermes wire format quotes strings, schema not needed
    ) -> ParsedResponse:
        return parse_qwen3(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end, self._endoftext]

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
            {self._im_end, self._endoftext},
            synthesize_close=self._im_end,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []

        # Bridge output is consumed as the next turn's prompt — the
        # caller blanket-masks it via ``prompt_mask=[False]*N``, so we
        # don't track sampled_mask here. Local helpers accept the kwarg
        # for signature compatibility with ``_render_tool`` and ignore
        # it; the returned ``RenderedTokens`` leaves ``sampled_mask``
        # empty.
        def emit_special(
            token_id: int, _msg_idx: int = -1, *, is_sampled: bool = False
        ) -> None:
            ext.append(token_id)

        def emit_text(
            text: str, _msg_idx: int = -1, *, is_sampled: bool = False
        ) -> None:
            ext.extend(self._encode(text))

        # Trailing ``\n`` after the turn-close token. ``render()`` emits this
        # as part of the prior turn, but vLLM stops on ``<|im_end|>`` so the
        # ``\n`` never lands in prev_completion.
        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            if role == "user":
                emit_special(self._im_start, i)
                emit_text("user\n" + content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)
            elif role == "system":
                emit_special(self._im_start, i)
                emit_text("system\n" + content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)
            elif role == "tool":
                self._render_tool(
                    new_messages,
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )
            else:
                return None

        emit_special(self._im_start, -1)
        emit_text("assistant\n", -1)
        if not self._enable_thinking:
            emit_text("<think>\n\n</think>\n\n", -1)

        return RenderedTokens(token_ids=previous_ids + ext)

    def _render_assistant(
        self,
        msg,
        msg_idx,
        content,
        last_query_index,
        is_last,
        *,
        preserve_thinking: bool = False,
        emit_special,
        emit_text,
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

        # ``<|im_start|>assistant\n`` is template-injected scaffolding —
        # at inference the chat template emits these as the generation
        # prompt and the model never samples them. Marking the role tag
        # as ``is_sampled=False`` keeps the SFT loss mask aligned with
        # what the model would actually have produced.
        emit_special(self._im_start, msg_idx, is_sampled=False)
        emit_text("assistant\n", msg_idx, is_sampled=False)

        # Build the model-sampled portion (think block + content + tool
        # calls). Text segments stay contiguous within each is_sampled
        # span to preserve BPE merges (e.g., ".\n" is a single token in
        # Qwen3); the only split we introduce here is at ``\n`` after the
        # role tag, which the existing renderer already treats as a
        # token boundary (cf. ``_render_tool``).
        tool_calls = msg.get("tool_calls") or []

        emit_in_template_window = msg_idx > last_query_index and (
            is_last or reasoning_content
        )
        emit_via_override = preserve_thinking and bool(reasoning_content)
        if emit_in_template_window or emit_via_override:
            body = (
                "<think>\n"
                + reasoning_content.strip("\n")
                + "\n</think>\n\n"
                + content.lstrip("\n")
            )
        else:
            body = content

        if not tool_calls:
            emit_text(body, msg_idx, is_sampled=True)
        else:
            for tc_idx, tc in enumerate(tool_calls):
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )

                # Text before this tool_call (includes separator)
                if tc_idx == 0:
                    separator = "\n" if content else ""
                    emit_text(body + separator, msg_idx, is_sampled=True)
                else:
                    emit_text("\n", msg_idx, is_sampled=True)

                emit_special(self._tool_call, msg_idx, is_sampled=True)
                emit_text(
                    '\n{"name": "' + name + '", "arguments": ' + args_str + "}\n",
                    msg_idx,
                    is_sampled=True,
                )
                emit_special(self._tool_call_end, msg_idx, is_sampled=True)

        # ``<|im_end|>`` is the model's stop signal — it samples this to
        # end its turn, so it is part of the sampled stream. The trailing
        # ``\n`` is template-appended between turns and never sampled.
        emit_special(self._im_end, msg_idx, is_sampled=True)
        emit_text("\n", msg_idx, is_sampled=False)

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False.
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )

        if not prev_is_tool:
            emit_special(self._im_start, msg_idx, is_sampled=False)
            emit_text("user", msg_idx, is_sampled=False)

        emit_text("\n", msg_idx, is_sampled=False)
        emit_special(self._tool_response, msg_idx, is_sampled=False)
        emit_text("\n" + content + "\n", msg_idx, is_sampled=False)
        emit_special(self._tool_response_end, msg_idx, is_sampled=False)

        if not next_is_tool:
            emit_special(self._im_end, msg_idx, is_sampled=False)
            emit_text("\n", msg_idx, is_sampled=False)
