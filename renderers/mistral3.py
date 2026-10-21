"""Mistral-3 Renderer — hard-coded Python mirroring Mistral's chat template.

Covers:
  * mistralai/Mistral-Small-3.1-24B-Instruct-2503
  * mistralai/Mistral-Small-3.2-24B-Instruct-2506
  * mistralai/Mistral-Nemo-Instruct-2407

Notable differences from the Llama-3 renderer:

* No reasoning channel — Mistral-3 ships no ``<think>`` concept.
* ``<s>`` (BOS) is emitted at the very start.
* Turn structure: ``[INST] {user} [/INST] {assistant}</s>``
* System message is prepended to the first user turn, separated by ``\n\n``.
* Tool calls: JSON array of ``{"name": ..., "arguments": ...}`` objects wrapped
  in ``[TOOL_CALLS]`` … ``</s>``.
* Tool responses: ``[TOOL_RESULTS]`` … ``[/TOOL_RESULTS]``.
"""

from __future__ import annotations

import json
from typing import Any

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    Tokenizer,
    _content_mask_or_empty,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import Mistral3RendererConfig
from renderers.parsing import parse_mistral3


class Mistral3Renderer:
    """Deterministic message → token renderer for Mistral-3.x Instruct models."""

    config_class = Mistral3RendererConfig

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Mistral3RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or Mistral3RendererConfig()
        # Mistral-3 ships no reasoning channel; thinking_retention is a no-op.
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config,
            "all",
        )

        self._bos = self._token_id("<s>")
        self._eos = self._token_id("</s>")
        self._inst_open = self._token_id("[INST]")
        self._inst_close = self._token_id("[/INST]")
        self._tool_calls = self._token_id("[TOOL_CALLS]")
        self._tool_results_open = self._token_id("[TOOL_RESULTS]")
        self._tool_results_close = self._token_id("[/TOOL_RESULTS]")

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

        # ── 0. BOS ──────────────────────────────────────────────────
        emit_special(self._bos, -1, is_sampled=False, is_content=False)

        # ── 1. Extract optional leading system message ───────────────
        first_is_system = messages[0].get("role") == "system"
        sys_text = (
            self._content_str(messages[0].get("content")).strip()
            if first_is_system
            else ""
        )
        body_messages = messages[1:] if first_is_system else messages
        offset = 1 if first_is_system else 0

        # ── 2. Body messages ─────────────────────────────────────────
        i = 0
        while i < len(body_messages):
            msg = body_messages[i]
            msg_idx = i + offset
            role = msg.get("role")
            tool_calls = msg.get("tool_calls")

            if role == "user":
                content = self._content_str(msg.get("content")).strip()
                emit_special(
                    self._inst_open, msg_idx, is_sampled=False, is_content=False
                )
                # System text is prepended only to the first user message.
                if i == 0 and sys_text:
                    segments: list[tuple[str, bool]] = [
                        (sys_text + "\n\n", False),
                        (content, True) if content else ("", False),
                    ]
                    emit_text_segments(segments, msg_idx, is_sampled=False)
                else:
                    segs: list[tuple[str, bool]] = []
                    if content:
                        segs.append((content, True))
                    if segs:
                        emit_text_segments(segs, msg_idx, is_sampled=False)
                emit_special(
                    self._inst_close, msg_idx, is_sampled=False, is_content=False
                )
                i += 1

            elif role == "assistant" and not tool_calls:
                content = self._content_str(msg.get("content")).strip()
                emit_text(" ", msg_idx, is_sampled=True, is_content=False)
                if content:
                    emit_text(content, msg_idx, is_sampled=True, is_content=True)
                emit_special(self._eos, msg_idx, is_sampled=True, is_content=True)
                i += 1

            elif role == "assistant" and tool_calls:
                # Tool-call assistant turn: [TOOL_CALLS] [{...}, ...] </s>
                emit_special(
                    self._tool_calls, msg_idx, is_sampled=True, is_content=False
                )
                calls: list[dict] = []
                for tc in tool_calls:
                    func = tc.get("function") or tc
                    name = func.get("name", "")
                    arguments = func.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    calls.append({"name": name, "arguments": arguments})
                call_str = json.dumps(calls, ensure_ascii=False)
                emit_text(
                    " " + call_str, msg_idx, is_sampled=True, is_content=True
                )
                emit_special(self._eos, msg_idx, is_sampled=True, is_content=True)
                i += 1

            elif role in ("tool", "ipython"):
                content = self._content_str(msg.get("content"))
                emit_special(
                    self._tool_results_open,
                    msg_idx,
                    is_sampled=False,
                    is_content=False,
                )
                if content:
                    emit_text(content, msg_idx, is_sampled=False, is_content=True)
                emit_special(
                    self._tool_results_close,
                    msg_idx,
                    is_sampled=False,
                    is_content=False,
                )
                i += 1

            else:
                raise ValueError(
                    f"Unexpected role {role!r} at message index {msg_idx}."
                )

        # ── 3. Generation prompt ─────────────────────────────────────
        if add_generation_prompt:
            emit_text(" ", -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=_content_mask_or_empty(self._tokenizer, content_mask),
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
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        return parse_mistral3(
            self._tokenizer,
            token_ids,
            stop_ids={self._eos},
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
        ext_content: list[bool] = []

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
            if role == "user":
                content = self._content_str(msg.get("content")).strip()
                emit_special(self._inst_open, i)
                if content:
                    emit_text_segments([(content, True)], i)
                emit_special(self._inst_close, i)
            elif role in ("tool", "ipython"):
                content = self._content_str(msg.get("content"))
                emit_special(self._tool_results_open, i)
                if content:
                    emit_text_segments([(content, True)], i)
                emit_special(self._tool_results_close, i)
            else:
                return None

        # Generation prompt.
        emit_text(" ", -1)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=_content_mask_or_empty(
                self._tokenizer, [False] * len(previous_ids) + ext_content
            ),
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )
