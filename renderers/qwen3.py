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
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import Qwen3RendererConfig
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
        config: Qwen3RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or Qwen3RendererConfig()
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config,
            "all" if not self.config.enable_thinking else "tool_cycle",
        )

        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")
        self._think_end = self._token_id("</think>")

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
    def _query_boundary_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _is_user_query_message(msg: Message) -> bool:
        if msg.get("role") != "user":
            return False
        content = Qwen3Renderer._query_boundary_text(msg.get("content"))
        return not (
            content.startswith("<tool_response>")
            and content.endswith("</tool_response>")
        )

    @staticmethod
    def _last_query_index(messages: list[Message]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            if Qwen3Renderer._is_user_query_message(messages[i]):
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

        # ── 1. System + tools ───────────────────────────────────────
        first_is_system = messages[0].get("role") == "system"

        if tools:
            sys_idx = 0 if first_is_system else -1
            emit_special(self._im_start, sys_idx, is_sampled=False, is_content=False)
            # Body = system content (if any). Everything else in this
            # block — role tag, tools header / footer, the JSON tool
            # specs — is scaffold. The tools dict is recoverable from
            # the ``tools`` argument; don't re-attribute its embedded
            # JSON as message body.
            segments: list[tuple[str, bool]] = [("system\n", False)]
            if first_is_system:
                sys_content = messages[0].get("content") or ""
                if sys_content:
                    segments.append((sys_content, True))
                segments.append(("\n\n", False))
            segments.append((_TOOLS_HEADER, False))
            for tool in tools:
                segments.append(("\n" + json.dumps(tool, ensure_ascii=False), False))
            segments.append((_TOOLS_FOOTER, False))
            emit_text_segments(segments, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False, is_content=False)
            emit_text("\n", sys_idx, is_sampled=False, is_content=False)
        elif first_is_system:
            emit_special(self._im_start, 0, is_sampled=False, is_content=False)
            sys_content = messages[0].get("content") or ""
            sys_segments: list[tuple[str, bool]] = [("system\n", False)]
            if sys_content:
                sys_segments.append((sys_content, True))
            emit_text_segments(sys_segments, 0, is_sampled=False)
            emit_special(self._im_end, 0, is_sampled=False, is_content=False)
            emit_text("\n", 0, is_sampled=False, is_content=False)

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
                emit_special(self._im_start, i, is_sampled=False, is_content=False)
                msg_segments: list[tuple[str, bool]] = [(role + "\n", False)]
                if content:
                    msg_segments.append((content, True))
                emit_text_segments(msg_segments, i, is_sampled=False)
                emit_special(self._im_end, i, is_sampled=False, is_content=False)
                emit_text("\n", i, is_sampled=False, is_content=False)

            elif role == "user":
                emit_special(self._im_start, i, is_sampled=False, is_content=False)
                user_segments: list[tuple[str, bool]] = [("user\n", False)]
                if content:
                    user_segments.append((content, True))
                emit_text_segments(user_segments, i, is_sampled=False)
                emit_special(self._im_end, i, is_sampled=False, is_content=False)
                emit_text("\n", i, is_sampled=False, is_content=False)

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    content,
                    last_qi,
                    i == num_messages - 1,
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

        # ── 4. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_start, -1, is_sampled=False, is_content=False)
            emit_text("assistant\n", -1, is_sampled=False, is_content=False)
            if not self.config.enable_thinking:
                emit_text(
                    "<think>\n\n</think>\n\n", -1, is_sampled=False, is_content=False
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
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — hermes wire format quotes strings, schema not needed
    ) -> ParsedResponse:
        return parse_qwen3(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
            reasoning_end_id=self._think_end,
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

        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention,
            new_messages,
            is_user_query=self._is_user_query_message,
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

        # Trailing ``\n`` after the turn-close token. ``render()`` emits this
        # as part of the prior turn, but vLLM stops on ``<|im_end|>`` so the
        # ``\n`` never lands in prev_completion.
        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            if role == "user":
                emit_special(self._im_start, i)
                user_segments: list[tuple[str, bool]] = [("user\n", False)]
                if content:
                    user_segments.append((content, True))
                emit_text_segments(user_segments, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)
            elif role == "system":
                emit_special(self._im_start, i)
                sys_segments: list[tuple[str, bool]] = [("system\n", False)]
                if content:
                    sys_segments.append((content, True))
                emit_text_segments(sys_segments, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)
            elif role == "tool":
                self._render_tool(
                    new_messages,
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )
            else:
                return None

        emit_special(self._im_start, -1)
        emit_text("assistant\n", -1)
        if not self.config.enable_thinking:
            emit_text("<think>\n\n</think>\n\n", -1)

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
        msg,
        msg_idx,
        content,
        last_query_index,
        is_last,
        *,
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

        # ``<|im_start|>assistant\n`` is template-injected scaffolding —
        # at inference the chat template emits these as the generation
        # prompt and the model never samples them. Marking the role tag
        # as ``is_sampled=False`` keeps the SFT loss mask aligned with
        # what the model would actually have produced. ``is_content`` is
        # also False here — the role tag isn't part of any message's
        # body, on any role.
        emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
        emit_text("assistant\n", msg_idx, is_sampled=False, is_content=False)

        # Build the model-sampled portion (think block + content + tool
        # calls). Text segments stay contiguous within each is_sampled
        # span to preserve BPE merges (e.g., ".\n" is a single token in
        # Qwen3); the only split we introduce here is at ``\n`` after the
        # role tag, which the existing renderer already treats as a
        # token boundary (cf. ``_render_tool``). For assistant messages
        # the invariant ``is_content == sampled_mask`` holds — every
        # sampled token is body, every scaffold token isn't.
        tool_calls = msg.get("tool_calls") or []

        emit_in_template_window = msg_idx > last_query_index and (
            is_last or reasoning_content
        )
        if emit_in_template_window:
            body = (
                "<think>\n"
                + reasoning_content.strip("\n")
                + "\n</think>\n\n"
                + content.lstrip("\n")
            )
        else:
            body = content

        if not tool_calls:
            emit_text(body, msg_idx, is_sampled=True, is_content=True)
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
                    emit_text(
                        body + separator, msg_idx, is_sampled=True, is_content=True
                    )
                else:
                    emit_text("\n", msg_idx, is_sampled=True, is_content=True)

                emit_special(self._tool_call, msg_idx, is_sampled=True, is_content=True)
                emit_text(
                    '\n{"name": "' + name + '", "arguments": ' + args_str + "}\n",
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
                emit_special(
                    self._tool_call_end, msg_idx, is_sampled=True, is_content=True
                )

        # ``<|im_end|>`` is the model's stop signal — it samples this to
        # end its turn, so it is part of the sampled stream (and the
        # assistant's body). The trailing ``\n`` is template-appended
        # between turns and never sampled — scaffold for is_content too.
        emit_special(self._im_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

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
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False. The ``content``
        # field's body bytes get ``is_content=True``; everything else —
        # the ``<|im_start|>user`` wrap, the inter-section ``\n``s, the
        # ``<|tool_response>`` specials — is scaffold so the SFT mask
        # for tool body never trains the model to emit them.
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )

        if not prev_is_tool:
            emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
            emit_text("user", msg_idx, is_sampled=False, is_content=False)

        emit_text("\n", msg_idx, is_sampled=False, is_content=False)
        emit_special(self._tool_response, msg_idx, is_sampled=False, is_content=False)
        # ``\n`` + content + ``\n`` — body is the middle segment only.
        # Single BPE pass over the joined text preserves boundary
        # merges (Qwen3 keeps ``\n`` as its own token, so this is
        # mostly a no-op, but we route through segments anyway so the
        # attribution doesn't depend on tokenizer-specific behaviour).
        emit_text_segments(
            [("\n", False), (content, True), ("\n", False)],
            msg_idx,
            is_sampled=False,
        )
        emit_special(
            self._tool_response_end, msg_idx, is_sampled=False, is_content=False
        )

        if not next_is_tool:
            emit_special(self._im_end, msg_idx, is_sampled=False, is_content=False)
            emit_text("\n", msg_idx, is_sampled=False, is_content=False)
