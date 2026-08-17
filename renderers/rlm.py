"""Renderer for the RLM chat format (``PrimeIntellect/RLM-Chat-Template``).

The format is deliberately minimal — one message, one tag block, nothing else:

    <system>...</system><user>...</user><assistant>{content}<ipython>{code}</ipython></assistant><output>{result}</output>

All ten delimiters are single added tokens (reserved ``<SPECIAL_18>``..``<SPECIAL_27>``
slots of the Nemotron-3 tokenizer, renamed), so every tag is one token and
``</assistant>`` (id 23 on the reference tokenizer) is the eos/stop token.

Semantics:

- **Thinking is never dropped.** Assistant content renders verbatim on every
  turn; thinking retention is structurally ``"all"`` and the bridge never
  re-renders for retention. A separate ``reasoning_content`` field (as parse
  APIs and some datasets carry) is glued back as ``<think>{reasoning}</think>``
  before the content — plain text, since the format has no native thinking
  syntax. The HF chat template ignores ``reasoning_content`` (jinja cannot
  guarantee the field exists); the renderer restores it instead of dropping
  it, which is the only divergence from byte-parity with the template.
- **ipython is the only tool.** An assistant turn may end with exactly one
  ``<ipython>{code}</ipython>`` call; the tool result comes back as an
  ``<output>...</output>`` turn (OpenAI ``role: "tool"``). Tool *schemas* are
  never rendered: the ``tools=`` kwarg is validation-only — the rlm harness
  sends ``tools=[ipython]`` on every chat-completion call, which is accepted
  and ignored; anything else raises. (The HF template raises on any
  ``tools=`` because ``apply_chat_template`` is never in the serving path —
  the renderer is; see the PR description for the end-to-end trace.)
"""

from __future__ import annotations

import json
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
    ToolSpec,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import RlmRendererConfig


class RlmRenderer:
    """Deterministic message ↔ token renderer for the RLM chat format."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: RlmRendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or RlmRendererConfig()
        # Retention is structurally "all": the format has no truncate rule and
        # the config validator rejects anything else.
        self.effective_thinking_retention = resolve_thinking_retention(self.config, "all")

        self._system = self._token_id("<system>")
        self._system_end = self._token_id("</system>")
        self._user = self._token_id("<user>")
        self._user_end = self._token_id("</user>")
        self._assistant = self._token_id("<assistant>")
        self._assistant_end = self._token_id("</assistant>")
        self._ipython = self._token_id("<ipython>")
        self._ipython_end = self._token_id("</ipython>")
        self._output = self._token_id("<output>")
        self._output_end = self._token_id("</output>")

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        if not isinstance(tid, int) or tid == self._tokenizer.unk_token_id:
            raise AssertionError(
                f"Special token {token!r} not found in tokenizer vocabulary — "
                "the rlm renderer requires the RLM-Chat-Template tokenizer "
                "(single-token role tags)."
            )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _render_content(content: Any) -> str:
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

    @staticmethod
    def _validate_tools(tools: list[ToolSpec] | None) -> None:
        """tools= is validation-only: exactly [ipython] is accepted, nothing is rendered."""
        if not tools:
            return
        if len(tools) != 1:
            raise ValueError(f"rlm accepts exactly one tool (ipython), got {len(tools)}")
        fn = tools[0].get("function") or {}
        if fn.get("name") != "ipython":
            raise ValueError(f"rlm's only tool is ipython, got {fn.get('name')!r}")

    def _assistant_body(self, msg: Message, content: str) -> str:
        """Assemble the assistant body: verbatim content (+ restored
        reasoning), then the single optional ipython call."""
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            content = "<think>" + reasoning + "</think>" + content

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return content
        if len(tool_calls) != 1:
            raise ValueError("rlm allows exactly one ipython call per assistant turn")
        name, arguments = self._normalize_tool_call(tool_calls[0])
        if name != "ipython":
            raise ValueError(f"unknown tool: {name!r} (only ipython exists)")
        if not isinstance(arguments, dict) or "code" not in arguments:
            raise ValueError("ipython arguments must carry a 'code' key")
        code = arguments["code"]
        if not isinstance(code, str):
            raise ValueError("ipython 'code' must be a string")
        return content + "<ipython>" + code + "</ipython>"

    @staticmethod
    def _normalize_tool_call(tc: Any) -> tuple[Any, Any]:
        """Return ``(name, arguments)`` from any of the shapes rlm data carries.

        Accepted: the OpenAI shape (``{"function": {"name", "arguments"}}``),
        the verifiers trace shape (flat ``{"name", "arguments"}``), and either
        as a JSON string (HF datasets store trace tool_calls as strings).
        ``arguments`` may itself be a JSON string; it is parsed here.
        """
        if isinstance(tc, str):
            tc = json.loads(tc)
        if not isinstance(tc, dict):
            raise ValueError(f"unparseable tool call: {tc!r}")
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        arguments = fn.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return fn.get("name"), arguments

    # The <ipython>/<\ipython> markers are single added tokens, so encoding the
    # assembled body yields the same ids as the chat template (which encodes a
    # rendered string). Mirrors Nemotron3Renderer._render_assistant.

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")
        self._validate_tools(tools)

        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit(ids: list[int], msg_idx: int, *, is_sampled: bool, is_content: bool) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = self._render_content(msg.get("content"))

            if role == "system":
                # Any position: the chat template renders <system> wherever it
                # appears, and rlm compaction traces carry mid-list systems.
                emit([self._system], i, is_sampled=False, is_content=False)
                emit(self._encode(content), i, is_sampled=False, is_content=True)
                emit([self._system_end], i, is_sampled=False, is_content=False)
            elif role == "user":
                emit([self._user], i, is_sampled=False, is_content=False)
                emit(self._encode(content), i, is_sampled=False, is_content=True)
                emit([self._user_end], i, is_sampled=False, is_content=False)
            elif role == "assistant":
                # The opening tag is the generation prompt (never sampled);
                # body and closing </assistant> are what the model produces —
                # on assistant the invariant is_content == sampled_mask holds
                # (mirrors Nemotron3Renderer._render_assistant).
                emit([self._assistant], i, is_sampled=False, is_content=False)
                body = self._assistant_body(msg, content)
                emit(self._encode(body), i, is_sampled=True, is_content=True)
                emit([self._assistant_end], i, is_sampled=True, is_content=True)
            elif role == "tool":
                emit([self._output], i, is_sampled=False, is_content=False)
                emit(self._encode(content), i, is_sampled=False, is_content=True)
                emit([self._output_end], i, is_sampled=False, is_content=False)
            else:
                raise ValueError(f"Unexpected message role: {role}")

        if add_generation_prompt:
            emit([self._assistant], -1, is_sampled=False, is_content=False)

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
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        ids = list(token_ids)
        while ids and ids[-1] == self._assistant_end:
            ids = ids[:-1]

        content_segments: list[str] = []
        tool_calls: list[ParsedToolCall] = []
        pos = 0
        while True:
            try:
                start = ids.index(self._ipython, pos)
            except ValueError:
                content_segments.append(self._decode(ids[pos:]))
                break
            content_segments.append(self._decode(ids[pos:start]))
            try:
                end = ids.index(self._ipython_end, start + 1)
            except ValueError:
                raw = self._decode(ids[start + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        name="ipython",
                        token_span=(start, len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            code = self._decode(ids[start + 1 : end])
            tool_calls.append(
                ParsedToolCall(
                    raw=code,
                    name="ipython",
                    arguments={"code": code},
                    token_span=(start, end + 1),
                    status=ToolCallParseStatus.OK,
                )
            )
            pos = end + 1

        # Thinking stays inline in content, verbatim — never split out.
        return ParsedResponse(
            content="".join(content_segments),
            reasoning_content=None,
            tool_calls=tool_calls,
        )

    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self._tokenizer.decode(ids)

    def get_stop_token_ids(self) -> list[int]:
        return [self._assistant_end]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if not previous_prompt_ids or not new_messages or reject_assistant_in_extension(new_messages):
            return None
        self._validate_tools(tools)
        # Retention is "all": appending never requires re-rendering history.

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._assistant_end},
            synthesize_close=self._assistant_end,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []
        ext_indices: list[int] = []
        ext_content: list[bool] = []

        def emit(ids: list[int], msg_idx: int, *, is_content: bool) -> None:
            ext.extend(ids)
            ext_indices.extend([msg_idx] * len(ids))
            ext_content.extend([is_content] * len(ids))

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = self._render_content(msg.get("content"))
            if role == "user":
                emit([self._user], i, is_content=False)
                emit(self._encode(content), i, is_content=True)
                emit([self._user_end], i, is_content=False)
            elif role == "tool":
                emit([self._output], i, is_content=False)
                emit(self._encode(content), i, is_content=True)
                emit([self._output_end], i, is_content=False)
            else:
                # System (or anything else) mid-conversation: not bridgeable.
                return None

        # Generation prompt.
        emit([self._assistant], -1, is_content=False)

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )
