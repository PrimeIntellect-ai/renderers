"""Default Renderer — falls back to tokenizer.apply_chat_template() for unsupported models.

This is the escape hatch: works with any model that has a Jinja chat template,
but doesn't provide message_indices (so build_training_sample uses incremental
rendering) and parse_response is basic text extraction unless tool/reasoning
parsers are plugged in.
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
    extract_message_tool_names,
    resolve_thinking_retention,
)
from renderers.configs import DefaultRendererConfig
from renderers.parsers import (
    get_reasoning_parser,
    get_tool_parser,
)


def _decode_tool_call_arguments(messages: list) -> list:
    """JSON-decode assistant tool_call ``arguments`` strings into dicts.

    OpenAI-format tool_calls carry ``arguments`` as a JSON-encoded string.
    Several chat templates (GLM-4.5, GLM-5) iterate ``arguments.items()``
    directly and crash on strings. Others (Qwen3 Hermes-style) branch on
    string-vs-dict and handle both. Decoding to dict is safe for both.

    Works on Pydantic AssistantMessage objects and plain dicts. Preserves
    non-JSON argument strings as-is so Hermes-style templates can still
    render them via the ``is string`` branch.
    """
    out: list[Any] = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role")
            tcs = m.get("tool_calls")
        else:
            role = getattr(m, "role", None)
            tcs = getattr(m, "tool_calls", None)
        if role != "assistant" or not tcs:
            out.append(m)
            continue

        md = m if isinstance(m, dict) else m.model_dump()  # type: ignore[attr-defined]
        md = dict(md)
        new_tcs: list[Any] = []
        for tc in md.get("tool_calls") or []:
            tc = dict(tc) if isinstance(tc, dict) else tc
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict):
                fn = dict(fn)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (ValueError, TypeError):
                        pass
                tc["function"] = fn
            else:
                args = tc.get("arguments") if isinstance(tc, dict) else None
                if isinstance(args, str):
                    try:
                        tc["arguments"] = json.loads(args)
                    except (ValueError, TypeError):
                        pass
            new_tcs.append(tc)
        md["tool_calls"] = new_tcs
        out.append(md)
    return out


class DefaultRenderer:
    """Fallback renderer using tokenizer.apply_chat_template().

    Works with any model. The config can carry ``tool_parser`` and/or
    ``reasoning_parser`` (resolved against ``renderers.parsers``) to
    enable structured output extraction, plus arbitrary additional Jinja
    template kwargs captured as ``model_extra`` (``extra="allow"`` on
    :class:`renderers.DefaultRendererConfig`).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DefaultRendererConfig | None = None,
    ):
        cfg = config or DefaultRendererConfig()
        if cfg.thinking_retention is not None:
            raise ValueError(
                "DefaultRenderer cannot implement explicit thinking_retention "
                "bridge policy because its template close/turn structure is "
                "opaque. Use a typed renderer for this model."
            )
        self.effective_thinking_retention = resolve_thinking_retention(
            cfg,
            "template",
        )
        self._tokenizer = tokenizer
        self.config = cfg
        self._tool_parser = _resolve_parser(cfg.tool_parser, tokenizer, get_tool_parser)
        self._reasoning_parser = _resolve_parser(
            cfg.reasoning_parser, tokenizer, get_reasoning_parser
        )

    @property
    def supports_tools(self) -> bool:
        return self._tool_parser is not None

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        # Incremental rendering to get per-token message attribution
        token_ids: list[int] = []
        message_indices: list[int] = []
        prev_len = 0

        for idx, message in enumerate(messages):
            cur_ids = self._apply(messages[: idx + 1], tools=tools)
            new_tokens = cur_ids[prev_len:]
            token_ids = cur_ids
            message_indices.extend([idx] * len(new_tokens))
            prev_len = len(cur_ids)

        if add_generation_prompt:
            full_ids = self._apply(messages, tools=tools, add_generation_prompt=True)
            gen_tokens = full_ids[prev_len:]
            token_ids = full_ids
            message_indices.extend([-1] * len(gen_tokens))

        message_roles = [m.get("role") or "" for m in messages]
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            message_roles=message_roles,
            message_tool_names=extract_message_tool_names(messages),
        )

    def _apply(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        kwargs = dict(self.config.model_extra or {})
        kwargs["add_generation_prompt"] = add_generation_prompt
        kwargs["tokenize"] = True
        if tools is not None:
            kwargs["tools"] = tools
        kwargs["return_dict"] = False
        messages = _decode_tool_call_arguments(messages)
        result = self._tokenizer.apply_chat_template(messages, **kwargs)
        return list(result)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self._apply(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — DefaultRenderer relies on configured tool_parser, schema not consulted here
    ) -> ParsedResponse:
        # 1. Extract tool calls while we still have token ids (most formats
        #    use special-token delimiters, so id-level matching is reliable).
        if self._tool_parser is not None:
            content_ids, tool_calls = self._tool_parser.extract(list(token_ids))
        else:
            content_ids = list(token_ids)
            tool_calls = []

        # 2. Decode (keep special tokens so a downstream reasoning parser can
        #    still see things like <think>/</think> when they're tokens).
        text = self._tokenizer.decode(content_ids, skip_special_tokens=False)

        # 3. Extract reasoning from the decoded text. Falls back to a built-in
        #    <think>...</think> sniff so unconfigured users get the same behavior
        #    as before. Preserve whitespace at the <think>/</think> boundary —
        #    the chat template round-trips it verbatim (e.g. GLM emits
        #    `{{ '\\n<think>' + reasoning_content + '</think>' }}` then
        #    `{{- content }}` with no separator), so a leading `\\n` on content
        #    or trailing `\\n` on reasoning_content must stay in the parsed
        #    fields for re-render to be byte-identical. Stripping here causes
        #    the re-rendered assistant message to shift by one BPE token after
        #    `</think>`, cascading through downstream tokenization and breaking
        #    the "extension property" in trajectory step tokenization.
        if self._reasoning_parser is not None:
            reasoning_content, text = self._reasoning_parser.extract(text)
        else:
            reasoning_content = None
            if "</think>" in text:
                before, after = text.split("</think>", 1)
                if "<think>" in before:
                    reasoning_content = before.split("<think>", 1)[-1]
                else:
                    reasoning_content = before
                text = after

        # Strip any remaining special tokens from the final content (we kept
        # them around for the reasoning parser above).
        text = _strip_special_tokens(self._tokenizer, text)

        return ParsedResponse(
            content=text,
            reasoning_content=reasoning_content if reasoning_content else None,
            tool_calls=tool_calls,
        )

    def get_stop_token_ids(self) -> list[int]:
        stop_ids = []
        if self._tokenizer.eos_token_id is not None:
            stop_ids.append(self._tokenizer.eos_token_id)
        return stop_ids

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        """DefaultRenderer wraps an unknown Jinja template — it has no
        hand-coded extension logic to emit. Return ``None`` so the caller
        falls back to a full re-render; that's correct whenever the
        template is prefix-stable under the new messages, which our parity
        suite enforces for anything we ship a renderer for.
        """
        return None


def _resolve_parser(value, tokenizer, factory):
    if value is None:
        return None
    if isinstance(value, str):
        return factory(value, tokenizer)
    return value


def _strip_special_tokens(tokenizer, text: str) -> str:
    """Remove any special-token substrings that slipped into decoded text."""
    specials = getattr(tokenizer, "all_special_tokens", None) or []
    for token in specials:
        if token and token in text:
            text = text.replace(token, "")
    return text
