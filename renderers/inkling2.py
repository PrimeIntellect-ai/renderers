"""Template-parity text renderer for Thinking Machines Inkling 2."""

from __future__ import annotations

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    extract_message_tool_names,
)
from renderers.configs import Inkling2RendererConfig
from renderers.default import DefaultRenderer
from renderers.inkling import _classify_part
from renderers.parsers import Inkling2ToolParser, _token_id
from renderers.parsing import _strip_stop_tokens


class Inkling2Renderer(DefaultRenderer):
    """Use the model template for IDs and attribute its text-only message blocks."""

    config: Inkling2RendererConfig

    def __init__(self, tokenizer, config: Inkling2RendererConfig | None = None):
        super().__init__(tokenizer, config or Inkling2RendererConfig())
        # The template keeps every historical reasoning block.
        self.effective_thinking_retention = "all"
        self._message_system_id = self._special_token_id("<|message_system|>")
        self._content_text_id = self._special_token_id("<|content_text|>")
        self._end_message_id = self._special_token_id("<|end_message|>")
        self._model_end_sampling_id = self._special_token_id(
            "<|content_model_end_sampling|>"
        )
        self._native_parser = (
            self._tool_parser
            if isinstance(self._tool_parser, Inkling2ToolParser)
            else Inkling2ToolParser(tokenizer)
        )

    def _special_token_id(self, token: str) -> int:
        tid = _token_id(self._tokenizer, token)
        if tid is None:
            raise ValueError(
                f"Inkling 2 tokenizer is missing the special token {token!r}"
            )
        return tid

    def get_stop_token_ids(self) -> list[int]:
        return [self._model_end_sampling_id]

    def _template_kwargs(self) -> dict:
        kwargs = super()._template_kwargs()
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        return kwargs

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    kind, _ = _classify_part(part)
                    if kind != "text":
                        raise ValueError(
                            f"Inkling2Renderer is text-only and does not support {kind} content"
                        )
        return super().render_ids(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            return super().render(
                messages, tools=tools, add_generation_prompt=add_generation_prompt
            )
        ids = self.render_ids(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )
        indices = [-1] * len(ids)
        sampled = [False] * len(ids)
        is_content = [False] * len(ids)

        # A bodyless system message yields just the tools and effort scaffold.
        # The template moves effort after the leading system messages, so
        # incremental prefix lengths would count it against the first system.
        scaffold = self._apply([{"role": "system", "content": []}], tools=tools)
        offset = len(scaffold) - 1 - scaffold[::-1].index(self._message_system_id)
        effort_len = len(scaffold) - offset
        effort_pending = True

        # The template concatenates all matching function names for an unnamed
        # tool result, including reused IDs later in the conversation.
        tool_names: dict[str, list[str]] = {}
        for message in messages:
            if message["role"] == "assistant":
                for call in message.get("tool_calls") or []:
                    if call_id := call.get("id"):
                        tool_names.setdefault(call_id, []).append(
                            call["function"]["name"]
                        )

        for index, message in enumerate(messages):
            role = message["role"]
            if effort_pending and role != "system":
                offset += effort_len
                effort_pending = False

            template_message: Message = message
            if role == "tool" and not message.get("name"):
                name = "".join(tool_names.get(message.get("tool_call_id", ""), []))
                template_message = message.copy()
                template_message["name"] = name
            standalone = self._apply([template_message])
            span = (
                standalone[:-effort_len]
                if role == "system"
                else standalone[effort_len:]
            )
            end = offset + len(span)
            if span != ids[offset:end]:
                raise ValueError("Inkling 2 template changed a message's token span")

            indices[offset:end] = [index] * len(span)
            if role == "assistant":
                # The template supplies no generation prefill. The entire turn,
                # including its first role/content markers, is sampled.
                sampled[offset:end] = [True] * len(span)
                is_content[offset:end] = [True] * len(span)
            else:
                is_content[offset:end] = self._text_content_mask(message, span)
            offset = end

        return RenderedTokens(
            token_ids=ids,
            message_indices=indices,
            message_roles=[message["role"] for message in messages],
            message_tool_names=extract_message_tool_names(messages),
            sampled_mask=sampled,
            is_content=is_content,
        )

    def _text_content_mask(self, message: Message, ids: list[int]) -> list[bool]:
        mask = [False] * len(ids)
        content = message.get("content")
        parts = [content] if isinstance(content, str) else (content or [])
        start = 0
        for part in parts:
            _, body = _classify_part(part)
            body_ids = list(self._tokenizer.encode(body, add_special_tokens=False))
            body_start = ids.index(self._content_text_id, start) + 1
            end = body_start + len(body_ids)
            if ids[body_start : end + 1] != [*body_ids, self._end_message_id]:
                raise ValueError("Inkling 2 text body does not match its template span")
            mask[body_start:end] = [True] * len(body_ids)
            start = end + 1
        return mask

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — declared in the previous prompt
    ) -> RenderedTokens | None:
        """Append user and tool messages to a stop-terminated sampled turn.

        The template renders each message independently of its neighbours, so the
        extension is the template's own span for every new message. A truncated
        turn needs a close the template would add, and an unnamed tool result
        takes its name from the issuing assistant, which ``new_messages`` omits;
        both decline so the caller re-renders.
        """
        stop = self._model_end_sampling_id
        if (
            not previous_prompt_ids
            or not new_messages
            or previous_completion_ids.count(stop) != 1
            or previous_completion_ids[-1] != stop
        ):
            return None
        for message in new_messages:
            role = message.get("role")
            if role not in ("user", "tool"):
                return None
            if (
                role == "tool"
                and not message.get("name")
                and message.get("tool_call_id")
            ):
                return None

        effort_len = len(self._apply([{"role": "system", "content": []}]))
        ids = [*previous_prompt_ids, *previous_completion_ids]
        indices = [-1] * len(ids)
        is_content = [False] * len(ids)
        for index, message in enumerate(new_messages):
            span = self.render_ids([message])[effort_len:]
            ids.extend(span)
            indices.extend([index] * len(span))
            is_content.extend(self._text_content_mask(message, span))
        return RenderedTokens(
            token_ids=ids,
            message_indices=indices,
            message_roles=[message["role"] for message in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
            sampled_mask=[False] * len(ids),
            is_content=is_content,
        )

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — native JSON needs no schema
        prompt_ids: list[int] | None = None,  # noqa: ARG002 — no generation prefill
    ) -> ParsedResponse:
        ids = _strip_stop_tokens(token_ids, set(self.get_stop_token_ids()))
        if (
            self._tool_parser is not None
            and self._tool_parser is not self._native_parser
        ):
            ids, tool_calls = self._tool_parser.extract(ids)
        else:
            tool_calls = []

        response = self._native_parser.parse(
            ids, parse_tools=self._tool_parser is self._native_parser
        )
        if self._reasoning_parser is not None and response.reasoning_content is None:
            reasoning, content = self._reasoning_parser.extract(response.content)
            response.reasoning_content = reasoning
            response.content = content
        response.tool_calls.extend(tool_calls)
        return response


__all__ = ["Inkling2Renderer"]
