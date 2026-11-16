"""Inkling 2 renderer.

The Inkling 2 chat template is token-delimited like Inkling 1 (atomic role and
content special tokens, ``<|end_message|>`` closed blocks), but it drops audio,
adds optional assistant/tool name headers and closes every assistant turn with
``<|content_model_end_sampling|>``.

This renderer keeps the token stream byte-identical to
``tokenizer.apply_chat_template`` (it renders through the template) and adds the
attribution SFT needs:

* ``message_indices`` — from the default incremental (prefix) rendering;
* ``sampled_mask`` — an assistant turn is sampled from the first body token
  after its ``<|content_text|>`` marker through ``<|end_message|>`` and
  ``<|content_model_end_sampling|>``; header tokens are prompt;
* ``is_content`` — the body tokens only.

Assistant turns may carry ``reasoning_content`` and ``tool_calls``: the
template renders them as additional ``<|message_model|>``-tagged blocks, and
every token after the turn's first role tag is sampled and content, matching
the Inkling renderer.
"""

from __future__ import annotations

from renderers.base import Message, RenderedTokens, ToolSpec
from renderers.configs import Inkling2RendererConfig
from renderers.default import DefaultRenderer


class Inkling2Renderer(DefaultRenderer):
    """Template-parity Inkling 2 renderer with sampled-token attribution."""

    def __init__(self, tokenizer, config: Inkling2RendererConfig | None = None):
        super().__init__(tokenizer, config or Inkling2RendererConfig())
        self._content_text_id = self._special_token_id("<|content_text|>")
        self._message_model_id = self._special_token_id("<|message_model|>")
        self._model_end_sampling_id = self._special_token_id(
            "<|content_model_end_sampling|>"
        )

    def _special_token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        if not isinstance(tid, int) or tid == self._tokenizer.unk_token_id:
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

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        rendered = super().render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )
        ids = list(rendered.token_ids)
        indices = list(rendered.message_indices)
        sampled = [False] * len(ids)
        is_content = [False] * len(ids)

        start = 0
        while start < len(ids):
            end = start
            while end < len(ids) and indices[end] == indices[start]:
                end += 1
            message_index = indices[start]
            if message_index >= 0:
                message = messages[message_index]
                if message.get("role") == "assistant":
                    self._mark_assistant_span(
                        ids, message, start, end, sampled, is_content
                    )
            start = end

        return RenderedTokens(
            token_ids=ids,
            message_indices=indices,
            message_roles=rendered.message_roles,
            message_tool_names=rendered.message_tool_names,
            sampled_mask=sampled,
            is_content=is_content,
        )

    def _mark_assistant_span(
        self,
        ids: list[int],
        message: Message,
        start: int,
        end: int,
        sampled: list[bool],
        is_content: list[bool],
    ) -> None:
        if message.get("reasoning_content") or message.get("tool_calls"):
            self._mark_rich_assistant_span(ids, start, end, sampled, is_content)
            return
        span = ids[start:end]
        if self._content_text_id not in span:
            raise ValueError(
                "assistant message rendered without a <|content_text|> marker"
            )
        body_start = start + len(span) - 1 - span[::-1].index(self._content_text_id) + 1
        body = message.get("content") or ""
        body_ids = self._encode_body(body) if body else []
        if body_ids:
            if ids[body_start : body_start + len(body_ids)] != body_ids:
                raise ValueError(
                    "assistant body does not tokenize contiguously inside the rendering"
                )
            for offset in range(len(body_ids)):
                sampled[body_start + offset] = True
                is_content[body_start + offset] = True
        # Closing markers the model samples at inference.
        for position in range(body_start + len(body_ids), end):
            sampled[position] = True

    def _mark_rich_assistant_span(
        self,
        ids: list[int],
        start: int,
        end: int,
        sampled: list[bool],
        is_content: list[bool],
    ) -> None:
        """Mark a turn that renders thinking, text and/or tool-call blocks.

        The template opens the turn with one role tag (scaffold) and tags every
        later block of the turn; everything after that first role tag is the
        model's emission, so it is sampled and content at once.
        """
        if ids[start] != self._message_model_id:
            raise ValueError(
                "assistant message rendered without a <|message_model|> tag"
            )
        for position in range(start + 1, end):
            sampled[position] = True
            is_content[position] = True

    def _encode_body(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))


__all__ = ["Inkling2Renderer"]
