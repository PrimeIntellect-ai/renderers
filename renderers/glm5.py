"""GLM-5 Renderer — hard-coded Python mirroring the GLM-5 Jinja chat template.

Key differences from Qwen family:
- Prefix: [gMASK]<sop> before all content
- Role markers: <|system|>, <|user|>, <|assistant|>, <|observation|> (no im_start/im_end)
- No end-of-message token — messages separated by next role marker
- Assistant always emits </think> as separator (even without thinking content)
- Tool calls: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
- Tool responses: <|observation|><tool_response>content</tool_response>
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
from renderers.configs import GLM5RendererConfig, GLM51RendererConfig
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
    "<tool_call>{function-name}"
    "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
    "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>"
    "...</tool_call>"
)


class GLM5Renderer:
    """Deterministic message → token renderer for GLM-5 models."""

    # GLM-5.1 flips this on: even when the most-recent assistant has no
    # reasoning content, the template wraps it with ``<think></think>``
    # instead of just emitting ``</think>`` as a separator. Subclassed in
    # GLM51Renderer; GLM-5 proper keeps this off.
    empty_think_on_last_assistant: bool = False

    # GLM-5.1 uses the same template surface and binds the same kwargs.
    # Subclassed in ``GLM51Renderer`` so the registry can dispatch on the
    # ``glm-5.1`` discriminator while sharing this implementation.
    _config_cls: type = GLM5RendererConfig

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: GLM5RendererConfig | GLM51RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or type(self)._config_cls()
        if not self.config.clear_thinking:
            implied_thinking_retention = "all"
        elif not self.config.enable_thinking:
            implied_thinking_retention = "all"
        else:
            implied_thinking_retention = "tool_cycle"
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config,
            implied_thinking_retention,
        )

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
        self._tool_response_tok = self._token_id("<tool_response>")
        self._tool_response_end_tok = self._token_id("</tool_response>")
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

    @staticmethod
    def _format_tool_spec(tool: ToolSpec) -> str:
        """Serialise a single tool spec to the exact JSON the Jinja template
        emits. GLM-5 just ``tojson``s the dict as passed; GLM-5.1 overrides
        this to unwrap the OpenAI-style ``{"type":"function","function":…}``
        envelope and filter internal-only keys first.
        """
        return json.dumps(tool, ensure_ascii=False)

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
        # ``[gMASK]<sop>`` is unconditional template scaffolding at the
        # very start of the stream — the model never samples these and
        # they are not part of any message body.
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
                tool_text += self._format_tool_spec(tool) + "\n"
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
                emit_text(content, i, is_sampled=False, is_content=True)

            elif role == "user":
                emit_special(
                    self._user,
                    i,
                    is_sampled=closes_assistant_turn,
                    is_content=False,
                )
                emit_text(content, i, is_sampled=False, is_content=True)

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    content,
                    last_ui,
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
        # Gen prompt tokens are what the chat template prepends before
        # sampling starts — the model continues from these, never emits
        # them. Always is_sampled=False / is_content=False.
        if add_generation_prompt:
            emit_special(self._assistant, -1, is_sampled=False, is_content=False)
            if self.config.enable_thinking:
                emit_special(self._think, -1, is_sampled=False, is_content=False)
            else:
                emit_special(self._think_end, -1, is_sampled=False, is_content=False)

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

        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention,
            new_messages,
        ):
            return None

        # GLM has no per-turn close token. An assistant turn ends when the
        # next turn's role marker appears, OR the model emits <|endoftext|>.
        # vLLM includes these in ``stop_token_ids`` so a clean stop leaves
        # one of {endoftext, user, observation} at the tail of
        # previous_completion_ids. Truncation means none is there yet.
        previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
        stop_ids = {self._endoftext, self._user, self._observation}
        if (
            not previous_ids[len(previous_prompt_ids) :]
            or previous_ids[-1] not in stop_ids
        ):
            # Truncation: synthesise <|endoftext|> as the canonical turn end.
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
                # Dedup: model already emitted <|user|> as its stop token.
                if not (i == 0 and last_prev == self._user):
                    emit_special(self._user, i)
                emit_text(content, i, is_content=True)
            elif role == "system":
                emit_special(self._system, i)
                emit_text(content, i, is_content=True)
            elif role == "tool":
                prev_is_tool = i > 0 and new_messages[i - 1].get("role") == "tool"
                if i == 0 and last_prev == self._observation:
                    # Model already emitted <|observation|>; don't repeat.
                    pass
                elif not prev_is_tool:
                    emit_special(self._observation, i)
                emit_special(self._tool_response_tok, i)
                emit_text(content, i, is_content=True)
                emit_special(self._tool_response_end_tok, i)
            else:
                return None

        # Generation prompt — match the gen-prompt branch of ``render()``.
        emit_special(self._assistant, -1)
        if self.config.enable_thinking:
            emit_special(self._think, -1)
        else:
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

    def _render_assistant(
        self,
        msg,
        msg_idx,
        content,
        last_user_index,
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

        # ``<|assistant|>`` is template-injected: the chat template emits
        # it as the generation prompt at inference, and the model never
        # samples it. Same for the ``<think>`` open / standalone
        # ``</think>`` separator that the template wraps around the
        # assistant body — see the per-branch comments below.
        #
        # Invariant on assistant tokens: ``is_content == sampled_mask``.
        # Every scaffold token here gets ``is_sampled=False/is_content=False``;
        # every model-sampled emit gets both True.
        emit_special(self._assistant, msg_idx, is_sampled=False, is_content=False)

        # Chat-template default: keep ``<think>`` only on the in-flight cycle
        # (post-last-user). Past-cycle assistants drop their reasoning.
        # ``clear_thinking=False`` mirrors
        # the template's per-call ``clear_thinking is defined and not
        # clear_thinking`` gate: a chat_template_kwarg surface for the
        # same behaviour, gated explicitly by the caller per render.
        include_thinking = (
            msg_idx > last_user_index or not self.config.clear_thinking
        ) and reasoning_content

        if include_thinking:
            # ``<think>`` matches the gen-prompt's trailing token at
            # inference (gen prompt = ``<|assistant|><think>``), so it's
            # template-injected scaffolding. The reasoning text and the
            # closing ``</think>`` are what the model actually samples.
            emit_special(self._think, msg_idx, is_sampled=False, is_content=False)
            emit_text(
                reasoning_content.strip(), msg_idx, is_sampled=True, is_content=True
            )
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
        elif (
            self.empty_think_on_last_assistant
            and msg_idx > last_user_index
            and self.config.enable_thinking
        ):
            # GLM-5.1: wrap the last assistant with an empty <think></think>
            # even without reasoning, matching the Jinja template. With
            # ``enable_thinking=True`` the gen prompt already includes
            # ``<think>``; the model then samples ``</think>`` to close an
            # empty think block. So ``<think>`` is scaffolding,
            # ``</think>`` is sampled.
            #
            # When ``enable_thinking=False`` the GLM-5.1 template skips
            # the opening ``<think>`` for the most-recent assistant too
            # — it emits only the lone ``</think>`` separator (and the
            # gen prompt likewise switches to ``</think>``). Fall
            # through to the else branch below so we match.
            emit_special(self._think, msg_idx, is_sampled=False, is_content=False)
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
        else:
            # Lone ``</think>`` separator the template injects when no
            # reasoning is rendered (historical assistants, GLM-5 default
            # with no thinking). Not sampled.
            emit_special(self._think_end, msg_idx, is_sampled=False, is_content=False)

        if content.strip():
            emit_text(content.strip(), msg_idx, is_sampled=True, is_content=True)

        # Tool calls (directly after content, no newlines). All of these
        # are the model's sampled output — both is_sampled and is_content
        # are True across the entire tool-call span.
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function") or tc
            name = func.get("name", "")
            arguments = func.get("arguments", {})

            emit_special(self._tool_call_tok, msg_idx, is_sampled=True, is_content=True)
            emit_text(name, msg_idx, is_sampled=True, is_content=True)
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

        emit_special(
            self._tool_response_tok, msg_idx, is_sampled=False, is_content=False
        )
        emit_text(content, msg_idx, is_sampled=False, is_content=True)
        emit_special(
            self._tool_response_end_tok, msg_idx, is_sampled=False, is_content=False
        )


class GLM51Renderer(GLM5Renderer):
    """Deterministic message → token renderer for GLM-5.1 models.

    Diverges from GLM-5 in two places:

    - The most-recent assistant turn is wrapped with an empty
      ``<think></think>`` block even when no ``reasoning_content`` is
      supplied. Historical assistants collapse to just ``</think>``.
    - Tool specs are unwrapped before serialisation: if the caller
      passes the OpenAI ``{"type":"function","function":{…}}`` envelope,
      only the inner ``function`` payload is rendered (minus
      ``defer_loading`` / ``strict`` internal keys).
    """

    empty_think_on_last_assistant = True
    _config_cls = GLM51RendererConfig

    @staticmethod
    def _format_tool_spec(tool: ToolSpec) -> str:
        spec = tool["function"] if "function" in tool else tool
        spec = {k: v for k, v in spec.items() if k not in ("defer_loading", "strict")}
        return json.dumps(spec, ensure_ascii=False)
