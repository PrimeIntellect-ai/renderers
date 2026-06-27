"""Nemotron 3 Renderer — hard-coded Python that mirrors the Nemotron 3 chat template.

Nemotron 3 uses the same <|im_start|>/<|im_end|> format as Qwen3.5 but differs in:

1. Tool declarations: XML format inside <tools>...</tools> (not JSON-per-line).
2. System message ordering: system prompt goes BEFORE tools block.
3. Thinking block scope: <think></think> is prepended to ALL assistant messages
   that lack thinking content (not just those after the last user query).
4. Think separator: single \\n after </think> (not \\n\\n like Qwen3.5).
5. Empty system message: always prepends an empty system message if none exists.
6. Disable-thinking generation suffix: <think></think> with no trailing newlines.
7. Tool response format: trailing newline after </tool_response>.
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
    trim_to_turn_close,
)
from renderers.configs import Nemotron3RendererConfig, Nemotron3UltraRendererConfig
from renderers.parsing import parse_qwen35

# ---------------------------------------------------------------------------
# Tool system prompt constants
# ---------------------------------------------------------------------------

_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"

_TOOLS_FOOTER = "\n</tools>"

_TOOLS_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1"
    "\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter"
    "\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:"
    "\n- Function calls MUST follow the specified format:"
    " an inner <function=...></function> block must be nested within"
    " <tool_call></tool_call> XML tags"
    "\n- Required parameters MUST be specified"
    "\n- You may provide optional reasoning for your function call"
    " in natural language BEFORE the function call, but NOT after"
    "\n- If there is no function call available, answer the question like normal"
    " with your current knowledge and do not tell the user about function calls"
    "\n</IMPORTANT>"
)


def _render_extra_keys(obj: dict[str, Any], handled_keys: set[str]) -> list[str]:
    """Render extra dict keys as XML, mirroring the HF template's render_extra_keys macro.

    Dicts and lists are JSON-encoded; scalars are string-coerced.
    """
    lines: list[str] = []
    for key, value in obj.items():
        if key in handled_keys:
            continue
        if isinstance(value, (dict, list)):
            lines.append(f"<{key}>{json.dumps(value)}</{key}>")
        else:
            lines.append(f"<{key}>{value!s}</{key}>")
    return lines


# The Nemotron-3 family ships two chat-template variants. Nano / Super share
# one (renderer ``Nemotron3Renderer`` / config ``name="nemotron-3"``); Ultra
# differs in the reasoning-block glue — no ``\n`` around ``</think>`` — and is
# the ``Nemotron3UltraRenderer`` subclass (``name="nemotron-3-ultra"``). Which
# variant a checkpoint uses is carried by ``MODEL_RENDERER_MAP``, so the right
# renderer class is constructed and the variant is encoded by the class itself.


def _is_super(tokenizer) -> bool:
    """Does this checkpoint use the **Super** flavour of the shared Nano/Super
    template — i.e. the one whose Jinja defines the ``low_effort`` kwarg?

    Nano and Super share one config (``nemotron-3``), so the model name is the
    only signal that separates them. Detected by substring; unknown / fine-tuned
    checkpoints default to ``False`` so ``low_effort`` is a no-op there —
    matching how the Nano template silently ignores it.
    """
    return "super" in (getattr(tokenizer, "name_or_path", "") or "").lower()


class Nemotron3Renderer:
    """Deterministic message → token renderer for Nemotron-3 Nano / Super.

    The Ultra variant (distinct ``</think>`` glue) is the
    :class:`Nemotron3UltraRenderer` subclass below; both are registered under
    their own discriminator and differ only by the class-level hooks here.
    """

    # Variant hooks (overridden by ``Nemotron3UltraRenderer``): the default
    # config to build when none is passed, and whether to use Ultra's
    # reasoning-block glue.
    _config_cls: type = Nemotron3RendererConfig
    _ultra: bool = False

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Nemotron3RendererConfig | Nemotron3UltraRendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        cfg = config or type(self)._config_cls()
        self.config = cfg
        if not cfg.truncate_history_thinking:
            implied_thinking_retention = "all"
        elif not cfg.enable_thinking:
            implied_thinking_retention = "all"
        else:
            implied_thinking_retention = "tool_cycle"
        self.effective_thinking_retention = resolve_thinking_retention(
            cfg,
            implied_thinking_retention,
        )

        # Resolve the per-variant reasoning-effort hint appended to the last
        # user message. Ultra honours ``medium_effort``; Super honours
        # ``low_effort``; Nano honours neither. The non-matching kwarg is
        # silently ignored (empty hint), exactly as ``apply_chat_template``
        # ignores a template variable the variant's Jinja never defines.
        if self._ultra:
            self._effort_hint = (
                "\n\n{reasoning effort: efficient}"
                if getattr(cfg, "medium_effort", False)
                else ""
            )
        elif getattr(cfg, "low_effort", False) and _is_super(tokenizer):
            self._effort_hint = "\n\n{reasoning effort: low}"
        else:
            self._effort_hint = ""

        # Look up special token IDs from the tokenizer (not hardcoded).
        # <|endoftext|> is optional: Nemotron-3 Nano / Super tokenizers ship
        # <|im_end|> as the sole EOS; older / larger variants additionally
        # include <|endoftext|>. Both work with the same chat template.
        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>", optional=True)
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")

    def _token_id(self, token: str, *, optional: bool = False) -> int | None:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        if not isinstance(tid, int) or tid == self._tokenizer.unk_token_id:
            if optional:
                return None
            raise AssertionError(
                f"Special token {token!r} not found in tokenizer vocabulary"
            )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Content rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_content(content: Any) -> str:
        """Render message content to a text string (before tokenization)."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        parts.append(item["text"])
                    else:
                        raise ValueError(f"Unexpected content item: {item}")
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    # ------------------------------------------------------------------
    # Tool declaration formatting (XML, Nemotron 3 style)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tool_declaration(tool: ToolSpec) -> str:
        """Format a single tool declaration in Nemotron 3 XML format."""
        # Accept the OpenAI-style ``{"type":"function","function":{...}}``
        # envelope by unwrapping before formatting.
        if "function" in tool and isinstance(tool["function"], dict):
            tool = tool["function"]
        lines = [
            "<function>",
            f"<name>{tool['name']}</name>",
        ]
        description = tool.get("description", "").strip()
        if description:
            lines.append(f"<description>{description}</description>")
        lines.append("<parameters>")
        params = tool.get("parameters") or {}
        if isinstance(params, dict) and "properties" in params:
            for param_name, param_fields in params["properties"].items():
                lines.append("<parameter>")
                lines.append(f"<name>{param_name}</name>")
                if "type" in param_fields:
                    lines.append(f"<type>{param_fields['type']!s}</type>")
                if "description" in param_fields:
                    lines.append(
                        f"<description>{param_fields['description'].strip()}</description>"
                    )
                if "enum" in param_fields:
                    lines.append(f"<enum>{json.dumps(param_fields['enum'])}</enum>")
                lines.extend(
                    _render_extra_keys(
                        param_fields, {"name", "type", "description", "enum"}
                    )
                )
                lines.append("</parameter>")
        if isinstance(params, dict):
            lines.extend(_render_extra_keys(params, {"type", "properties", "required"}))
        if isinstance(params, dict) and "required" in params:
            lines.append(f"<required>{json.dumps(params['required'])}</required>")
        lines.append("</parameters>")
        lines.extend(
            _render_extra_keys(tool, {"type", "name", "description", "parameters"})
        )
        lines.append("</function>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Message normalization
    # ------------------------------------------------------------------

    def _normalize_messages(
        self, messages: list[Message]
    ) -> tuple[list[Message], bool]:
        """Prepend empty system message if none exists.

        Nemotron 3's HF template always outputs a system message block even
        when none is provided. Returns ``(messages, auto_injected)`` so the
        caller can emit the injected system's tokens with ``msg_idx=-1``
        (keeping message_indices aligned with the caller's input list —
        ``build_training_sample`` relies on this).
        """
        if not messages or messages[0].get("role") != "system":
            return [{"role": "system", "content": ""}] + list(messages), True
        return list(messages), False

    # ------------------------------------------------------------------
    # Core render method
    # ------------------------------------------------------------------

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        original_messages = list(messages)
        # Always ensure an empty system message is present.
        messages, auto_system_injected = self._normalize_messages(messages)
        # Offset to map indices in the normalized list back to the caller's
        # original message list. The injected system itself uses msg_idx=-1
        # (sentinel) so build_training_sample can't dereference it.
        idx_offset = -1 if auto_system_injected else 0

        def orig_idx(i: int) -> int:
            if auto_system_injected and i == 0:
                return -1
            return i + idx_offset

        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit_ids(
            ids: list[int], msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

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
            emit_ids(
                self._encode(text),
                msg_idx,
                is_sampled=is_sampled,
                is_content=is_content,
            )

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

        # ── 1. System message + optional tools ──────────────────────
        first_is_system = messages[0].get("role") == "system"

        if tools:
            # Nemotron 3: system prompt BEFORE tools block
            sys_idx = orig_idx(0) if first_is_system else -1

            emit_special(self._im_start, sys_idx, is_sampled=False, is_content=False)

            # Build system content: user's system text first, then tools.
            # The template emits ``system_message`` verbatim (no trim) and
            # gates the ``\n\n`` separator on its raw length, so keep the
            # caller's content unstripped.
            if first_is_system:
                sys_content = self._render_content(messages[0].get("content"))
            else:
                sys_content = ""

            tool_declarations = "\n".join(
                self._format_tool_declaration(t) for t in tools
            )
            tools_block = (
                _TOOLS_HEADER
                + "\n"
                + tool_declarations
                + _TOOLS_FOOTER
                + _TOOLS_INSTRUCTIONS
            )

            # Body = caller's system text only; tools block (header, per-
            # tool XML, footer, instructions) is scaffold.
            sys_segments: list[tuple[str, bool]] = [("system\n", False)]
            if sys_content:
                sys_segments.append((sys_content, True))
                sys_segments.append(("\n\n", False))
            sys_segments.append((tools_block, False))
            emit_text_segments(sys_segments, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False, is_content=False)
            emit_text("\n", sys_idx, is_sampled=False, is_content=False)

        elif first_is_system:
            sys_idx = orig_idx(0)
            sys_content = self._render_content(messages[0].get("content"))
            emit_special(self._im_start, sys_idx, is_sampled=False, is_content=False)
            sys_segments2: list[tuple[str, bool]] = [("system\n", False)]
            if sys_content:
                sys_segments2.append((sys_content, True))
            emit_text_segments(sys_segments2, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False, is_content=False)
            emit_text("\n", sys_idx, is_sampled=False, is_content=False)

        # All Nemotron-3 variants (Nano / Super / Ultra) truncate historical
        # thinking on every assistant turn *before the last user message* —
        # the template rule ``truncate_history_thinking and loop.index0 <
        # last_user_idx`` is byte-identical across the three chat templates.
        # Compute the last-user index over the normalized ``messages`` list (a
        # leading system never holds a user, so the relative comparison is
        # unaffected).
        last_user_idx_norm = -1
        for j in range(len(messages) - 1, -1, -1):
            if messages[j].get("role") == "user":
                last_user_idx_norm = j
                break

        # ── 2. Iterate messages ─────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]
            # Keep content unstripped: the template emits user / system / tool
            # content verbatim, and assistant trimming happens inside
            # ``_assistant_body`` exactly where the template applies it.
            content = self._render_content(msg.get("content"))
            msg_orig_idx = orig_idx(i)

            if role == "system":
                if i != 0:
                    raise ValueError("System message must be at the beginning.")
                continue  # Already handled above

            elif role == "user":
                emit_special(
                    self._im_start, msg_orig_idx, is_sampled=False, is_content=False
                )
                user_segments: list[tuple[str, bool]] = [("user\n", False)]
                if content:
                    user_segments.append((content, True))
                # Reasoning-effort hint rides on the LAST user message only,
                # glued to the content so BPE sees them as one chunk (matching
                # the template's ``content + '\n\n{reasoning effort: …}'``). It
                # is template scaffold, not caller content → is_content=False.
                if self._effort_hint and i == last_user_idx_norm:
                    user_segments.append((self._effort_hint, False))
                emit_text_segments(user_segments, msg_orig_idx, is_sampled=False)
                emit_special(
                    self._im_end, msg_orig_idx, is_sampled=False, is_content=False
                )
                emit_text("\n", msg_orig_idx, is_sampled=False, is_content=False)

            elif role == "assistant":
                # Template: ``include_content = not (truncate_history_thinking
                # and loop.index0 < last_user_idx)``.
                include_content = (
                    not self.config.truncate_history_thinking or i >= last_user_idx_norm
                )
                self._render_assistant(
                    msg,
                    msg_orig_idx,
                    content,
                    include_content=include_content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                self._render_tool(
                    messages,
                    i,
                    content,
                    msg_orig_idx=msg_orig_idx,
                    auto_system_injected=auto_system_injected,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

            else:
                raise ValueError(f"Unexpected message role: {role}")

        # ── 3. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_start, -1, is_sampled=False, is_content=False)
            emit_text("assistant\n", -1, is_sampled=False, is_content=False)
            if self.config.enable_thinking:
                emit_special(self._think, -1, is_sampled=False, is_content=False)
                emit_text("\n", -1, is_sampled=False, is_content=False)
            else:
                # Disable-thinking suffix: <think></think> with no trailing newlines
                emit_special(self._think, -1, is_sampled=False, is_content=False)
                emit_special(self._think_end, -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in original_messages],
            message_tool_names=extract_message_tool_names(original_messages),
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
        stop_ids = {self._im_end}
        if self._endoftext is not None:
            stop_ids.add(self._endoftext)
        return parse_qwen35(
            self._tokenizer,
            token_ids,
            stop_ids=stop_ids,
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
            tools=tools,
        )

    def get_stop_token_ids(self) -> list[int]:
        stop = [self._im_end]
        if self._endoftext is not None:
            stop.append(self._endoftext)
        return stop

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
            # An active effort hint rides on the *last* user message. Appending
            # a new turn can move which user is last, which would strand the
            # hint on the frozen previous prompt — the append-only bridge can't
            # rewrite it. Bail so the caller does a full, correct re-render.
            or self._effort_hint
        ):
            return None

        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention,
            new_messages,
        ):
            return None

        close_ids: set[int] = {self._im_end}
        if self._endoftext is not None:
            close_ids.add(self._endoftext)
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            close_ids,
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
        # and read each step's own body mask.
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

        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            # Unstripped — the template emits user / system / tool content
            # verbatim (see :meth:`render`).
            content = self._render_content(msg.get("content"))
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
                    msg_orig_idx=i,
                    auto_system_injected=False,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )
            else:
                return None

        # Generation prompt.
        emit_special(self._im_start, -1)
        emit_text("assistant\n", -1)
        if self.config.enable_thinking:
            emit_special(self._think, -1)
            emit_text("\n", -1)
        else:
            emit_special(self._think, -1)
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

    # ------------------------------------------------------------------
    # Assistant message rendering
    # ------------------------------------------------------------------

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        include_content: bool,
        emit_special,
        emit_text,
    ) -> None:
        # ``<|im_start|>assistant\n`` is template-injected scaffolding —
        # at inference the chat template emits these as the generation
        # prompt and the model never samples them. Marking the role tag
        # as ``is_sampled=False`` keeps the SFT loss mask aligned with
        # what the model would actually have produced. On assistant the
        # invariant ``is_content == sampled_mask`` holds.
        emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
        emit_text("assistant\n", msg_idx, is_sampled=False, is_content=False)

        # Build the body (everything between ``assistant\n`` and ``<|im_end|>``)
        # as a single string mirroring the chat template's own string algebra,
        # then tokenise it in one pass. The ``<think>`` / ``</think>`` /
        # ``<tool_call>`` / ``</tool_call>`` markers are added tokens, so the
        # tokenizer isolates them — encoding the assembled body yields the same
        # ids as ``apply_chat_template`` (which likewise encodes a rendered
        # string). The whole body is sampled content; ``<|im_end|>`` is the
        # model's stop signal (sampled), and the inter-turn ``\n`` is not.
        body = self._assistant_body(msg, content, include_content=include_content)
        if body:
            emit_text(body, msg_idx, is_sampled=True, is_content=True)
        emit_special(self._im_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

    def _assistant_body(
        self, msg: Message, raw_content: str, *, include_content: bool
    ) -> str:
        """Assemble the assistant body string exactly as the chat template.

        ``include_content`` is the template's ``not (truncate_history_thinking
        and loop.index0 < last_user_idx)`` (already OR-ed with the
        ``thinking_retention`` override by the caller): ``True`` keeps the
        full think+content block,
        ``False`` collapses historical thinking to an empty ``<think></think>``.
        """
        ultra = self._ultra

        # 1. Assemble ``content`` — wrap a ``reasoning_content`` field in
        #    <think> tags (raw, not stripped: interior whitespace is part of
        #    the reasoning), else prepend an empty <think></think> only when
        #    the content carries no inline think tags of its own (which are
        #    passed through verbatim, like the template).
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            if ultra:
                content = "<think>\n" + reasoning + "</think>" + raw_content
            else:
                content = "<think>\n" + reasoning + "\n</think>\n" + raw_content
        else:
            content = raw_content
            if "<think>" not in content and "</think>" not in content:
                content = "<think></think>" + content

        tool_calls = msg.get("tool_calls") or []

        if tool_calls:
            parts: list[str] = []
            if content.strip():
                if include_content:
                    parts.append(content.strip() + "\n")
                else:
                    # Drop historical thinking: keep only what follows the last
                    # </think> (or precedes a dangling <think>), then re-stamp
                    # an empty block. Nano/Super trim the remainder; Ultra glues
                    # it raw (its template omits the trailing ``| trim``).
                    c = content
                    if "</think>" in c:
                        c = c.split("</think>")[-1]
                    elif "<think>" in c:
                        c = c.split("<think>")[0]
                    c = "<think></think>" + (c if ultra else c.strip())
                    if c:
                        parts.append(c + "\n")
            else:
                # Non-string / empty content: bare collapsed think block, no \n.
                parts.append("<think></think>")
            for tc in tool_calls:
                parts.append(self._format_tool_call(tc))
            return "".join(parts)

        # No tool calls.
        if include_content:
            return content.strip()
        c = content
        if "<think>" in c and "</think>" in c:
            c = "<think></think>" + c.split("</think>")[-1]
        return c.strip()

    @staticmethod
    def _format_tool_call(tc: dict[str, Any]) -> str:
        """Render one tool call as ``<tool_call>…</tool_call>\\n`` XML."""
        func = tc.get("function") or tc
        name = func.get("name", "")
        arguments = func.get("arguments", {})
        # OpenAI canonical form: arguments is a JSON string. Parse it so the
        # per-argument rendering below still works.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        parts = ["<tool_call>\n<function=" + name + ">\n"]
        if isinstance(arguments, dict):
            for arg_name, arg_value in arguments.items():
                if isinstance(arg_value, (dict, list)):
                    value_str = json.dumps(arg_value, ensure_ascii=False)
                else:
                    value_str = str(arg_value)
                parts.append(
                    "<parameter=" + arg_name + ">\n" + value_str + "\n</parameter>\n"
                )
        parts.append("</function>\n</tool_call>\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Tool message rendering
    # ------------------------------------------------------------------

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        content: str,
        *,
        msg_orig_idx: int,
        auto_system_injected: bool,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False. The ``content``
        # body bytes get ``is_content=True``; the surrounding wrap is
        # scaffold.
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )
        oi = msg_orig_idx

        if not prev_is_tool:
            emit_special(self._im_start, oi, is_sampled=False, is_content=False)
            emit_text("user\n", oi, is_sampled=False, is_content=False)
        # else: the previous tool's trailing \n already provides the
        # separator into this block.

        emit_special(self._tool_response, oi, is_sampled=False, is_content=False)
        emit_text_segments(
            [("\n", False), (content, True), ("\n", False)], oi, is_sampled=False
        )
        emit_special(self._tool_response_end, oi, is_sampled=False, is_content=False)
        # Nemotron 3: trailing \n after </tool_response>
        emit_text("\n", oi, is_sampled=False, is_content=False)

        if not next_is_tool:
            emit_special(self._im_end, oi, is_sampled=False, is_content=False)
            emit_text("\n", oi, is_sampled=False, is_content=False)


class Nemotron3UltraRenderer(Nemotron3Renderer):
    """Renderer for Nemotron-3 **Ultra**.

    Identical to :class:`Nemotron3Renderer` except the reasoning block is glued
    as ``<think>\\n{reasoning}</think>{content}`` (no ``\\n`` around
    ``</think>``) and truncated historical turns collapse to
    ``<think></think>{content}`` (no ``\\n``) — the difference is carried by the
    ``_ultra`` class hook. Honours the Ultra-only ``medium_effort`` kwarg.
    """

    _config_cls = Nemotron3UltraRendererConfig
    _ultra = True
