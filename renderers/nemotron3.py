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
    should_preserve_past_thinking,
    trim_to_turn_close,
)
from renderers.configs import Nemotron3RendererConfig
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


# Per-model ``ultra`` default, applied when the renderer config leaves it
# ``None``. The Nemotron-3 family ships two chat-template variants: Nano /
# Super share one; Ultra differs in the reasoning-block glue (no ``\n`` around
# ``</think>``) and the thinking-truncation boundary (drop thinking on every
# assistant turn before the last user message). BF16 and FP8 share the same
# tokenizer and template. Hard-coded keyed by
# ``tokenizer.name_or_path`` rather than probed from the live template — the
# same convention as Qwen3.5's ``_ENABLE_THINKING_DEFAULTS`` (avoids pulling
# ``apply_chat_template`` onto the construction hot path and keeps
# bring-your-own-tokenizer use working).
_ULTRA_DEFAULTS: dict[str, bool] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": False,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": False,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": True,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": True,
}


def _default_ultra(tokenizer) -> bool:
    """Hard-coded ``ultra`` default for ``tokenizer``'s model.

    Falls back to ``False`` (the Nano / Super template, and the majority of
    the family) for unknown / fine-tuned checkpoints whose ``name_or_path``
    isn't in ``_ULTRA_DEFAULTS`` — pass an explicit ``ultra=True`` for an
    Ultra fine-tune or a locally-pathed Ultra checkpoint.
    """
    return _ULTRA_DEFAULTS.get(getattr(tokenizer, "name_or_path", ""), False)


class Nemotron3Renderer:
    """Deterministic message → token renderer for Nemotron 3 models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Nemotron3RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        cfg = config or Nemotron3RendererConfig()
        # ``ultra=None`` defers to the model's known default (see
        # ``_ULTRA_DEFAULTS``). Materialise here so downstream reads see a
        # concrete bool; rebind the frozen config with the resolved value so
        # introspection sees the same.
        if cfg.ultra is None:
            cfg = cfg.model_copy(update={"ultra": _default_ultra(tokenizer)})
        self.config = cfg

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

            # Build system content: user's system text first, then tools
            if first_is_system:
                sys_content = self._render_content(messages[0].get("content")).strip()
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
            sys_content = self._render_content(messages[0].get("content")).strip()
            emit_special(self._im_start, sys_idx, is_sampled=False, is_content=False)
            sys_segments2: list[tuple[str, bool]] = [("system\n", False)]
            if sys_content:
                sys_segments2.append((sys_content, True))
            emit_text_segments(sys_segments2, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False, is_content=False)
            emit_text("\n", sys_idx, is_sampled=False, is_content=False)

        # Track the most-recent plain (non-tool-call) assistant so we can
        # preserve its reasoning while stripping reasoning from earlier
        # assistants — the Nemotron-3 template matches this pattern.
        last_plain_assistant_idx = -1
        for j in range(len(messages) - 1, -1, -1):
            if messages[j].get("role") == "assistant" and not messages[j].get(
                "tool_calls"
            ):
                last_plain_assistant_idx = j
                break

        # Ultra truncates thinking on every assistant turn *before the last
        # user message* (template rule ``loop.index0 < last_user_idx``),
        # whereas Nano/Super preserve only the last plain assistant. Compute
        # the last-user index over the normalized ``messages`` list (a leading
        # system never holds a user, so the relative comparison is unaffected).
        last_user_idx_norm = -1
        for j in range(len(messages) - 1, -1, -1):
            if messages[j].get("role") == "user":
                last_user_idx_norm = j
                break

        # ── 2. Iterate messages ─────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = self._render_content(msg.get("content")).strip()
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
                emit_text_segments(user_segments, msg_orig_idx, is_sampled=False)
                emit_special(
                    self._im_end, msg_orig_idx, is_sampled=False, is_content=False
                )
                emit_text("\n", msg_orig_idx, is_sampled=False, is_content=False)

            elif role == "assistant":
                if self.config.ultra:
                    is_last_turn = i >= last_user_idx_norm
                else:
                    is_last_turn = i >= last_plain_assistant_idx
                preserve_thinking = msg_orig_idx >= 0 and should_preserve_past_thinking(
                    original_messages,
                    msg_orig_idx,
                    preserve_all_thinking=self.config.preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self.config.preserve_thinking_between_tool_calls,
                )
                self._render_assistant(
                    msg,
                    msg_orig_idx,
                    content,
                    is_last_turn=is_last_turn,
                    preserve_thinking=preserve_thinking,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_ids=emit_ids,
                    emit_text_segments=emit_text_segments,
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
            content = self._render_content(msg.get("content")).strip()
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
        is_last_turn: bool,
        preserve_thinking: bool = False,
        emit_special,
        emit_text,
        emit_ids,
        emit_text_segments,
    ) -> None:
        # Extract reasoning_content
        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        elif "</think>" in content:
            before_think_end, after_think_end = content.split("</think>", 1)
            if "<think>" in before_think_end:
                reasoning_content = before_think_end.split("<think>")[-1].lstrip("\n")
            else:
                reasoning_content = before_think_end.lstrip("\n")
            reasoning_content = reasoning_content.rstrip("\n")
            content = after_think_end.lstrip("\n")

        reasoning_content = reasoning_content.strip()
        ultra = self.config.ultra

        # ``<|im_start|>assistant\n`` is template-injected scaffolding —
        # at inference the chat template emits these as the generation
        # prompt and the model never samples them. Marking the role tag
        # as ``is_sampled=False`` keeps the SFT loss mask aligned with
        # what the model would actually have produced. On assistant the
        # invariant ``is_content == sampled_mask`` holds.
        emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
        emit_text("assistant\n", msg_idx, is_sampled=False, is_content=False)

        # Nemotron 3 keeps reasoning on the most-recent plain assistant but
        # strips it from historical turns, which collapse to an empty
        # <think></think> block. Empty <think></think> is also emitted when
        # the turn has no reasoning at all. The trailing ``\n`` (when
        # tool_calls follow) is glued to ``content`` in a single emit_text
        # so BPE sees ``content\n`` as one chunk, matching how
        # apply_chat_template tokenises the concatenated template string.
        tool_calls = msg.get("tool_calls") or []
        # A \n is always required between the text/think block and the first
        # <tool_call>, whether the content is empty or not.
        content_suffix = "\n" if tool_calls else ""

        if reasoning_content and (
            is_last_turn
            or preserve_thinking
            or not self.config.truncate_history_thinking
        ):
            emit_special(self._think, msg_idx, is_sampled=True, is_content=True)
            # Ultra: <think>\n{reasoning}</think>{content} (no \n around </think>).
            # Nano/Super: <think>\n{reasoning}\n</think>\n{content}.
            emit_text(
                ("\n" + reasoning_content)
                if ultra
                else ("\n" + reasoning_content + "\n"),
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
            # Single \n separator (not \n\n like Qwen3.5); Ultra glues directly.
            emit_text(
                (content + content_suffix)
                if ultra
                else ("\n" + content + content_suffix),
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
        elif reasoning_content:
            # Historical assistant whose reasoning got stripped. Nano/Super keep
            # a single \n between the collapsed <think></think> and the content
            # as a marker that reasoning existed; Ultra glues content directly.
            emit_special(self._think, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
            emit_text(
                (content + content_suffix)
                if ultra
                else ("\n" + content + content_suffix),
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
        else:
            # No reasoning ever — <think></think> glued directly to content.
            emit_special(self._think, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
            emit_text(
                content + content_suffix,
                msg_idx,
                is_sampled=True,
                is_content=True,
            )

        # Tool calls (leading \n was glued to the content above; each
        # iteration's trailing \n after </tool_call> handles the
        # separator to the next block).
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})

                emit_special(self._tool_call, msg_idx, is_sampled=True, is_content=True)
                emit_text(
                    "\n<function=" + name + ">\n",
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )

                # Render arguments
                # OpenAI canonical form: arguments is a JSON string. Parse it so the
                # per-argument rendering below still works.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if isinstance(arguments, dict):
                    for arg_name, arg_value in arguments.items():
                        if isinstance(arg_value, (dict, list)):
                            value_str = json.dumps(arg_value, ensure_ascii=False)
                        else:
                            value_str = str(arg_value)
                        emit_text(
                            "<parameter="
                            + arg_name
                            + ">\n"
                            + value_str
                            + "\n</parameter>\n",
                            msg_idx,
                            is_sampled=True,
                            is_content=True,
                        )

                emit_text("</function>\n", msg_idx, is_sampled=True, is_content=True)
                emit_special(
                    self._tool_call_end, msg_idx, is_sampled=True, is_content=True
                )
                # Trailing \n after </tool_call> (Nemotron 3 specific)
                emit_text("\n", msg_idx, is_sampled=True, is_content=True)

        # ``<|im_end|>`` is the model's stop signal — it samples this to
        # end its turn, so it is part of the sampled stream. The trailing
        # ``\n`` is template-appended between turns and never sampled.
        emit_special(self._im_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

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
