"""ZAYA1 renderer — hard-coded Python mirroring Zyphra/ZAYA1-8B's Jinja template.

Notes:
- The template always emits an empty system prelude, even when the caller did not pass one.
- Multi-turn bridging must strip that synthetic BOS + empty system prelude from subsequent turns.
- Tool calls use Zyphra's XML-ish ``<zyphra_tool_call><function=...>`` format.
- Thinking can be optionally truncated from history.
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
from renderers.parsing import parse_zaya1


_TOOL_INSTRUCTIONS = """

If you choose to call a function ONLY reply in the following format with NO suffix:

<zyphra_tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</zyphra_tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <zyphra_tool_call></zyphra_tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


class Zaya1Renderer:
    """Deterministic message → token renderer for ``Zyphra/ZAYA1-8B``."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enable_thinking: bool = True,
        truncate_history_thinking: bool = False,
        preserve_all_thinking: bool = False,
        preserve_thinking_between_tool_calls: bool = False,
    ):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking
        self._truncate_history_thinking = truncate_history_thinking
        self._preserve_all_thinking = preserve_all_thinking
        self._preserve_thinking_between_tool_calls = (
            preserve_thinking_between_tool_calls
        )
        self._bos = tokenizer.bos_token_id
        self._eos = tokenizer.eos_token_id
        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")

    @property
    def supports_tools(self) -> bool:
        return True

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

        if self._bos is not None:
            emit_special(self._bos, -1, is_sampled=False)

        first_is_system = messages[0].get("role") == "system"
        system_message = str(messages[0].get("content") or "") if first_is_system else ""
        loop_messages = messages[1:] if first_is_system else messages
        loop_offset = 1 if first_is_system else 0
        last_user_idx = max(
            (j for j, m in enumerate(loop_messages) if m.get("role") == "user"),
            default=-1,
        )

        # The upstream template always defines system_message, so it always
        # emits a system block, even when the caller did not supply one.
        sys_idx = 0 if first_is_system else -1
        emit_special(self._im_start, sys_idx, is_sampled=False)
        emit_text("system\n" + system_message, sys_idx, is_sampled=False)
        if tools:
            if system_message:
                emit_text("\n\n", sys_idx, is_sampled=False)
            emit_text(self._render_tools(tools), sys_idx, is_sampled=False)
        emit_special(self._im_end, sys_idx, is_sampled=False)
        emit_text("\n", sys_idx, is_sampled=False)

        for rel_i, msg in enumerate(loop_messages):
            i = rel_i + loop_offset
            role = msg.get("role")
            content = self._string_content(msg.get("content") or "")
            if role == "assistant":
                preserve_thinking = should_preserve_past_thinking(
                    messages,
                    i,
                    preserve_all_thinking=self._preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self._preserve_thinking_between_tool_calls,
                )
                include_content = not (
                    self._truncate_history_thinking and rel_i < last_user_idx
                ) or preserve_thinking
                self._render_assistant(
                    msg,
                    i,
                    content,
                    include_content=include_content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )
            elif role in {"user", "system"}:
                emit_special(self._im_start, i, is_sampled=False)
                emit_text(f"{role}\n{content}", i, is_sampled=False)
                emit_special(self._im_end, i, is_sampled=False)
                emit_text("\n", i, is_sampled=False)
            elif role == "tool":
                self._render_tool(loop_messages, rel_i, i, content, emit_special, emit_text)
            else:
                emit_special(self._im_start, i, is_sampled=False)
                emit_text(f"{role}\n{content}", i, is_sampled=False)
                emit_special(self._im_end, i, is_sampled=False)
                emit_text("\n", i, is_sampled=False)

        if add_generation_prompt:
            emit_special(self._im_start, -1, is_sampled=False)
            if self._enable_thinking:
                emit_text("assistant\n<think>\n", -1, is_sampled=False)
            else:
                emit_text("assistant\n<think>\n</think>\n\n", -1, is_sampled=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            message_roles=[m.get("role") or "" for m in messages],
        )

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def parse_response(self, token_ids: list[int], *, tools: list[ToolSpec] | None = None) -> ParsedResponse:
        return parse_zaya1(self._tokenizer, token_ids, stop_ids={self._im_end, self._eos} if self._eos is not None else {self._im_end}, tools=tools)

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end] + ([] if self._eos is None else [self._eos])

    def bridge_to_next_turn(self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None):
        if not previous_prompt_ids or not new_messages or reject_assistant_in_extension(new_messages):
            return None
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            set(self.get_stop_token_ids()),
            synthesize_close=self._im_end,
        )
        if previous_ids is None:
            return None
        rendered = self.render(new_messages, tools=None, add_generation_prompt=True)
        # Drop BOS + the template's synthetic empty system block for extensions.
        prefix = [] if self._bos is None else [self._bos]
        empty_system = prefix + self._encode("<|im_start|>system\n<|im_end|>\n")
        ext = rendered.token_ids[len(empty_system) :] if rendered.token_ids[: len(empty_system)] == empty_system else rendered.token_ids
        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + rendered.message_indices[-len(ext) :],
            sampled_mask=[False] * total_len,
            message_roles=[m.get("role") or "" for m in new_messages],
        )

    def _render_assistant(self, msg, msg_idx, content, *, include_content, emit_special, emit_text) -> None:
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            rendered_content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        elif isinstance(content, str) and "<think>" not in content and "</think>" not in content:
            rendered_content = f"<think>\n</think>\n\n{content}"
        else:
            rendered_content = content

        if not include_content:
            rendered_content = self._truncate_thinking(rendered_content)

        tool_calls = msg.get("tool_calls") or []
        emit_special(self._im_start, msg_idx, is_sampled=False)
        emit_text("assistant\n", msg_idx, is_sampled=False)
        if tool_calls:
            body = rendered_content.strip()
            if body:
                emit_text(body + "\n\n", msg_idx, is_sampled=True)
            else:
                emit_text("<think>\n</think>\n\n", msg_idx, is_sampled=True)
            for tc in tool_calls:
                emit_text(self._render_tool_call(tc), msg_idx, is_sampled=True)
            emit_special(self._im_end, msg_idx, is_sampled=True)
            emit_text("\n", msg_idx, is_sampled=False)
        else:
            emit_text(rendered_content.strip(), msg_idx, is_sampled=True)
            emit_special(self._im_end, msg_idx, is_sampled=True)
            emit_text("\n", msg_idx, is_sampled=False)

    def _render_tool(self, loop_messages, rel_i, msg_idx, content, emit_special, emit_text) -> None:
        prev_is_tool = rel_i > 0 and loop_messages[rel_i - 1].get("role") == "tool"
        next_is_tool = rel_i + 1 < len(loop_messages) and loop_messages[rel_i + 1].get("role") == "tool"
        if not prev_is_tool:
            emit_special(self._im_start, msg_idx, is_sampled=False)
            emit_text("user\n", msg_idx, is_sampled=False)
        emit_text(f"<zyphra_tool_response>\n{content}\n</zyphra_tool_response>\n", msg_idx, is_sampled=False)
        if not next_is_tool:
            emit_special(self._im_end, msg_idx, is_sampled=False)
            emit_text("\n", msg_idx, is_sampled=False)

    def _render_tools(self, tools: list[ToolSpec]) -> str:
        text = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for raw_tool in tools:
            tool = raw_tool.get("function", raw_tool) if isinstance(raw_tool, dict) else raw_tool
            text += f"\n<function>\n<name>{tool.get('name', '')}</name>"
            if tool.get("description") is not None:
                text += f"\n<description>{str(tool['description']).strip()}</description>"
            params = tool.get("parameters") or {}
            text += "\n<parameters>"
            props = params.get("properties") if isinstance(params, dict) else None
            if isinstance(props, dict):
                for name, fields in props.items():
                    text += f"\n<parameter>\n<name>{name}</name>"
                    if fields.get("type") is not None:
                        text += f"\n<type>{fields['type']}</type>"
                    if fields.get("description") is not None:
                        text += f"\n<description>{str(fields['description']).strip()}</description>"
                    if fields.get("enum") is not None:
                        text += "\n<enum>" + json.dumps(fields["enum"], ensure_ascii=False) + "</enum>"
                    text += "\n</parameter>"
            if isinstance(params, dict) and params.get("required") is not None:
                text += "\n<required>" + json.dumps(params["required"], ensure_ascii=False) + "</required>"
            text += "\n</parameters>\n</function>"
        return text + "\n</tools>" + _TOOL_INSTRUCTIONS

    def _render_tool_call(self, tc) -> str:
        func = tc.get("function") or tc
        text = f"<zyphra_tool_call>\n<function={func.get('name', '')}>\n"
        arguments = func.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        for name, value in arguments.items():
            if isinstance(value, (dict, list)):
                value_text = json.dumps(value, ensure_ascii=False)
            else:
                value_text = str(value)
            text += f"<parameter={name}>\n{value_text}\n</parameter>\n"
        return text + "</function>\n</zyphra_tool_call>\n"

    @staticmethod
    def _string_content(content) -> str:
        if isinstance(content, list):
            return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        return str(content)

    @staticmethod
    def _truncate_thinking(content: str) -> str:
        if "</think>" in content:
            content = content.split("</think>")[-1]
        elif "<think>" in content:
            content = content.split("<think>")[0]
        return ("<think>\n</think>\n\n" + content.strip()).strip()
