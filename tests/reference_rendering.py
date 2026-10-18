"""Reference-oracle registry for the shared test barrage.

The oracle is selected from the resolved renderer, not assumed to be Hugging
Face ``apply_chat_template``. Most checkpoints currently route to that adapter,
DeepSeek V4 routes to its shipped Python-encoder contract, and GPT-OSS routes to
Harmony. ``render_reference`` is the single invocation path for all three.

The compact DSV4 implementation below is deliberately test-side and independent
of ``renderers.deepseek_v4``.  It mirrors the public chat/tool branches of the
official encoder needed by the shared barrage; the official repository fixtures
remain covered separately in ``test_deepseek_v4.py``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Protocol

from renderers.base import MODEL_RENDERER_MAP


DEEPSEEK_V4_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"


class OracleRenderer(Protocol):
    """Common callable contract implemented by every reference oracle."""

    def __call__(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[int]: ...


@dataclass(frozen=True)
class ReferenceOracle:
    """Named adapter that renders one conversation to reference token IDs."""

    name: str
    render: OracleRenderer


_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_LATEST_REMINDER = "<｜latest_reminder｜>"
_THINK_START = "<think>"
_THINK_END = "</think>"
_DSML = "｜DSML｜"

_TASK_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}

_REASONING_EFFORT_PROMPTS = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively "
        "decompose the problem to resolve the root cause, rigorously "
        "stress-testing your logic against all potential paths, edge cases, "
        "and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting "
        "every intermediate step, considered alternative, and rejected "
        "hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
        "uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely "
        "nothing to chance: exhaustively decompose the problem into its most "
        "fundamental components, trace every causal chain to its root, and "
        "resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the "
        "solution from multiple angles and are certain that no assumption "
        "remains unchecked and no error remains undiscovered.\n\n"
    ),
}

_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _function(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    function = spec.get("function")
    return function if isinstance(function, Mapping) else spec


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)
    return "\n\n".join(
        str(part.get("text", ""))
        for part in value
        if isinstance(part, Mapping) and part.get("type") == "text"
    )


def _render_tools(tools: list[dict[str, Any]]) -> str:
    return _TOOLS_TEMPLATE.format(
        dsml=_DSML,
        think_start=_THINK_START,
        think_end=_THINK_END,
        tool_schemas="\n".join(_json(dict(_function(tool))) for tool in tools),
    )


def _render_tool_call(tool_call: Mapping[str, Any]) -> str:
    function = _function(tool_call)
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {"arguments": raw_arguments}
    elif isinstance(raw_arguments, Mapping):
        # Renderer inputs allow decoded dictionaries even though the OpenAI
        # wire format normally carries a JSON string.
        arguments = dict(raw_arguments)
    else:
        arguments = {"arguments": raw_arguments}

    parameters = []
    for key, value in arguments.items():
        is_string = isinstance(value, str)
        encoded = value if is_string else _json(value)
        parameters.append(
            f'<{_DSML}parameter name="{key}" '
            f'string="{str(is_string).lower()}">{encoded}'
            f"</{_DSML}parameter>"
        )
    return (
        f'<{_DSML}invoke name="{function.get("name")}">\n'
        + "\n".join(parameters)
        + f"\n</{_DSML}invoke>"
    )


def _merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for original in messages:
        message = copy.deepcopy(original)
        role = message.get("role")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": message.get("content", ""),
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
        elif role == "user":
            block = {"type": "text", "text": _text(message.get("content"))}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(block)
            else:
                message["content"] = block["text"]
                message["content_blocks"] = [block]
                merged.append(message)
        else:
            merged.append(message)

    call_order: dict[str, int] = {}
    for message in merged:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            call_order = {}
            for index, tool_call in enumerate(message["tool_calls"]):
                function = _function(tool_call)
                call_id = tool_call.get("id") or function.get("id")
                if call_id:
                    call_order[str(call_id)] = index
        elif message.get("role") == "user" and message.get("content_blocks"):
            blocks = message["content_blocks"]
            results = [b for b in blocks if b.get("type") == "tool_result"]
            if len(results) > 1 and call_order:
                results.sort(
                    key=lambda block: call_order.get(block.get("tool_use_id", ""), 0)
                )
                ordered = iter(results)
                message["content_blocks"] = [
                    next(ordered) if block.get("type") == "tool_result" else block
                    for block in blocks
                ]
    return merged


def _drop_historical_thinking(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    last_user = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") in {"user", "developer"}
        ),
        default=-1,
    )
    kept = []
    for index, original in enumerate(messages):
        role = original.get("role")
        if role in {"user", "system", "tool", "latest_reminder"} or index >= last_user:
            kept.append(original)
        elif role == "assistant":
            message = copy.copy(original)
            message.pop("reasoning_content", None)
            kept.append(message)
    return kept


def _render_deepseek_v4_reference(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    enable_thinking: bool,
    drop_thinking: bool,
    reasoning_effort: str,
) -> str:
    messages = copy.deepcopy(messages)

    if tools:
        target: dict[str, Any] | None = next(
            (m for m in messages if m.get("role") == "developer"),
            None,
        )
        if target is None:
            target = next(
                (m for m in messages if m.get("role") == "system"),
                None,
            )
        if target is None:
            target = {"role": "system", "content": ""}
            messages.insert(0, target)
        target["tools"] = copy.deepcopy(tools)

    messages = _merge_tool_messages(messages)
    effective_drop = drop_thinking and not any(m.get("tools") for m in messages)
    if enable_thinking and effective_drop:
        messages = _drop_historical_thinking(messages)

    last_user = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") in {"user", "developer"}
        ),
        default=-1,
    )
    prompt = _BOS
    if enable_thinking:
        prompt += _REASONING_EFFORT_PROMPTS[reasoning_effort]

    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            prompt += _text(content)
            if message.get("tools"):
                prompt += "\n\n" + _render_tools(message["tools"])
            if message.get("response_format") is not None:
                prompt += (
                    "\n\n## Response Format:\n\n"
                    "You MUST strictly adhere to the following schema to reply:\n"
                    + _json(message["response_format"])
                )
        elif role == "developer":
            prompt += _USER + _text(content)
            if message.get("tools"):
                prompt += "\n\n" + _render_tools(message["tools"])
            if message.get("response_format") is not None:
                prompt += (
                    "\n\n## Response Format:\n\n"
                    "You MUST strictly adhere to the following schema to reply:\n"
                    + _json(message["response_format"])
                )
        elif role == "user":
            prompt += _USER
            rendered_blocks = []
            for block in message.get("content_blocks") or []:
                if block.get("type") == "text":
                    rendered_blocks.append(_text(block.get("text")))
                elif block.get("type") == "tool_result":
                    rendered_blocks.append(
                        f"<tool_result>{_text(block.get('content'))}</tool_result>"
                    )
            prompt += "\n\n".join(rendered_blocks)
        elif role == "latest_reminder":
            prompt += _LATEST_REMINDER + _text(content)
        elif role == "assistant":
            previous_has_task = (
                index > 0 and messages[index - 1].get("task") is not None
            )
            keep_reasoning = (
                enable_thinking
                and not previous_has_task
                and (not effective_drop or index > last_user)
            )
            if keep_reasoning:
                prompt += str(message.get("reasoning_content") or "") + _THINK_END
            prompt += _text(content)
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                prompt += (
                    f"\n\n<{_DSML}tool_calls>\n"
                    + "\n".join(_render_tool_call(call) for call in tool_calls)
                    + f"\n</{_DSML}tool_calls>"
                )
            if not message.get("wo_eos", False):
                prompt += _EOS
        else:
            raise ValueError(f"Unsupported DeepSeek V4 reference role: {role!r}")

        next_role = (
            messages[index + 1].get("role") if index + 1 < len(messages) else None
        )
        if next_role is not None and next_role not in {
            "assistant",
            "latest_reminder",
        }:
            continue

        task = message.get("task")
        if task is not None:
            if task not in _TASK_TOKENS:
                raise ValueError(f"Invalid DeepSeek V4 reference task: {task!r}")
            if task != "action":
                prompt += _TASK_TOKENS[task]
            else:
                prompt += _ASSISTANT
                prompt += _THINK_START if enable_thinking else _THINK_END
                prompt += _TASK_TOKENS[task]
            continue

        if (next_role is None and not add_generation_prompt) or role not in {
            "user",
            "developer",
        }:
            continue
        prompt += _ASSISTANT
        if enable_thinking and (not effective_drop or index >= last_user):
            prompt += _THINK_START
        else:
            prompt += _THINK_END

    return prompt


def _render_hugging_face(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        **kwargs,
    )
    if isinstance(result, dict):
        return list(result["input_ids"])
    if isinstance(result, str):
        return list(tokenizer.encode(result, add_special_tokens=False))
    return list(result)


def _render_deepseek_v4(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    kwargs = dict(kwargs)
    text = _render_deepseek_v4_reference(
        messages,
        tools=kwargs.pop("tools", None),
        add_generation_prompt=kwargs.pop("add_generation_prompt"),
        enable_thinking=kwargs.pop("enable_thinking", False),
        drop_thinking=kwargs.pop("drop_thinking", True),
        reasoning_effort=kwargs.pop("reasoning_effort", "low"),
    )
    if kwargs:
        raise TypeError(f"Unsupported DeepSeek V4 reference kwargs: {sorted(kwargs)}")
    return list(tokenizer.encode(text, add_special_tokens=False))


def _harmony_tool_description(tool: Mapping[str, Any]):
    from openai_harmony import ToolDescription

    fn = tool.get("function", tool)
    return ToolDescription.new(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=fn.get("parameters") or {},
    )


def _harmony_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    reasoning_effort: str,
    conversation_start_date: str | None,
):
    from openai_harmony import (
        Author,
        DeveloperContent,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
    )

    effort = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }[reasoning_effort]
    system = (
        SystemContent.new()
        .with_reasoning_effort(effort)
        .with_conversation_start_date(
            conversation_start_date or datetime.now().strftime("%Y-%m-%d")
        )
    )
    out = [Message.from_role_and_content(Role.SYSTEM, system)]
    first_system = next(
        (
            index
            for index, message in enumerate(messages)
            if message["role"] == "system"
        ),
        None,
    )
    if first_system is not None or tools:
        developer = DeveloperContent.new()
        if first_system is not None and messages[first_system].get("content"):
            developer = developer.with_instructions(messages[first_system]["content"])
        if tools:
            developer = developer.with_function_tools(
                [_harmony_tool_description(tool) for tool in tools]
            )
        out.append(Message.from_role_and_content(Role.DEVELOPER, developer))

    for index, message in enumerate(messages):
        if index == first_system:
            continue
        role = message["role"]
        content = message.get("content") or ""
        if role == "user":
            out.append(Message.from_role_and_content(Role.USER, content))
            continue
        if role in {"system", "developer"}:
            developer = DeveloperContent.new().with_instructions(content)
            out.append(Message.from_role_and_content(Role.DEVELOPER, developer))
            continue
        if role == "tool":
            name = message.get("name") or "unknown"
            if not name.startswith("functions."):
                name = f"functions.{name}"
            tool_message = Message.from_author_and_content(
                Author.new(Role.TOOL, name), content
            )
            out.append(
                tool_message.with_recipient("assistant").with_channel("commentary")
            )
            continue
        if role != "assistant":
            raise AssertionError(f"Harmony oracle does not support role={role!r}")

        tool_calls = message.get("tool_calls") or []
        later_final = any(
            later.get("role") == "assistant"
            and not later.get("tool_calls")
            and bool(later.get("content"))
            for later in messages[index + 1 :]
        )
        reasoning = message.get("reasoning_content")
        if reasoning and tool_calls and not later_final:
            out.append(
                Message.from_role_and_content(Role.ASSISTANT, reasoning).with_channel(
                    "analysis"
                )
            )
        if content:
            out.append(
                Message.from_role_and_content(Role.ASSISTANT, content).with_channel(
                    "final"
                )
            )
        for tool_call in tool_calls:
            fn = tool_call.get("function", tool_call)
            arguments = fn.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            name = fn.get("name", "")
            if not name.startswith("functions."):
                name = f"functions.{name}"
            out.append(
                Message.from_role_and_content(Role.ASSISTANT, arguments)
                .with_channel("commentary")
                .with_recipient(name)
            )
        if not content and not tool_calls and not reasoning:
            out.append(
                Message.from_role_and_content(Role.ASSISTANT, "").with_channel("final")
            )
    return out


def _render_harmony(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    del tokenizer
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Role,
        load_harmony_encoding,
    )

    kwargs = dict(kwargs)
    add_generation_prompt = kwargs.pop("add_generation_prompt")
    tools = kwargs.pop("tools", None)
    reasoning_effort = kwargs.pop("reasoning_effort", "medium")
    conversation_start_date = kwargs.pop("conversation_start_date", None)
    if kwargs:
        raise TypeError(f"Unsupported Harmony reference kwargs: {sorted(kwargs)}")

    encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    conversation = Conversation.from_messages(
        _harmony_messages(
            messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
            conversation_start_date=conversation_start_date,
        )
    )
    if add_generation_prompt:
        prompt = encoder.render_conversation_for_completion(
            conversation, next_turn_role=Role.ASSISTANT
        )
        return prompt + encoder.encode(
            "<|channel|>analysis<|message|>", allowed_special="all"
        )
    return encoder.render_conversation_for_training(conversation)


REFERENCE_ORACLES: Mapping[str, ReferenceOracle] = MappingProxyType(
    {
        "hugging-face": ReferenceOracle("hugging-face", _render_hugging_face),
        "deepseek-v4": ReferenceOracle("deepseek-v4", _render_deepseek_v4),
        "harmony": ReferenceOracle("harmony", _render_harmony),
    }
)
"""All available reference implementations, keyed by stable oracle name."""

DEFAULT_REFERENCE_ORACLE = "hugging-face"

RENDERER_ORACLE_ROUTES: Mapping[str, str] = MappingProxyType(
    {
        "deepseek-v4": "deepseek-v4",
        "gpt-oss": "harmony",
    }
)
"""Renderer families whose canonical reference is not a Jinja template."""


def reference_oracle_for_renderer(renderer_name: str) -> str:
    """Return the oracle registered for a resolved renderer name."""
    return RENDERER_ORACLE_ROUTES.get(renderer_name, DEFAULT_REFERENCE_ORACLE)


def reference_oracle_for_model(model_name: str) -> str:
    """Resolve a canonical model through the production renderer routing map."""
    renderer_name = MODEL_RENDERER_MAP.get(model_name, "default")
    return reference_oracle_for_renderer(renderer_name)


def render_reference(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    oracle: str | None = None,
    **kwargs: Any,
) -> list[int]:
    """Render ``messages`` through one resolved reference-oracle adapter."""
    oracle_name = oracle or reference_oracle_for_model(
        getattr(tokenizer, "name_or_path", "")
    )
    try:
        adapter = REFERENCE_ORACLES[oracle_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown reference oracle {oracle_name!r}; "
            f"registered oracles: {sorted(REFERENCE_ORACLES)}"
        ) from error

    kwargs.setdefault("add_generation_prompt", False)
    return adapter.render(tokenizer, messages, **kwargs)


__all__ = [
    "DEFAULT_REFERENCE_ORACLE",
    "DEEPSEEK_V4_MODEL",
    "REFERENCE_ORACLES",
    "RENDERER_ORACLE_ROUTES",
    "ReferenceOracle",
    "reference_oracle_for_model",
    "reference_oracle_for_renderer",
    "render_reference",
]
