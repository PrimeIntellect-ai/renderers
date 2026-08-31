"""Unified renderer parity matrix.

Every valid cell is the product of one model, one shared conversation
scenario, and all explicit reference-controlled values accepted by that
model's typed config. Unsupported cells are excluded declaratively in the
model catalog rather than discovered at runtime through skips or xfails. The
reference is model-aware: most models use Hugging Face Jinja, DeepSeek V4 uses
its shipped Python encoder, and GPT-OSS uses Harmony.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from typing import Any, Mapping, cast

import pytest

from parity import (
    KWARG_VALUES,
    MODEL_CATALOG,
    SCENARIOS,
    ModelCase,
    Scenario,
    kwarg_combinations,
    scenario_is_valid,
)
from renderers import create_renderer
from renderers.base import MODEL_RENDERER_MAP, load_tokenizer
from renderers.configs import RendererConfig, _config_class_for
from tests.reference_rendering import render_reference


def _id(case: ModelCase, scenario: Scenario, kwargs: Mapping[str, Any]) -> str:
    values = ",".join(f"{key}={value!r}" for key, value in kwargs.items())
    suffix = values or "defaults"
    return f"{case.model}-{scenario.id}-{suffix}"


def _matrix():
    for case in MODEL_CATALOG:
        for kwargs in kwarg_combinations(case):
            for scenario in SCENARIOS:
                if scenario_is_valid(case, scenario, kwargs):
                    yield pytest.param(
                        case,
                        scenario,
                        kwargs,
                        id=_id(case, scenario, kwargs),
                    )


@lru_cache(maxsize=None)
def _tokenizer(model: str):
    return load_tokenizer(model)


@lru_cache(maxsize=None)
def _renderer(model: str, renderer_name: str, items: tuple[tuple[str, Any], ...]):
    tokenizer = _tokenizer(model)
    resolved = (
        MODEL_RENDERER_MAP.get(model, "default")
        if renderer_name == "auto"
        else renderer_name
    )
    config = cast(RendererConfig, _config_class_for(resolved)())
    kwargs = dict(items)
    return create_renderer(tokenizer, config, chat_template_kwargs=kwargs or None)


def _expected_reference(
    tokenizer,
    scenario: Scenario,
    kwargs: Mapping[str, Any],
) -> list[int]:
    # Config ``None`` values mean "defer to the reference default"; omission
    # expresses the same state for each model-aware oracle.
    explicit_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return render_reference(
        tokenizer,
        [dict(message) for message in scenario.messages],
        **explicit_kwargs,
        **scenario.render_kwargs,
    )


def _harmony_tool_description(tool):
    from openai_harmony import ToolDescription

    fn = tool.get("function", tool)
    return ToolDescription.new(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=fn.get("parameters") or {},
    )


def _harmony_messages(scenario: Scenario, kwargs: Mapping[str, Any]):
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
    }[kwargs.get("reasoning_effort", "medium")]
    system = (
        SystemContent.new()
        .with_reasoning_effort(effort)
        .with_conversation_start_date(
            kwargs.get("conversation_start_date") or datetime.now().strftime("%Y-%m-%d")
        )
    )
    out = [Message.from_role_and_content(Role.SYSTEM, system)]
    messages = list(scenario.messages)
    first_system = next(
        (
            index
            for index, message in enumerate(messages)
            if message["role"] == "system"
        ),
        None,
    )
    if first_system is not None or scenario.tools:
        developer = DeveloperContent.new()
        if first_system is not None and messages[first_system].get("content"):
            developer = developer.with_instructions(messages[first_system]["content"])
        if scenario.tools:
            developer = developer.with_function_tools(
                [_harmony_tool_description(tool) for tool in scenario.tools]
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


def _expected_harmony(scenario: Scenario, kwargs: Mapping[str, Any]) -> list[int]:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Role,
        load_harmony_encoding,
    )

    encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    conversation = Conversation.from_messages(_harmony_messages(scenario, kwargs))
    if scenario.add_generation_prompt:
        prompt = encoder.render_conversation_for_completion(
            conversation, next_turn_role=Role.ASSISTANT
        )
        return prompt + encoder.encode(
            "<|channel|>analysis<|message|>", allowed_special="all"
        )
    return encoder.render_conversation_for_training(conversation)


def test_catalog_covers_every_declared_kwarg():
    declared = {
        field
        for case in MODEL_CATALOG
        for field in _config_class_for(case.resolved_renderer).template_field_names()
    }
    assert declared <= KWARG_VALUES.keys()


def test_catalog_routes_every_auto_model_to_its_declared_renderer():
    for case in MODEL_CATALOG:
        if case.renderer == "auto":
            assert case.model in MODEL_RENDERER_MAP


@pytest.mark.parametrize("case,scenario,kwargs", tuple(_matrix()))
def test_renderer_matches_reference(
    case: ModelCase,
    scenario: Scenario,
    kwargs: Mapping[str, Any],
):
    tokenizer = _tokenizer(case.model)
    renderer = _renderer(case.model, case.renderer, tuple(kwargs.items()))
    for key, value in kwargs.items():
        if value is not None:
            assert getattr(renderer.config, key) == value

    if case.oracle == "harmony":
        expected = _expected_harmony(scenario, kwargs)
    elif case.oracle == "reference":
        oracle_kwargs = dict(case.oracle_defaults)
        oracle_kwargs.update(kwargs)
        expected = _expected_reference(tokenizer, scenario, oracle_kwargs)
    else:
        raise AssertionError(f"Unknown reference oracle: {case.oracle!r}")
    got = renderer.render_ids(list(scenario.messages), **scenario.render_kwargs)
    assert got == expected, (
        f"{case.model} / {scenario.id} / {dict(kwargs)!r}: renderer diverged "
        f"from {case.oracle} oracle (got {len(got)} tokens, expected "
        f"{len(expected)})"
    )
