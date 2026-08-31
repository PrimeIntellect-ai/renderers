"""Shared catalog and scenario corpus for renderer reference-parity tests."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from itertools import product
from typing import Any, Iterable, Mapping

from renderers.base import MODEL_RENDERER_MAP
from renderers.configs import _config_class_for
from tests.reference_rendering import reference_oracle_for_renderer


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    }
]


@dataclass(frozen=True)
class ModelCase:
    model: str
    renderer: str = "auto"
    suites: frozenset[str] = frozenset({"parity"})
    excluded_scenarios: frozenset[str] = frozenset()
    oracle_defaults: tuple[tuple[str, Any], ...] = ()

    @property
    def resolved_renderer(self) -> str:
        if self.renderer != "auto":
            return self.renderer
        return MODEL_RENDERER_MAP.get(self.model, "default")

    @property
    def oracle(self) -> str:
        return reference_oracle_for_renderer(self.resolved_renderer)


def _model(
    model: str,
    *,
    renderer: str = "auto",
    shared: bool = True,
    roundtrip: bool = True,
    bridge: bool = False,
    excluded: Iterable[str] = (),
    oracle_defaults: Mapping[str, Any] | None = None,
    extra_suites: Iterable[str] = (),
) -> ModelCase:
    suites = {"parity"}
    if shared:
        suites.update({"shared", "plain-parser", "build-helpers"})
    if roundtrip:
        suites.add("roundtrip")
    if bridge:
        suites.add("bridge")
    resolved_renderer = (
        MODEL_RENDERER_MAP.get(model, "default") if renderer == "auto" else renderer
    )
    if reference_oracle_for_renderer(resolved_renderer) == "harmony":
        suites.discard("plain-parser")
        suites.discard("build-helpers")
    if model.startswith("meta-llama/"):
        suites.discard("build-helpers")
    suites.update(extra_suites)
    return ModelCase(
        model=model,
        renderer=renderer,
        suites=frozenset(suites),
        excluded_scenarios=frozenset(excluded),
        oracle_defaults=tuple((oracle_defaults or {}).items()),
    )


# This is the only test-side model catalog. Suites select views of it instead
# of maintaining their own nearly-identical model lists.
MODEL_CATALOG = (
    _model(
        "Qwen/Qwen3-8B",
        bridge=True,
        extra_suites={"tool-arg-types", "disabled-thinking"},
    ),
    _model(
        "PrimeIntellect/Qwen3-0.6B",
        bridge=True,
        extra_suites={"tool-arg-types"},
    ),
    _model("PrimeIntellect/Qwen3-1.7B", shared=False, roundtrip=False),
    _model("Qwen/Qwen3.5-0.8B", shared=False, roundtrip=False),
    _model("Qwen/Qwen3.5-2B", shared=False, roundtrip=False),
    _model("Qwen/Qwen3.5-4B", shared=False, roundtrip=False),
    _model(
        "Qwen/Qwen3.5-9B",
        bridge=True,
        extra_suites={"tool-arg-types", "disabled-thinking"},
    ),
    _model("Qwen/Qwen3.5-35B-A3B", shared=False, roundtrip=False),
    _model("Qwen/Qwen3.5-122B-A10B", shared=False, roundtrip=False),
    _model("Qwen/Qwen3.5-397B-A17B", shared=False, roundtrip=False),
    _model(
        "Qwen/Qwen3.6-35B-A3B",
        bridge=True,
        extra_suites={"disabled-thinking"},
    ),
    _model(
        "Qwen/Qwen3.8-27B",
        bridge=True,
        extra_suites={"disabled-thinking"},
    ),
    _model(
        "Qwen/Qwen3.8-Flash-Next",
        bridge=True,
        extra_suites={"disabled-thinking"},
    ),
    _model(
        "Qwen/Qwen3-VL-4B-Instruct",
        bridge=False,
        excluded={"tool-call-none"},
    ),
    _model(
        "google/gemma-4-E2B-it",
        shared=False,
        roundtrip=False,
        extra_suites={"gemma-checkpoints"},
    ),
    _model(
        "google/gemma-4-E4B-it",
        shared=False,
        roundtrip=False,
        extra_suites={"gemma-checkpoints"},
    ),
    _model(
        "google/gemma-4-26B-A4B-it",
        shared=False,
        roundtrip=False,
        extra_suites={"gemma-checkpoints", "disabled-thinking"},
    ),
    _model(
        "google/gemma-4-31B-it",
        bridge=True,
        extra_suites={"gemma-checkpoints"},
    ),
    _model(
        "zai-org/GLM-5",
        bridge=True,
        extra_suites={"tool-arg-types", "glm-tool-names"},
    ),
    _model("zai-org/GLM-5.1", bridge=True),
    _model("zai-org/GLM-4.7-Flash"),
    _model(
        "THUDM/GLM-4.5-Air",
        bridge=True,
        extra_suites={"glm-tool-names"},
    ),
    _model(
        "MiniMaxAI/MiniMax-M2.5",
        bridge=True,
        extra_suites={"tool-arg-types"},
    ),
    _model(
        "moonshotai/Kimi-K2-Instruct",
        bridge=True,
        extra_suites={"tool-arg-types"},
    ),
    _model(
        "moonshotai/Kimi-K2.5",
        bridge=True,
        excluded={"inline-thinking-history"},
    ),
    _model(
        "moonshotai/Kimi-K2.6",
        bridge=False,
        excluded={"inline-thinking-history"},
    ),
    _model(
        "deepseek-ai/DeepSeek-V3",
        shared=False,
        roundtrip=False,
        excluded={
            "tool-call-content",
            "tool-call-none",
            "multiple-tool-calls",
            "tool-response",
            "named-tool-response",
            "consecutive-tool-responses",
            "full-tool-cycle",
            "multi-step-tool-cycle",
        },
    ),
    _model(
        "deepseek-ai/DeepSeek-R1",
        shared=False,
        roundtrip=False,
        excluded={
            "tool-call-content",
            "tool-call-none",
            "multiple-tool-calls",
            "tool-response",
            "named-tool-response",
            "consecutive-tool-responses",
            "full-tool-cycle",
            "multi-step-tool-cycle",
        },
    ),
    # DeepSeek V4 ships a Python encoder instead of a Jinja chat template.
    # tests/reference_rendering.py supplies its independent reference oracle.
    _model("deepseek-ai/DeepSeek-V4-Flash-0731", bridge=True),
    _model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        bridge=True,
        extra_suites={"tool-arg-types"},
    ),
    _model("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
    _model("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"),
    _model("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", bridge=True),
    _model(
        "poolside/Laguna-XS.2",
        bridge=False,
        extra_suites={"tool-arg-types"},
    ),
    _model("poolside/Laguna-M.1", bridge=False),
    _model("poolside/Laguna-XS-2.1", roundtrip=False, bridge=False),
    _model("poolside/Laguna-S-2.1", roundtrip=False, bridge=False),
    _model(
        "meta-llama/Llama-3.2-1B-Instruct",
        bridge=True,
        excluded={"multiple-tool-calls", "consecutive-tool-responses"},
        oracle_defaults={"date_string": "26 Jul 2024"},
        extra_suites={"llama-checkpoints"},
    ),
    _model(
        "meta-llama/Llama-3.2-3B-Instruct",
        shared=False,
        roundtrip=False,
        excluded={"multiple-tool-calls", "consecutive-tool-responses"},
        oracle_defaults={"date_string": "26 Jul 2024"},
        extra_suites={"llama-checkpoints"},
    ),
    _model(
        "openai/gpt-oss-20b",
        renderer="gpt-oss",
        bridge=True,
        excluded={"multi-step-tool-cycle"},
    ),
    _model("tencent/Hy3", bridge=True),
    _model("thinkingmachines/Inkling", bridge=True),
    _model("thinkingmachines/Inkling-Small", bridge=True),
    _model(
        "Qwen/Qwen2.5-0.5B-Instruct",
        renderer="default",
        roundtrip=True,
        bridge=False,
    ),
)


def models_for(suite: str) -> tuple[ModelCase, ...]:
    return tuple(case for case in MODEL_CATALOG if suite in case.suites)


@dataclass(frozen=True)
class Scenario:
    id: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    add_generation_prompt: bool = False
    only_renderers: frozenset[str] = dataclass_field(default_factory=frozenset)

    @property
    def render_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"add_generation_prompt": self.add_generation_prompt}
        if self.tools:
            kwargs["tools"] = list(self.tools)
        return kwargs


def _scenario(
    id: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    only_renderers: Iterable[str] = (),
) -> Scenario:
    return Scenario(
        id=id,
        messages=tuple(messages),
        tools=tuple(tools or ()),
        add_generation_prompt=add_generation_prompt,
        only_renderers=frozenset(only_renderers),
    )


def _call(city: str, *, id: str | None = None) -> dict:
    call: dict[str, Any] = {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": city}},
    }
    if id is not None:
        call["id"] = id
    return call


# Scenarios are deliberately renderer-agnostic. Renderer/template limitations
# live in MODEL_CATALOG, so the corpus itself stays reusable and auditable.
SCENARIOS = (
    _scenario("user", [{"role": "user", "content": "Hello!"}]),
    _scenario(
        "system-user",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ],
    ),
    _scenario(
        "terminal-assistant",
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    ),
    _scenario(
        "multi-turn",
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ],
    ),
    _scenario(
        "empty-assistant",
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
        ],
    ),
    _scenario(
        "reasoning",
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Simple arithmetic.",
                "content": "4",
            },
        ],
    ),
    _scenario(
        "historical-reasoning",
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Adding small ints.",
                "content": "4",
            },
            {"role": "user", "content": "Now 3+3?"},
            {
                "role": "assistant",
                "reasoning_content": "Same idea.",
                "content": "6",
            },
        ],
    ),
    _scenario(
        "inline-thinking-history",
        [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "content": "<think>raw reasoning</think>visible answer",
            },
            {"role": "user", "content": "Second"},
        ],
        only_renderers={
            "prime-qwen3",
            "kimi-k2",
            "nemotron-3",
            "nemotron-3-ultra",
            "nemotron-3.5",
            "deepseek-v3",
            "deepseek-r1",
        },
    ),
    _scenario(
        "whitespace",
        [
            {"role": "system", "content": "  system text  \n"},
            {"role": "user", "content": "  user text  \n"},
            {
                "role": "assistant",
                "reasoning_content": "  reasoning  \n",
                "content": "  answer  \n",
            },
        ],
        only_renderers={"nemotron-3", "nemotron-3-ultra", "nemotron-3.5"},
    ),
    _scenario(
        "generation-prompt",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        add_generation_prompt=True,
    ),
    _scenario(
        "generation-prompt-no-system",
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True,
    ),
    _scenario(
        "tools-with-system",
        [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "Weather?"},
        ],
        tools=TOOLS,
    ),
    _scenario(
        "tools-without-system",
        [{"role": "user", "content": "Weather?"}],
        tools=TOOLS,
    ),
    _scenario(
        "tool-call-content",
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [_call("Paris", id="call_1")],
            },
        ],
        tools=TOOLS,
    ),
    _scenario(
        "tool-call-none",
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("Paris", id="call_1")],
            },
        ],
        tools=TOOLS,
    ),
    _scenario(
        "multiple-tool-calls",
        [
            {"role": "user", "content": "Weather in Paris and London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("Paris", id="call_1"),
                    _call("London", id="call_2"),
                ],
            },
        ],
        tools=TOOLS,
    ),
    _scenario(
        "tool-response",
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("Paris", id="call_1")],
            },
            {"role": "tool", "content": '{"temp": 20}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "It is 20 degrees."},
        ],
        tools=TOOLS,
    ),
    _scenario(
        "named-tool-response",
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("Paris", id="call_1")],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temp": 20}',
                "tool_call_id": "call_1",
            },
            {"role": "assistant", "content": "It is 20 degrees."},
        ],
        tools=TOOLS,
    ),
    _scenario(
        "consecutive-tool-responses",
        [
            {"role": "user", "content": "Weather in Paris and London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("Paris", id="call_1"),
                    _call("London", id="call_2"),
                ],
            },
            {"role": "tool", "content": '{"temp": 20}', "tool_call_id": "call_1"},
            {"role": "tool", "content": '{"temp": 15}', "tool_call_id": "call_2"},
            {"role": "assistant", "content": "Paris: 20, London: 15."},
        ],
        tools=TOOLS,
    ),
    _scenario(
        "full-tool-cycle",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "reasoning_content": "I should use the tool.",
                "content": "",
                "tool_calls": [_call("Paris", id="call_1")],
            },
            {"role": "tool", "content": '{"temp": 20}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "It is 20 degrees."},
        ],
        tools=TOOLS,
        add_generation_prompt=True,
    ),
    _scenario(
        "complex-tool-cycle",
        [
            {"role": "user", "content": "Compare Paris and London"},
            {
                "role": "assistant",
                "reasoning_content": "Two cities need two calls.",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "Paris",
                                "options": {"unit": "c", "days": [1, 2]},
                            },
                        },
                    },
                    _call("London"),
                ],
            },
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "tool", "content": '{"temp": 15}'},
            {"role": "assistant", "content": "Paris is warmer."},
        ],
        tools=TOOLS,
        only_renderers={
            "prime-qwen3",
            "nemotron-3",
            "nemotron-3-ultra",
            "nemotron-3.5",
        },
    ),
    _scenario(
        "empty-reasoning-assistant",
        [
            {"role": "user", "content": "Think"},
            {
                "role": "assistant",
                "reasoning_content": "working",
                "content": "",
            },
        ],
        only_renderers={
            "prime-qwen3",
            "nemotron-3",
            "nemotron-3-ultra",
            "nemotron-3.5",
        },
    ),
    _scenario(
        "generation-after-tool-response",
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("Paris")],
            },
            {"role": "tool", "content": '{"temp": 20}'},
        ],
        tools=TOOLS,
        add_generation_prompt=True,
        only_renderers={
            "nemotron-3",
            "nemotron-3-ultra",
            "nemotron-3.5",
            "laguna-m.1",
        },
    ),
    _scenario(
        "multi-step-tool-cycle",
        [
            {"role": "user", "content": "Compare Paris and London"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("Paris", id="call_1")],
            },
            {"role": "tool", "content": '{"temp": 20}', "tool_call_id": "call_1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("London", id="call_2")],
            },
            {"role": "tool", "content": '{"temp": 15}', "tool_call_id": "call_2"},
            {"role": "assistant", "content": "Paris: 20, London: 15."},
        ],
        tools=TOOLS,
    ),
    _scenario(
        "empty-system",
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hi"},
        ],
        add_generation_prompt=True,
    ),
    _scenario(
        "later-system",
        [
            {"role": "user", "content": "A"},
            {"role": "system", "content": "Late instruction"},
            {"role": "user", "content": "B"},
        ],
        only_renderers={"laguna-xs-2.1", "laguna-s-2.1"},
    ),
    _scenario(
        "tool-declare-message",
        [
            {
                "role": "tool_declare",
                "content": "function calc(x: number): number;",
            },
            {"role": "user", "content": "Use the calc tool"},
        ],
        only_renderers={"kimi-k2.5"},
    ),
)


# Representative values for each finite semantic branch. Pydantic validation
# filters the shared union down to the domain accepted by a renderer config.
KWARG_VALUES: Mapping[str, tuple[Any, ...]] = {
    "enable_thinking": (True, False, None),
    "thinking": (True, False),
    "reasoning_effort": (
        "no_think",
        "low",
        "medium",
        "high",
        "xhigh",
        "none",
        "minimal",
        "max",
        0.0,
        0.37,
        0.99,
    ),
    "preserved_thinking": (True, False, None),
    "is_training": (True, False),
    "raw_last_assistant": (True, False),
    "fallback_strategy": ("reasoning_toolcall_retry", None),
    "clear_thinking": (True, False),
    "drop_thinking": (True, False),
    "truncate_history_thinking": (True, False),
    "low_effort": (True, False),
    "medium_effort": (True, False),
    "model_identity": (
        "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.",
        "You are CustomBot, a research assistant.",
    ),
    "render_assistant_messages_raw": (True, False),
    "add_vision_id": (True, False),
    "preserve_thinking": (True, False),
    "conversation_start_date": ("2025-01-15", None),
    "date_string": ("26 Jul 2024", "01 Jan 2030"),
    "tools_in_user_message": (True, False),
}


def kwarg_combinations(case: ModelCase) -> tuple[dict[str, Any], ...]:
    """Cartesian product of every valid explicit reference-control value."""
    config_cls = _config_class_for(case.resolved_renderer)
    fields = sorted(config_cls.template_field_names())
    missing = set(fields) - KWARG_VALUES.keys()
    if missing:
        raise AssertionError(
            f"{case.resolved_renderer} declares uncovered template fields: "
            f"{sorted(missing)}"
        )
    if not fields:
        return ({},)

    combinations = [{}]
    for values in product(*(KWARG_VALUES[field] for field in fields)):
        kwargs = dict(zip(fields, values))
        try:
            config_cls(**kwargs)
        except Exception:
            continue
        combinations.append(kwargs)
    return tuple(combinations)


def scenario_is_valid(
    case: ModelCase, scenario: Scenario, kwargs: Mapping[str, Any]
) -> bool:
    if scenario.id in case.excluded_scenarios:
        return False
    if (
        scenario.only_renderers
        and case.resolved_renderer not in scenario.only_renderers
    ):
        return False

    # These renderers intentionally retain a generated empty thinking wrapper
    # on plain historical turns when thinking is disabled. That stability
    # contract is tested separately; it is not upstream-reference parity.
    thinking_disabled = kwargs.get("enable_thinking") is False or (
        kwargs.get("enable_thinking") is None
        and case.model in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"}
    )
    if (
        case.resolved_renderer in {"qwen3", "qwen3.5", "qwen3.6", "qwen3.8"}
        and thinking_disabled
        and kwargs.get("preserve_thinking") is not True
        and scenario.id
        in (
            {
                "multi-turn",
                "tool-response",
                "named-tool-response",
                "consecutive-tool-responses",
                "full-tool-cycle",
                "multi-step-tool-cycle",
            }
            if case.resolved_renderer == "qwen3"
            else {"multi-turn"}
        )
    ):
        return False

    # Gemma 4's 26B/31B template revision strips the empty disabled-thinking
    # generation prefill from assistant history. The renderer deliberately
    # retains it so sampled streams remain byte-prefix-stable across rerenders.
    # That behavior is covered by the stability suite, not reference parity.
    if (
        case.model in {"google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it"}
        and kwargs.get("enable_thinking", False) is False
        and any(
            message.get("role") == "assistant"
            and not (message.get("reasoning") or message.get("reasoning_content"))
            for message in scenario.messages
        )
    ):
        return False
    return True
