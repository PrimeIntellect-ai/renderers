#!/usr/bin/env python
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "transformers>=4.50.0",
# ]
# ///
"""Compare pure-Python renderer latency with native PyO3 renderer latency.

Run from a checkout after building the native extension:

    uv run maturin develop --manifest-path crates/renderers-py/Cargo.toml --release
    uv run python benchmarks/native_vs_python_qwen3.py --families all

The benchmark intentionally uses the public Python APIs on both sides. Native
timings include PyO3 boundary and Python object conversion costs, which is the
relevant number for Python callers. Use the Criterion bench for pure Rust
hot-path timings.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from renderers import _native_router as router
from renderers.base import Message, ToolSpec, load_tokenizer


TOOLS = cast(
    list[ToolSpec],
    [
        {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
        {
            "name": "search_places",
            "description": "Find places matching a set of constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "kid_friendly": {"type": "boolean"},
                            "max_walk_minutes": {"type": "integer"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "required": ["city", "query"],
            },
        },
        {
            "name": "book_table",
            "description": "Create a restaurant booking request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant": {"type": "string"},
                    "party_size": {"type": "integer"},
                    "time": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["restaurant", "party_size", "time"],
            },
        },
    ],
)


@dataclass(frozen=True)
class FamilySpec:
    family: str
    model: str


@dataclass(frozen=True)
class RenderScenario:
    name: str
    messages: list[Message]
    tools: list[ToolSpec] | None = None
    add_generation_prompt: bool = False


@dataclass(frozen=True)
class ParseScenario:
    name: str
    prompt: list[Message]
    assistant: Message
    tools: list[ToolSpec] | None = None


@dataclass(frozen=True)
class BridgeScenario:
    name: str
    prompt: list[Message]
    assistant: Message
    new_messages: list[Message]
    tools: list[ToolSpec] | None = None


@dataclass(frozen=True)
class Timing:
    loops: int
    median_ns: float
    min_ns: float
    max_ns: float

    @property
    def median_us(self) -> float:
        return self.median_ns / 1_000.0


@dataclass(frozen=True)
class Memory:
    loops: int
    peak_bytes: int

    @property
    def peak_kib(self) -> float:
        return self.peak_bytes / 1024


@dataclass(frozen=True)
class BenchCase:
    family: str
    model: str
    operation: str
    scenario: str
    token_count: int
    py_fn: Callable[[], object]
    native_fn: Callable[[], object]
    native_np_fn: Callable[[], object] | None


@dataclass(frozen=True)
class BenchRow:
    family: str
    model: str
    operation: str
    scenario: str
    token_count: int
    py_timing: Timing
    native_timing: Timing
    native_np_timing: Timing | None
    py_memory: Memory
    native_memory: Memory
    native_np_memory: Memory | None

    @property
    def list_speedup(self) -> float:
        return self.py_timing.median_ns / self.native_timing.median_ns

    @property
    def np_speedup(self) -> float | None:
        if self.native_np_timing is None:
            return None
        return self.py_timing.median_ns / self.native_np_timing.median_ns


DEFAULT_FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("qwen3", "Qwen/Qwen3-8B"),
    FamilySpec("qwen35", "Qwen/Qwen3.5-9B"),
    FamilySpec("qwen36", "Qwen/Qwen3.6-35B-A3B"),
    FamilySpec("glm5", "zai-org/GLM-5"),
    FamilySpec("glm51", "zai-org/GLM-5.1"),
    FamilySpec("glm45", "THUDM/GLM-4.5-Air"),
    FamilySpec("deepseek_v3", "deepseek-ai/DeepSeek-V3"),
    FamilySpec("kimi_k2", "moonshotai/Kimi-K2-Instruct"),
    FamilySpec("minimax_m2", "MiniMaxAI/MiniMax-M2.5"),
    FamilySpec("nemotron3", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
)


FAMILY_BY_NAME = {spec.family: spec for spec in DEFAULT_FAMILIES}


def _medium_messages() -> list[Message]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that calls tools when needed.",
        },
        {
            "role": "user",
            "content": "Plan a weekend trip to Lisbon for two; we like food and walking.",
        },
        {
            "role": "assistant",
            "content": (
                "I'll help. First, let me check the weather and find some restaurants."
            ),
        },
        {"role": "user", "content": "Sounds good - go ahead."},
        {
            "role": "assistant",
            "content": (
                "Here's a plan: Friday evening tapas at Time Out Market, Saturday "
                "morning walk through Alfama, Saturday lunch at Ramiro, Saturday "
                "afternoon Belem pasteis, Sunday morning Sao Jorge castle, Sunday "
                "lunch at Cervejaria Trindade."
            ),
        },
    ]


def _long_history(rounds: int = 18) -> list[Message]:
    messages: list[Message] = [
        {
            "role": "system",
            "content": (
                "You are an itinerary planner. Preserve constraints, cite tradeoffs, "
                "and keep tool observations separate from recommendations."
            ),
        }
    ]
    for idx in range(rounds):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Leg {idx}: compare museum, food, and walking options. "
                    f"We have budget band {idx % 4}, transit pass {idx % 3}, "
                    "and one traveler who avoids late dinners."
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"For leg {idx}, start with a walkable cluster, keep the meal "
                    "close to transit, and leave a fallback indoor option. "
                    "The strongest tradeoff is time certainty versus variety."
                ),
            }
        )
    messages.append(
        {"role": "user", "content": "Now produce the final plan with the best swaps."}
    )
    return messages


def _reasoning_history(rounds: int = 10) -> list[Message]:
    messages: list[Message] = [
        {"role": "system", "content": "You are concise but keep prior reasoning."}
    ]
    for idx in range(rounds):
        messages.append({"role": "user", "content": f"Score option {idx}."})
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": (
                    f"Option {idx} has a distance score of {idx % 5}, a food score "
                    f"of {(idx + 2) % 5}, and a weather risk score of {(idx + 3) % 5}."
                ),
                "content": f"Option {idx}: viable with one caveat.",
            }
        )
    return messages


def _structured_text_messages() -> list[Message]:
    return [
        {"role": "system", "content": "You preserve structured text parts."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare two plans. "},
                {"type": "text", "text": "Prefer the one with fewer transfers."},
            ],
        },
        {"role": "assistant", "content": "The lower-transfer plan is better."},
    ]


def _tool_cycle_messages() -> list[Message]:
    return [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                },
            ],
        },
        {"role": "tool", "content": "sunny, 22 C"},
        {
            "role": "assistant",
            "content": "It's sunny and 22 C in Paris.",
        },
    ]


def _large_tool_only_messages() -> list[Message]:
    return [
        {"role": "system", "content": "You are a travel operations assistant."},
        {
            "role": "user",
            "content": (
                "Use the available tools to build a food-first morning plan, "
                "but only call tools if missing information blocks the answer."
            ),
        },
    ]


def render_scenarios(family: str) -> list[RenderScenario]:
    scenarios = [
        RenderScenario(
            "medium_gen_prompt", _medium_messages(), add_generation_prompt=True
        ),
        RenderScenario(
            "long_history_gen_prompt",
            _long_history(),
            add_generation_prompt=True,
        ),
        RenderScenario("reasoning_history", _reasoning_history()),
        RenderScenario("tool_cycle_large_schema", _tool_cycle_messages(), tools=TOOLS),
        RenderScenario(
            "large_tools_gen_prompt",
            _large_tool_only_messages(),
            tools=TOOLS,
            add_generation_prompt=True,
        ),
    ]
    if family in {"qwen35", "qwen36"}:
        scenarios.insert(
            3, RenderScenario("structured_text_parts", _structured_text_messages())
        )
    return scenarios


def parse_scenarios() -> list[ParseScenario]:
    prompt: list[Message] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Answer with the needed structure."},
    ]
    return [
        ParseScenario(
            "plain_content",
            prompt,
            {"role": "assistant", "content": "The answer is four."},
        ),
        ParseScenario(
            "reasoning_and_content",
            prompt,
            {
                "role": "assistant",
                "reasoning_content": (
                    "The user asks for arithmetic, so compute two plus two."
                ),
                "content": "The answer is four.",
            },
        ),
        ParseScenario(
            "multi_tool_call",
            prompt,
            {
                "role": "assistant",
                "content": "I will inspect the required details.",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Lisbon","units":"celsius"}',
                        },
                    },
                    {
                        "id": "call_places",
                        "type": "function",
                        "function": {
                            "name": "search_places",
                            "arguments": json.dumps(
                                {
                                    "city": "Lisbon",
                                    "query": "kid friendly Sunday morning",
                                    "filters": {
                                        "kid_friendly": True,
                                        "max_walk_minutes": 20,
                                        "tags": ["parks", "pastries", "views"],
                                    },
                                },
                                separators=(",", ":"),
                            ),
                        },
                    },
                ],
            },
            tools=TOOLS,
        ),
        ParseScenario(
            "long_content",
            prompt,
            {
                "role": "assistant",
                "content": " ".join(
                    f"Recommendation {idx}: keep the plan walkable and reversible."
                    for idx in range(80)
                ),
            },
        ),
    ]


def bridge_scenarios() -> list[BridgeScenario]:
    medium = _medium_messages()
    tool_cycle = _tool_cycle_messages()
    return [
        BridgeScenario(
            "medium_extend_user",
            medium[:-1],
            medium[-1],
            [
                {
                    "role": "user",
                    "content": "Add a kid-friendly option for Sunday morning.",
                }
            ],
        ),
        BridgeScenario(
            "long_history_extend_user",
            _long_history(14)[:-1],
            {
                "role": "assistant",
                "content": (
                    "Here is the compressed plan: keep mornings flexible, cluster "
                    "food stops near transit, and reserve one indoor fallback."
                ),
            },
            [
                {
                    "role": "user",
                    "content": "Add one backup if rain starts before lunch.",
                }
            ],
        ),
        BridgeScenario(
            "tool_response_extension",
            tool_cycle[:-1],
            tool_cycle[-1],
            [
                {
                    "role": "tool",
                    "name": "book_table",
                    "content": '{"status": "waitlist", "eta_minutes": 15}',
                },
                {"role": "user", "content": "Adjust if the restaurant is waitlisted."},
            ],
            tools=TOOLS,
        ),
    ]


def build_python_renderer(family: str, tokenizer: Any) -> Any:
    saved = os.environ.pop("RENDERERS_NATIVE", None)
    try:
        if family == "qwen3":
            from renderers.qwen3 import Qwen3Renderer

            return Qwen3Renderer(tokenizer)
        if family == "qwen35":
            from renderers.qwen35 import Qwen35Renderer

            return Qwen35Renderer(tokenizer)
        if family == "qwen36":
            from renderers.qwen36 import Qwen36Renderer

            return Qwen36Renderer(tokenizer)
        if family == "glm5":
            from renderers.glm5 import GLM5Renderer

            return GLM5Renderer(tokenizer)
        if family == "glm51":
            from renderers.glm5 import GLM51Renderer

            return GLM51Renderer(tokenizer)
        if family == "glm45":
            from renderers.glm45 import GLM45Renderer

            return GLM45Renderer(tokenizer)
        if family == "deepseek_v3":
            from renderers.deepseek_v3 import DeepSeekV3Renderer

            return DeepSeekV3Renderer(tokenizer)
        if family == "kimi_k2":
            from renderers.kimi_k2 import KimiK2Renderer

            return KimiK2Renderer(tokenizer)
        if family == "minimax_m2":
            from renderers.minimax_m2 import MiniMaxM2Renderer

            return MiniMaxM2Renderer(tokenizer)
        if family == "nemotron3":
            from renderers.nemotron3 import Nemotron3Renderer

            return Nemotron3Renderer(tokenizer)
    finally:
        if saved is not None:
            os.environ["RENDERERS_NATIVE"] = saved
    raise ValueError(f"unknown family: {family}")


def build_native_renderer(native_module: Any, family: str, tokenizer_path: str) -> Any:
    factory = {
        "qwen3": native_module.Renderer.qwen3,
        "qwen35": native_module.Renderer.qwen35,
        "qwen36": native_module.Renderer.qwen36,
        "glm5": native_module.Renderer.glm5,
        "glm51": native_module.Renderer.glm51,
        "glm45": native_module.Renderer.glm45,
        "deepseek_v3": native_module.Renderer.deepseek_v3,
        "kimi_k2": native_module.Renderer.kimi_k2,
        "minimax_m2": native_module.Renderer.minimax_m2,
        "nemotron3": native_module.Renderer.nemotron3,
    }.get(family)
    if factory is None:
        raise ValueError(f"unknown family: {family}")
    return factory(tokenizer_path)


def parse_families(raw: str) -> list[FamilySpec]:
    if raw in {"all", "native"}:
        return list(DEFAULT_FAMILIES)
    selected: list[FamilySpec] = []
    for item in raw.split(","):
        family = item.strip()
        if not family:
            continue
        try:
            selected.append(FAMILY_BY_NAME[family])
        except KeyError as exc:
            known = ", ".join(sorted(FAMILY_BY_NAME))
            raise SystemExit(f"unknown family {family!r}; known: {known}") from exc
    if not selected:
        raise SystemExit("--families resolved to an empty set")
    return selected


def apply_model_overrides(
    specs: Sequence[FamilySpec], overrides: Sequence[str]
) -> list[FamilySpec]:
    by_family = {spec.family: spec for spec in specs}
    for override in overrides:
        if "=" not in override:
            if len(specs) != 1:
                raise SystemExit(
                    "--model without FAMILY=MODEL is only valid with one family"
                )
            family, model = specs[0].family, override
        else:
            family, model = override.split("=", 1)
        family = family.strip()
        model = model.strip()
        if family not in by_family:
            raise SystemExit(
                f"--model override references unselected family {family!r}"
            )
        by_family[family] = FamilySpec(family, model)
    return [by_family[spec.family] for spec in specs]


def time_case(
    fn: Callable[[], object],
    *,
    min_time_s: float,
    repeats: int,
) -> Timing:
    loops = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        elapsed_s = (time.perf_counter_ns() - start) / 1_000_000_000
        if elapsed_s >= min_time_s:
            break
        loops *= 2

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter_ns() - start) / loops)

    return Timing(
        loops=loops,
        median_ns=statistics.median(samples),
        min_ns=min(samples),
        max_ns=max(samples),
    )


def memory_case(fn: Callable[[], object], *, loops: int) -> Memory:
    gc.collect()
    tracemalloc.start()
    try:
        for _ in range(loops):
            fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Memory(loops=loops, peak_bytes=peak)


def _as_ids(value: Any) -> list[int]:
    if hasattr(value, "token_ids"):
        return list(value.token_ids)
    return list(value)


def _assert_parsed_equal(py_value: Any, native_value: Any) -> None:
    if py_value.content != native_value.content:
        raise AssertionError("parse_response content parity failed before benchmarking")
    if (py_value.reasoning_content or None) != (native_value.reasoning_content or None):
        raise AssertionError(
            "parse_response reasoning parity failed before benchmarking"
        )
    if len(py_value.tool_calls) != len(native_value.tool_calls):
        raise AssertionError("parse_response tool-call count parity failed")
    for py_call, native_call in zip(
        py_value.tool_calls, native_value.tool_calls, strict=True
    ):
        if (
            py_call.raw,
            py_call.name,
            py_call.arguments,
            py_call.status,
        ) != (
            native_call.raw,
            native_call.name,
            native_call.arguments,
            native_call.status,
        ):
            raise AssertionError("parse_response tool-call parity failed")


def _completion_ids(renderer: Any, scenario: ParseScenario) -> list[int]:
    prompt_ids = renderer.render_ids(
        scenario.prompt,
        tools=scenario.tools,
        add_generation_prompt=True,
    )
    full_ids = renderer.render_ids(
        scenario.prompt + [scenario.assistant],
        tools=scenario.tools,
    )
    completion = list(full_ids)[len(prompt_ids) :]
    if not completion:
        raise AssertionError(f"{scenario.name} produced an empty completion")
    return completion


def _bridge_inputs(
    renderer: Any, scenario: BridgeScenario
) -> tuple[list[int], list[int]]:
    previous_prompt_ids = renderer.render_ids(
        scenario.prompt,
        tools=scenario.tools,
        add_generation_prompt=True,
    )
    full_ids = renderer.render_ids(
        scenario.prompt + [scenario.assistant],
        tools=scenario.tools,
    )
    previous_completion_ids = list(full_ids)[len(previous_prompt_ids) :]
    if not previous_completion_ids:
        raise AssertionError(f"{scenario.name} produced an empty completion")
    return list(previous_prompt_ids), previous_completion_ids


def _add_render_cases(
    cases: list[BenchCase],
    skipped: list[str],
    *,
    spec: FamilySpec,
    py_renderer: Any,
    native_renderer: Any,
    strict: bool,
) -> None:
    for scenario in render_scenarios(spec.family):
        try:
            py_ids = _as_ids(
                py_renderer.render_ids(
                    scenario.messages,
                    tools=scenario.tools,
                    add_generation_prompt=scenario.add_generation_prompt,
                )
            )
            native_ids = _as_ids(
                native_renderer.render_ids(
                    scenario.messages,
                    tools=scenario.tools,
                    add_generation_prompt=scenario.add_generation_prompt,
                )
            )
            if py_ids != native_ids:
                raise AssertionError("render_ids parity failed")
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{spec.family}:render_ids:{scenario.name}: {exc}"
                ) from exc
            skipped.append(f"{spec.family}:render_ids:{scenario.name}: {exc}")
            continue
        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "render_ids",
                scenario.name,
                len(py_ids),
                lambda scenario=scenario: py_renderer.render_ids(
                    scenario.messages,
                    tools=scenario.tools,
                    add_generation_prompt=scenario.add_generation_prompt,
                ),
                lambda scenario=scenario: native_renderer.render_ids(
                    scenario.messages,
                    tools=scenario.tools,
                    add_generation_prompt=scenario.add_generation_prompt,
                ),
                lambda scenario=scenario: native_renderer.render_ids_np(
                    scenario.messages,
                    tools=scenario.tools,
                    add_generation_prompt=scenario.add_generation_prompt,
                ),
            )
        )


def _add_parse_cases(
    cases: list[BenchCase],
    skipped: list[str],
    *,
    spec: FamilySpec,
    py_renderer: Any,
    native_renderer: Any,
    strict: bool,
) -> None:
    for scenario in parse_scenarios():
        try:
            py_completion_ids = _completion_ids(py_renderer, scenario)
            native_completion_ids = _completion_ids(native_renderer, scenario)
            native_prompt_np = native_renderer.render_ids_np(
                scenario.prompt,
                tools=scenario.tools,
                add_generation_prompt=True,
            )
            native_full_np = native_renderer.render_ids_np(
                scenario.prompt + [scenario.assistant],
                tools=scenario.tools,
            )
            native_completion_np = native_full_np[len(native_prompt_np) :]
            if py_completion_ids != native_completion_ids:
                raise AssertionError("completion parity failed")
            _assert_parsed_equal(
                py_renderer.parse_response(py_completion_ids),
                native_renderer.parse_response(py_completion_ids),
            )
            _assert_parsed_equal(
                py_renderer.parse_response(py_completion_ids),
                native_renderer.parse_response_np(native_completion_np),
            )
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{spec.family}:parse_response:{scenario.name}: {exc}"
                ) from exc
            skipped.append(f"{spec.family}:parse_response:{scenario.name}: {exc}")
            continue
        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "parse_response",
                scenario.name,
                len(py_completion_ids),
                lambda ids=py_completion_ids: py_renderer.parse_response(ids),
                lambda ids=py_completion_ids: native_renderer.parse_response(ids),
                lambda ids=native_completion_np: native_renderer.parse_response_np(ids),
            )
        )


def _add_bridge_cases(
    cases: list[BenchCase],
    skipped: list[str],
    *,
    spec: FamilySpec,
    py_renderer: Any,
    native_renderer: Any,
    strict: bool,
) -> None:
    for scenario in bridge_scenarios():
        try:
            prev_prompt, prev_completion = _bridge_inputs(py_renderer, scenario)
            native_prev_prompt, native_prev_completion = _bridge_inputs(
                native_renderer, scenario
            )
            if (
                prev_prompt != native_prev_prompt
                or prev_completion != native_prev_completion
            ):
                raise AssertionError("bridge input parity failed")

            py_bridge = py_renderer.bridge_to_next_turn(
                prev_prompt,
                prev_completion,
                scenario.new_messages,
                tools=scenario.tools,
            )
            native_bridge = native_renderer.bridge_to_next_turn(
                prev_prompt,
                prev_completion,
                scenario.new_messages,
                tools=scenario.tools,
            )
            if py_bridge is None and native_bridge is None:
                continue
            if py_bridge is None or native_bridge is None:
                raise AssertionError("bridge None parity failed")
            if list(py_bridge.token_ids) != list(native_bridge.token_ids):
                raise AssertionError("bridge parity failed")

            native_prev_prompt_np = native_renderer.render_ids_np(
                scenario.prompt,
                tools=scenario.tools,
                add_generation_prompt=True,
            )
            native_full_np = native_renderer.render_ids_np(
                scenario.prompt + [scenario.assistant],
                tools=scenario.tools,
            )
            native_prev_completion_np = native_full_np[len(native_prev_prompt_np) :]
            native_bridge_np = native_renderer.bridge_to_next_turn_np(
                native_prev_prompt_np,
                native_prev_completion_np,
                scenario.new_messages,
                tools=scenario.tools,
            )
            if native_bridge_np is None:
                raise AssertionError("numpy bridge returned None")
            if list(py_bridge.token_ids) != native_bridge_np.tolist():
                raise AssertionError("numpy bridge parity failed")
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{spec.family}:bridge_to_next_turn:{scenario.name}: {exc}"
                ) from exc
            skipped.append(f"{spec.family}:bridge_to_next_turn:{scenario.name}: {exc}")
            continue

        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "bridge_to_next_turn",
                scenario.name,
                len(py_bridge.token_ids),
                lambda scenario=scenario, pp=prev_prompt, pc=prev_completion: (
                    py_renderer.bridge_to_next_turn(
                        pp,
                        pc,
                        scenario.new_messages,
                        tools=scenario.tools,
                    )
                ),
                lambda scenario=scenario, pp=prev_prompt, pc=prev_completion: (
                    native_renderer.bridge_to_next_turn(
                        pp,
                        pc,
                        scenario.new_messages,
                        tools=scenario.tools,
                    )
                ),
                lambda scenario=scenario, pp=native_prev_prompt_np, pc=native_prev_completion_np: (
                    native_renderer.bridge_to_next_turn_np(
                        pp,
                        pc,
                        scenario.new_messages,
                        tools=scenario.tools,
                    )
                ),
            )
        )


def build_cases(
    *,
    specs: Sequence[FamilySpec],
    native_module: Any,
    strict: bool,
) -> tuple[list[BenchCase], list[str]]:
    cases: list[BenchCase] = []
    skipped: list[str] = []
    for spec in specs:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tokenizer = load_tokenizer(spec.model)
            tokenizer_path = router.resolve_tokenizer_path(tokenizer)
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(tokenizer_path)
            py_renderer = build_python_renderer(spec.family, tokenizer)
            native_renderer = build_native_renderer(
                native_module, spec.family, tokenizer_path
            )
            family_cases: list[BenchCase] = []
            _add_render_cases(
                family_cases,
                skipped,
                spec=spec,
                py_renderer=py_renderer,
                native_renderer=native_renderer,
                strict=strict,
            )
            _add_parse_cases(
                family_cases,
                skipped,
                spec=spec,
                py_renderer=py_renderer,
                native_renderer=native_renderer,
                strict=strict,
            )
            _add_bridge_cases(
                family_cases,
                skipped,
                spec=spec,
                py_renderer=py_renderer,
                native_renderer=native_renderer,
                strict=strict,
            )
            cases.extend(family_cases)
            print(
                f"prepared family={spec.family} model={spec.model} "
                f"tokenizer_path={tokenizer_path}",
                file=sys.stderr,
            )
        except Exception as exc:
            message = f"{spec.family} ({spec.model}): {exc}"
            if strict:
                raise RuntimeError(message) from exc
            skipped.append(message)
            print(f"skipped {message}", file=sys.stderr)
    return cases, skipped


def run_cases(
    cases: Sequence[BenchCase],
    *,
    min_time_s: float,
    repeats: int,
    memory_loops: int,
) -> list[BenchRow]:
    gc.collect()
    gc.disable()
    try:
        rows: list[BenchRow] = []
        for case in cases:
            py_timing = time_case(case.py_fn, min_time_s=min_time_s, repeats=repeats)
            native_timing = time_case(
                case.native_fn, min_time_s=min_time_s, repeats=repeats
            )
            native_np_timing = (
                time_case(case.native_np_fn, min_time_s=min_time_s, repeats=repeats)
                if case.native_np_fn is not None
                else None
            )
            py_memory = memory_case(case.py_fn, loops=memory_loops)
            native_memory = memory_case(case.native_fn, loops=memory_loops)
            native_np_memory = (
                memory_case(case.native_np_fn, loops=memory_loops)
                if case.native_np_fn is not None
                else None
            )
            rows.append(
                BenchRow(
                    case.family,
                    case.model,
                    case.operation,
                    case.scenario,
                    case.token_count,
                    py_timing,
                    native_timing,
                    native_np_timing,
                    py_memory,
                    native_memory,
                    native_np_memory,
                )
            )
    finally:
        gc.enable()
    return rows


def geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


def print_results(
    rows: Sequence[BenchRow], skipped: Sequence[str], memory_loops: int
) -> None:
    print(
        "| family | operation | scenario | tokens | python us | native list us | "
        "native np us | list speedup | np speedup | python peak KiB | "
        "native list peak KiB | native np peak KiB |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        np_us = (
            f"{row.native_np_timing.median_us:.3f}"
            if row.native_np_timing is not None
            else "-"
        )
        np_speedup = f"{row.np_speedup:.2f}x" if row.np_speedup is not None else "-"
        np_peak = (
            f"{row.native_np_memory.peak_kib:.1f}"
            if row.native_np_memory is not None
            else "-"
        )
        print(
            f"| `{row.family}` | `{row.operation}` | `{row.scenario}` | "
            f"{row.token_count} | {row.py_timing.median_us:.3f} | "
            f"{row.native_timing.median_us:.3f} | {np_us} | "
            f"{row.list_speedup:.2f}x | {np_speedup} | "
            f"{row.py_memory.peak_kib:.1f} | {row.native_memory.peak_kib:.1f} | "
            f"{np_peak} |"
        )

    print()
    print("| family | rows | list geomean speedup | np geomean speedup |")
    print("|---|---:|---:|---:|")
    families = sorted({row.family for row in rows})
    for family in families:
        family_rows = [row for row in rows if row.family == family]
        list_speedup = geometric_mean([row.list_speedup for row in family_rows])
        np_speedup = geometric_mean(
            [row.np_speedup for row in family_rows if row.np_speedup is not None]
        )
        print(
            f"| `{family}` | {len(family_rows)} | {list_speedup:.2f}x | "
            f"{np_speedup:.2f}x |"
        )

    print()
    print(
        "memory note: peak KiB uses Python tracemalloc over "
        f"{memory_loops} calls; Rust allocator and NumPy native data buffers "
        "are not included."
    )
    if skipped:
        print()
        print("Skipped cases:")
        for item in skipped:
            print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        default="all",
        help=(
            "Comma-separated family keys or 'all'. Known keys: "
            + ", ".join(sorted(FAMILY_BY_NAME))
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Override model id. Use MODEL when one family is selected, or "
            "FAMILY=MODEL for multi-family runs. May be repeated."
        ),
    )
    parser.add_argument("--min-time", type=float, default=0.35)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--memory-loops",
        type=int,
        default=1000,
        help=(
            "Iterations for tracemalloc peak measurement. This tracks Python "
            "heap allocations, including PyO3 boundary objects, not Rust malloc "
            "or NumPy native data buffers."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping families whose tokenizer is unavailable.",
    )
    args = parser.parse_args()

    os.environ.pop("RENDERERS_NATIVE", None)
    logging.getLogger("transformers_modules").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    native = router.load_native()
    if native is None:
        raise RuntimeError(
            "renderers_native is not built; run `uv run maturin develop "
            "--manifest-path crates/renderers-py/Cargo.toml --release`"
        )

    specs = apply_model_overrides(parse_families(args.families), args.model)
    cases, skipped = build_cases(specs=specs, native_module=native, strict=args.strict)
    if not cases:
        raise RuntimeError("no benchmark cases were prepared")
    rows = run_cases(
        cases,
        min_time_s=args.min_time,
        repeats=args.repeats,
        memory_loops=args.memory_loops,
    )
    print_results(rows, skipped, args.memory_loops)


if __name__ == "__main__":
    main()
