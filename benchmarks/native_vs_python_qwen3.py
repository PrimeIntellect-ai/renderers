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
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class BaselineDiff:
    family: str
    operation: str
    scenario: str
    path: str
    current_median_ns: float | None
    baseline_median_ns: float | None
    ratio: float | None

    @property
    def percent_change(self) -> float | None:
        if self.ratio is None:
            return None
        return (self.ratio - 1.0) * 100.0


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


def _batch_messages() -> list[list[Message]]:
    return [
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": f"Write option {idx} in one sentence."},
        ]
        for idx in range(16)
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


def _packed_batch_to_lists(value: Any) -> list[list[int]]:
    ids, offsets = value
    return [
        ids[offsets[idx] : offsets[idx + 1]].tolist() for idx in range(len(offsets) - 1)
    ]


def _sum_token_count(batch: Sequence[Sequence[int]]) -> int:
    return sum(len(ids) for ids in batch)


def _roles_and_contents(
    messages: Sequence[Message],
) -> tuple[list[str], list[str]] | None:
    roles: list[str] = []
    contents: list[str] = []
    for message in messages:
        if message.get("tool_calls") or message.get("reasoning_content"):
            return None
        content = message.get("content", "")
        if not isinstance(content, str):
            return None
        roles.append(str(message["role"]))
        contents.append(content)
    return roles, contents


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


def _session_bridge_to_next_turn(
    session: Any,
    previous_completion_ids: Sequence[int],
    new_messages: Sequence[Message],
) -> Any:
    return session.bridge_to_next_turn(
        previous_completion_ids, new_messages, update=False
    )


def _session_bridge_to_next_turn_np(
    session: Any,
    previous_completion_ids: Any,
    new_messages: Sequence[Message],
) -> Any:
    return session.bridge_to_next_turn_np(
        previous_completion_ids, new_messages, update=False
    )


def _bridge_loop(
    renderer: Any,
    previous_prompt_ids: Sequence[int],
    previous_completion_ids: Sequence[int],
    new_messages: Sequence[Message],
    tools: list[ToolSpec] | None,
    *,
    steps: int,
) -> Any:
    prompt_ids = list(previous_prompt_ids)
    bridged = None
    for _ in range(steps):
        bridged = renderer.bridge_to_next_turn(
            prompt_ids,
            previous_completion_ids,
            new_messages,
            tools=tools,
        )
        if bridged is None:
            raise AssertionError("bridge loop returned None")
        prompt_ids = list(bridged.token_ids)
    return bridged


def _session_bridge_loop(
    session: Any,
    previous_completion_ids: Sequence[int],
    new_messages: Sequence[Message],
    *,
    steps: int,
) -> Any:
    bridged = None
    for _ in range(steps):
        bridged = session.bridge_to_next_turn(
            previous_completion_ids, new_messages, update=True
        )
        if bridged is None:
            raise AssertionError("session bridge loop returned None")
    return bridged


def _session_bridge_loop_np(
    session: Any,
    previous_completion_ids: Any,
    new_messages: Sequence[Message],
    *,
    steps: int,
) -> Any:
    bridged = None
    for _ in range(steps):
        bridged = session.bridge_to_next_turn_np(
            previous_completion_ids, new_messages, update=True
        )
        if bridged is None:
            raise AssertionError("session numpy bridge loop returned None")
    return bridged


def _new_session_bridge_loop(
    renderer: Any,
    prompt: Sequence[Message],
    tools: Any,
    previous_completion_ids: Sequence[int],
    new_messages: Sequence[Message],
    *,
    steps: int,
) -> Any:
    session = renderer.new_session(prompt, tools=tools)
    session.render_ids(add_generation_prompt=True)
    return _session_bridge_loop(
        session,
        previous_completion_ids,
        new_messages,
        steps=steps,
    )


def _new_session_bridge_loop_np(
    renderer: Any,
    prompt: Sequence[Message],
    tools: Any,
    previous_completion_ids: Any,
    new_messages: Sequence[Message],
    *,
    steps: int,
) -> Any:
    session = renderer.new_session(prompt, tools=tools)
    session.render_ids_np(add_generation_prompt=True)
    return _session_bridge_loop_np(
        session,
        previous_completion_ids,
        new_messages,
        steps=steps,
    )


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
        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "render_ids_np_then_tolist",
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
                ).tolist(),
            )
        )
        if scenario.tools:
            prepared_tools = native_renderer.prepare_tools(scenario.tools)
            try:
                native_prepared_ids = _as_ids(
                    native_renderer.render_ids(
                        scenario.messages,
                        tools=prepared_tools,
                        add_generation_prompt=scenario.add_generation_prompt,
                    )
                )
                if py_ids != native_prepared_ids:
                    raise AssertionError("prepared tools render_ids parity failed")
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"{spec.family}:render_ids_prepared_tools:{scenario.name}: {exc}"
                    ) from exc
                skipped.append(
                    f"{spec.family}:render_ids_prepared_tools:{scenario.name}: {exc}"
                )
                continue
            cases.append(
                BenchCase(
                    spec.family,
                    spec.model,
                    "render_ids_prepared_tools",
                    scenario.name,
                    len(py_ids),
                    lambda scenario=scenario: py_renderer.render_ids(
                        scenario.messages,
                        tools=scenario.tools,
                        add_generation_prompt=scenario.add_generation_prompt,
                    ),
                    lambda scenario=scenario, prepared_tools=prepared_tools: (
                        native_renderer.render_ids(
                            scenario.messages,
                            tools=prepared_tools,
                            add_generation_prompt=scenario.add_generation_prompt,
                        )
                    ),
                    lambda scenario=scenario, prepared_tools=prepared_tools: (
                        native_renderer.render_ids_np(
                            scenario.messages,
                            tools=prepared_tools,
                            add_generation_prompt=scenario.add_generation_prompt,
                        )
                    ),
                )
            )
        fast_input = _roles_and_contents(scenario.messages)
        if fast_input is not None and scenario.tools is None:
            roles, contents = fast_input
            try:
                native_fast_ids = _as_ids(
                    native_renderer.render_fast_ids(
                        roles,
                        contents,
                        add_generation_prompt=scenario.add_generation_prompt,
                    )
                )
                if py_ids != native_fast_ids:
                    raise AssertionError("fast input render_ids parity failed")
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"{spec.family}:render_fast_ids:{scenario.name}: {exc}"
                    ) from exc
                skipped.append(f"{spec.family}:render_fast_ids:{scenario.name}: {exc}")
                continue
            cases.append(
                BenchCase(
                    spec.family,
                    spec.model,
                    "render_fast_ids",
                    scenario.name,
                    len(py_ids),
                    lambda scenario=scenario: py_renderer.render_ids(
                        scenario.messages,
                        add_generation_prompt=scenario.add_generation_prompt,
                    ),
                    lambda roles=roles, contents=contents, add_generation_prompt=scenario.add_generation_prompt: (
                        native_renderer.render_fast_ids(
                            roles,
                            contents,
                            add_generation_prompt=add_generation_prompt,
                        )
                    ),
                    lambda roles=roles, contents=contents, add_generation_prompt=scenario.add_generation_prompt: (
                        native_renderer.render_fast_ids_np(
                            roles,
                            contents,
                            add_generation_prompt=add_generation_prompt,
                        )
                    ),
                )
            )
        try:
            prepared_tools = (
                native_renderer.prepare_tools(scenario.tools)
                if scenario.tools is not None
                else None
            )
            session = native_renderer.new_session(
                scenario.messages,
                tools=prepared_tools,
            )
            session_np = native_renderer.new_session(
                scenario.messages,
                tools=prepared_tools,
            )
            session_ids = _as_ids(
                session.render_ids(
                    add_generation_prompt=scenario.add_generation_prompt,
                )
            )
            session_np_ids = session_np.render_ids_np(
                add_generation_prompt=scenario.add_generation_prompt,
            ).tolist()
            if py_ids != session_ids:
                raise AssertionError("session render_ids parity failed")
            if py_ids != session_np_ids:
                raise AssertionError("session numpy render_ids parity failed")
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{spec.family}:session_render_ids:{scenario.name}: {exc}"
                ) from exc
            skipped.append(f"{spec.family}:session_render_ids:{scenario.name}: {exc}")
            continue
        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "session_render_ids",
                scenario.name,
                len(py_ids),
                lambda scenario=scenario: py_renderer.render_ids(
                    scenario.messages,
                    tools=scenario.tools,
                    add_generation_prompt=scenario.add_generation_prompt,
                ),
                lambda session=session, add_generation_prompt=scenario.add_generation_prompt: (
                    session.render_ids(add_generation_prompt=add_generation_prompt)
                ),
                lambda session=session_np, add_generation_prompt=scenario.add_generation_prompt: (
                    session.render_ids_np(add_generation_prompt=add_generation_prompt)
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

        try:
            native_tools = (
                native_renderer.prepare_tools(scenario.tools)
                if scenario.tools is not None
                else None
            )
            session = native_renderer.new_session(scenario.prompt, tools=native_tools)
            session_prompt = list(session.render_ids(add_generation_prompt=True))
            if session_prompt != native_prev_prompt:
                raise AssertionError("session prompt parity failed")
            session_bridge = session.bridge_to_next_turn(
                prev_completion,
                scenario.new_messages,
                update=False,
            )
            if session_bridge is None:
                raise AssertionError("session bridge returned None")
            if list(py_bridge.token_ids) != list(session_bridge.token_ids):
                raise AssertionError("session bridge parity failed")

            session_np = native_renderer.new_session(
                scenario.prompt, tools=native_tools
            )
            session_prompt_np = session_np.render_ids_np(add_generation_prompt=True)
            if session_prompt_np.tolist() != native_prev_prompt:
                raise AssertionError("session numpy prompt parity failed")
            session_bridge_np = session_np.bridge_to_next_turn_np(
                native_prev_completion_np,
                scenario.new_messages,
                update=False,
            )
            if session_bridge_np is None:
                raise AssertionError("session numpy bridge returned None")
            if list(py_bridge.token_ids) != session_bridge_np.tolist():
                raise AssertionError("session numpy bridge parity failed")
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{spec.family}:session_bridge_to_next_turn:{scenario.name}: {exc}"
                ) from exc
            skipped.append(
                f"{spec.family}:session_bridge_to_next_turn:{scenario.name}: {exc}"
            )
            continue

        bench_session = native_renderer.new_session(scenario.prompt, tools=native_tools)
        bench_session.render_ids(add_generation_prompt=True)
        bench_session_np = native_renderer.new_session(
            scenario.prompt, tools=native_tools
        )
        bench_session_np.render_ids_np(add_generation_prompt=True)

        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "session_bridge_to_next_turn",
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
                lambda scenario=scenario, pc=prev_completion, session=bench_session: (
                    _session_bridge_to_next_turn(
                        session,
                        pc,
                        scenario.new_messages,
                    )
                ),
                lambda scenario=scenario, pc=native_prev_completion_np, session=bench_session_np: (
                    _session_bridge_to_next_turn_np(
                        session,
                        pc,
                        scenario.new_messages,
                    )
                ),
            )
        )

        loop_steps = 4
        try:
            py_loop = _bridge_loop(
                py_renderer,
                prev_prompt,
                prev_completion,
                scenario.new_messages,
                scenario.tools,
                steps=loop_steps,
            )
            native_loop = _bridge_loop(
                native_renderer,
                prev_prompt,
                prev_completion,
                scenario.new_messages,
                scenario.tools,
                steps=loop_steps,
            )
            if list(py_loop.token_ids) != list(native_loop.token_ids):
                raise AssertionError("bridge loop parity failed")

            session_loop = _new_session_bridge_loop(
                native_renderer,
                scenario.prompt,
                native_tools,
                prev_completion,
                scenario.new_messages,
                steps=loop_steps,
            )
            if list(py_loop.token_ids) != list(session_loop.token_ids):
                raise AssertionError("session bridge loop parity failed")

            session_loop_np = _new_session_bridge_loop_np(
                native_renderer,
                scenario.prompt,
                native_tools,
                native_prev_completion_np,
                scenario.new_messages,
                steps=loop_steps,
            )
            if list(py_loop.token_ids) != session_loop_np.tolist():
                raise AssertionError("session numpy bridge loop parity failed")
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{spec.family}:session_bridge_loop:{scenario.name}: {exc}"
                ) from exc
            skipped.append(f"{spec.family}:session_bridge_loop:{scenario.name}: {exc}")
            continue

        bench_loop_session = native_renderer.new_session(
            scenario.prompt, tools=native_tools
        )
        bench_loop_session.render_ids(add_generation_prompt=True)
        bench_loop_session_np = native_renderer.new_session(
            scenario.prompt, tools=native_tools
        )
        bench_loop_session_np.render_ids_np(add_generation_prompt=True)

        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "session_bridge_loop",
                f"{scenario.name}_{loop_steps}_steps",
                len(py_loop.token_ids),
                lambda scenario=scenario, pp=prev_prompt, pc=prev_completion: (
                    _bridge_loop(
                        py_renderer,
                        pp,
                        pc,
                        scenario.new_messages,
                        scenario.tools,
                        steps=loop_steps,
                    )
                ),
                lambda scenario=scenario, pc=prev_completion, session=bench_loop_session: (
                    _session_bridge_loop(
                        session.fork(),
                        pc,
                        scenario.new_messages,
                        steps=loop_steps,
                    )
                ),
                lambda scenario=scenario, pc=native_prev_completion_np, session=bench_loop_session_np: (
                    _session_bridge_loop_np(
                        session.fork(),
                        pc,
                        scenario.new_messages,
                        steps=loop_steps,
                    )
                ),
            )
        )


def _add_batch_cases(
    cases: list[BenchCase],
    skipped: list[str],
    *,
    spec: FamilySpec,
    py_renderer: Any,
    native_renderer: Any,
    strict: bool,
) -> None:
    batch = _batch_messages()
    batch_scenarios: list[tuple[str, list[ToolSpec] | None, Any]] = [
        ("short_batch", None, None)
    ]
    try:
        prepared_tools = native_renderer.prepare_tools(TOOLS)
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"{spec.family}:render_batch_ids:prepare_tools: {exc}"
            ) from exc
        skipped.append(f"{spec.family}:render_batch_ids:prepare_tools: {exc}")
        prepared_tools = None
    batch_scenarios.append(("short_batch_prepared_tools", TOOLS, prepared_tools))

    for scenario_name, tools, prepared_tools in batch_scenarios:
        if tools is not None and prepared_tools is None:
            continue
        native_tools = prepared_tools if prepared_tools is not None else None
        try:
            py_batch = [
                list(
                    py_renderer.render_ids(
                        messages, tools=tools, add_generation_prompt=True
                    )
                )
                for messages in batch
            ]
            native_batch = [
                list(ids)
                for ids in native_renderer.render_batch_ids(
                    batch,
                    tools=native_tools,
                    add_generation_prompt=True,
                )
            ]
            native_packed_batch = _packed_batch_to_lists(
                native_renderer.render_batch_ids_np_packed(
                    batch,
                    tools=native_tools,
                    add_generation_prompt=True,
                )
            )
            if py_batch != native_batch:
                raise AssertionError("batch render_ids parity failed")
            if py_batch != native_packed_batch:
                raise AssertionError("packed numpy batch render_ids parity failed")
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{spec.family}:render_batch_ids:{scenario_name}: {exc}"
                ) from exc
            skipped.append(f"{spec.family}:render_batch_ids:{scenario_name}: {exc}")
            continue

        cases.append(
            BenchCase(
                spec.family,
                spec.model,
                "render_batch_ids",
                scenario_name,
                _sum_token_count(py_batch),
                lambda batch=batch, tools=tools: [
                    py_renderer.render_ids(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                    )
                    for messages in batch
                ],
                lambda batch=batch, native_tools=native_tools: (
                    native_renderer.render_batch_ids(
                        batch,
                        tools=native_tools,
                        add_generation_prompt=True,
                    )
                ),
                lambda batch=batch, native_tools=native_tools: (
                    native_renderer.render_batch_ids_np_packed(
                        batch,
                        tools=native_tools,
                        add_generation_prompt=True,
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
            _add_batch_cases(
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
        current_family: str | None = None
        family_started_ns = time.perf_counter_ns()
        total_steps = len(cases) * 4
        step = 0

        def progress(case: BenchCase, label: str) -> None:
            nonlocal step
            step += 1
            print(
                f"[{step}/{total_steps}] {case.family} {case.operation} "
                f"{case.scenario}: {label}",
                file=sys.stderr,
            )

        def finish_family(family: str) -> None:
            family_rows = [row for row in rows if row.family == family]
            if not family_rows:
                return
            elapsed_s = (time.perf_counter_ns() - family_started_ns) / 1_000_000_000
            list_speedup = geometric_mean([row.list_speedup for row in family_rows])
            np_speedup = geometric_mean(
                [row.np_speedup for row in family_rows if row.np_speedup is not None]
            )
            print(
                f"family={family} rows={len(family_rows)} "
                f"list_geomean={list_speedup:.2f}x "
                f"np_geomean={np_speedup:.2f}x elapsed={elapsed_s:.1f}s",
                file=sys.stderr,
            )

        for case in cases:
            if current_family is None:
                current_family = case.family
                family_started_ns = time.perf_counter_ns()
            elif case.family != current_family:
                finish_family(current_family)
                current_family = case.family
                family_started_ns = time.perf_counter_ns()

            progress(case, "python")
            py_timing = time_case(case.py_fn, min_time_s=min_time_s, repeats=repeats)
            progress(case, "native list")
            native_timing = time_case(
                case.native_fn, min_time_s=min_time_s, repeats=repeats
            )
            progress(case, "native np")
            native_np_timing = (
                time_case(case.native_np_fn, min_time_s=min_time_s, repeats=repeats)
                if case.native_np_fn is not None
                else None
            )
            progress(case, "memory")
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
        if current_family is not None:
            finish_family(current_family)
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


def _run_text(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "short_commit": _run_text(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": bool(_run_text(["git", "status", "--porcelain"])),
        "branch": _run_text(["git", "branch", "--show-current"]),
    }


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        return _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    if sys.platform.startswith("linux"):
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        except OSError:
            return None
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _timing_dict(timing: Timing) -> dict[str, Any]:
    return {
        "loops": timing.loops,
        "median_ns": timing.median_ns,
        "median_us": timing.median_us,
        "min_ns": timing.min_ns,
        "max_ns": timing.max_ns,
    }


def _memory_dict(memory: Memory) -> dict[str, Any]:
    return {
        "loops": memory.loops,
        "peak_bytes": memory.peak_bytes,
        "peak_kib": memory.peak_kib,
    }


def _result_rows(rows: Sequence[BenchRow]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        base = {
            "family": row.family,
            "model": row.model,
            "operation": row.operation,
            "scenario": row.scenario,
            "token_count": row.token_count,
        }
        result.append(
            {
                **base,
                "path": "python",
                "timing": _timing_dict(row.py_timing),
                "memory": _memory_dict(row.py_memory),
                "speedup_vs_python": 1.0,
            }
        )
        result.append(
            {
                **base,
                "path": "native_list",
                "timing": _timing_dict(row.native_timing),
                "memory": _memory_dict(row.native_memory),
                "speedup_vs_python": row.list_speedup,
            }
        )
        if row.native_np_timing is not None and row.native_np_memory is not None:
            result.append(
                {
                    **base,
                    "path": "native_np",
                    "timing": _timing_dict(row.native_np_timing),
                    "memory": _memory_dict(row.native_np_memory),
                    "speedup_vs_python": row.np_speedup,
                }
            )
    return result


def _family_summaries(rows: Sequence[BenchRow]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family in sorted({row.family for row in rows}):
        family_rows = [row for row in rows if row.family == family]
        summaries.append(
            {
                "family": family,
                "rows": len(family_rows),
                "list_geomean_speedup": geometric_mean(
                    [row.list_speedup for row in family_rows]
                ),
                "np_geomean_speedup": geometric_mean(
                    [
                        row.np_speedup
                        for row in family_rows
                        if row.np_speedup is not None
                    ]
                ),
            }
        )
    return summaries


def _overall_summary(rows: Sequence[BenchRow]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "list_geomean_speedup": geometric_mean([row.list_speedup for row in rows]),
        "np_geomean_speedup": geometric_mean(
            [row.np_speedup for row in rows if row.np_speedup is not None]
        ),
    }


def build_result_document(
    *,
    rows: Sequence[BenchRow],
    skipped: Sequence[str],
    args: argparse.Namespace,
    native_module: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": {
            "git": _git_metadata(),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
            },
            "rust": {
                "rustc": _run_text(["rustc", "--version"]),
            },
            "platform": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_model": _cpu_model(),
            },
            "native_extension": {
                "module_file": getattr(native_module, "__file__", None),
                "build_mode": "unknown",
            },
        },
        "args": {
            "families": args.families,
            "model": args.model,
            "min_time": args.min_time,
            "repeats": args.repeats,
            "memory_loops": args.memory_loops,
            "strict": args.strict,
        },
        "summary": _overall_summary(rows),
        "families": _family_summaries(rows),
        "rows": _result_rows(rows),
        "skipped": list(skipped),
    }


def write_json(path: str, document: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["family"]),
        str(row["operation"]),
        str(row["scenario"]),
        str(row["path"]),
    )


def compare_to_baseline(
    current: dict[str, Any], baseline: dict[str, Any]
) -> list[BaselineDiff]:
    current_by_key = {_row_key(row): row for row in current.get("rows", [])}
    baseline_by_key = {_row_key(row): row for row in baseline.get("rows", [])}
    diffs: list[BaselineDiff] = []
    for key in sorted(set(current_by_key) | set(baseline_by_key)):
        current_row = current_by_key.get(key)
        baseline_row = baseline_by_key.get(key)
        current_median = (
            current_row["timing"]["median_ns"] if current_row is not None else None
        )
        baseline_median = (
            baseline_row["timing"]["median_ns"] if baseline_row is not None else None
        )
        ratio = (
            current_median / baseline_median
            if current_median is not None and baseline_median is not None
            else None
        )
        family, operation, scenario, path = key
        diffs.append(
            BaselineDiff(
                family=family,
                operation=operation,
                scenario=scenario,
                path=path,
                current_median_ns=current_median,
                baseline_median_ns=baseline_median,
                ratio=ratio,
            )
        )
    return diffs


def _load_baseline(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_us(ns: float | None) -> str:
    if ns is None:
        return "-"
    return f"{ns / 1000.0:.3f}"


def _format_change(diff: BaselineDiff) -> str:
    if diff.percent_change is None:
        return "-"
    return f"{diff.percent_change:+.1f}%"


def _diff_label(diff: BaselineDiff) -> str:
    return f"{diff.family}/{diff.operation}/{diff.scenario}/{diff.path}"


def write_markdown(
    path: str,
    document: dict[str, Any],
    baseline_diffs: Sequence[BaselineDiff],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = document["summary"]
    lines = [
        "# Native Runtime Benchmark",
        "",
        "## Summary",
        "",
        "| rows | list geomean | np geomean | commit | dirty |",
        "|---:|---:|---:|---|---|",
        (
            f"| {summary['rows']} | {summary['list_geomean_speedup']:.2f}x | "
            f"{summary['np_geomean_speedup']:.2f}x | "
            f"`{document['metadata']['git']['short_commit']}` | "
            f"{document['metadata']['git']['dirty']} |"
        ),
        "",
        "## Families",
        "",
        "| family | rows | list geomean | np geomean |",
        "|---|---:|---:|---:|",
    ]
    for item in document["families"]:
        lines.append(
            f"| `{item['family']}` | {item['rows']} | "
            f"{item['list_geomean_speedup']:.2f}x | "
            f"{item['np_geomean_speedup']:.2f}x |"
        )

    if baseline_diffs:
        comparable = [diff for diff in baseline_diffs if diff.ratio is not None]
        regressions = sorted(
            [diff for diff in comparable if diff.ratio and diff.ratio > 1.05],
            key=lambda diff: diff.ratio or 0.0,
            reverse=True,
        )[:10]
        improvements = sorted(
            [diff for diff in comparable if diff.ratio and diff.ratio < 0.95],
            key=lambda diff: diff.ratio or 1.0,
        )[:10]
        missing_current = [
            diff for diff in baseline_diffs if diff.current_median_ns is None
        ]
        new_rows = [diff for diff in baseline_diffs if diff.baseline_median_ns is None]

        lines.extend(
            [
                "",
                "## Worst Regressions",
                "",
                "| case | current us | baseline us | change |",
                "|---|---:|---:|---:|",
            ]
        )
        if regressions:
            for diff in regressions:
                lines.append(
                    f"| `{_diff_label(diff)}` | {_format_us(diff.current_median_ns)} | "
                    f"{_format_us(diff.baseline_median_ns)} | {_format_change(diff)} |"
                )
        else:
            lines.append("| none | - | - | - |")

        lines.extend(
            [
                "",
                "## Best Improvements",
                "",
                "| case | current us | baseline us | change |",
                "|---|---:|---:|---:|",
            ]
        )
        if improvements:
            for diff in improvements:
                lines.append(
                    f"| `{_diff_label(diff)}` | {_format_us(diff.current_median_ns)} | "
                    f"{_format_us(diff.baseline_median_ns)} | {_format_change(diff)} |"
                )
        else:
            lines.append("| none | - | - | - |")

        if missing_current or new_rows:
            lines.extend(["", "## Coverage Changes", ""])
            if missing_current:
                lines.append("Missing current rows:")
                lines.extend(f"- `{_diff_label(diff)}`" for diff in missing_current)
            if new_rows:
                lines.append("New rows:")
                lines.extend(f"- `{_diff_label(diff)}`" for diff in new_rows)

    lines.extend(["", "## Skipped Cases", ""])
    if document["skipped"]:
        lines.extend(f"- {item}" for item in document["skipped"])
    else:
        lines.append("None.")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument(
        "--json-out",
        help="Write structured benchmark results to this JSON file.",
    )
    parser.add_argument(
        "--markdown-out",
        help="Write a Markdown benchmark summary to this file.",
    )
    parser.add_argument(
        "--baseline",
        help="Compare current results against a previous JSON benchmark artifact.",
    )
    parser.add_argument(
        "--fail-on-regression",
        type=float,
        help=(
            "Exit non-zero when a baseline row regresses by more than this "
            "percentage. Missing current baseline rows also fail this gate."
        ),
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
    document = build_result_document(
        rows=rows,
        skipped=skipped,
        args=args,
        native_module=native,
    )
    baseline = _load_baseline(args.baseline)
    baseline_diffs = compare_to_baseline(document, baseline) if baseline else []
    print_results(rows, skipped, args.memory_loops)
    if args.json_out:
        if baseline_diffs:
            document["baseline"] = {
                "path": args.baseline,
                "diffs": [
                    {
                        "family": diff.family,
                        "operation": diff.operation,
                        "scenario": diff.scenario,
                        "path": diff.path,
                        "current_median_ns": diff.current_median_ns,
                        "baseline_median_ns": diff.baseline_median_ns,
                        "ratio": diff.ratio,
                        "percent_change": diff.percent_change,
                    }
                    for diff in baseline_diffs
                ],
            }
        write_json(args.json_out, document)
        print(f"wrote json={args.json_out}", file=sys.stderr)
    if args.markdown_out:
        write_markdown(args.markdown_out, document, baseline_diffs)
        print(f"wrote markdown={args.markdown_out}", file=sys.stderr)

    if args.fail_on_regression is not None and baseline_diffs:
        threshold = args.fail_on_regression / 100.0
        regressions = [
            diff
            for diff in baseline_diffs
            if diff.ratio is not None and diff.ratio > 1.0 + threshold
        ]
        missing_current = [
            diff for diff in baseline_diffs if diff.current_median_ns is None
        ]
        if regressions or missing_current:
            details = ", ".join(
                _diff_label(diff) for diff in [*regressions, *missing_current][:5]
            )
            raise SystemExit(
                f"benchmark regression gate failed: {len(regressions)} "
                f"regressions, {len(missing_current)} missing current rows; {details}"
            )


if __name__ == "__main__":
    main()
