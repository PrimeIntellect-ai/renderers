#!/usr/bin/env python
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "transformers>=4.50.0",
# ]
# ///
"""Compare Qwen3 pure-Python renderer latency with the native PyO3 path.

Run from a checkout after building the native extension:

    uv run --with maturin maturin develop \
      --manifest-path crates/renderers-py/Cargo.toml --release
    uv run python benchmarks/native_vs_python_qwen3.py

The benchmark intentionally uses the public Python APIs on both sides. That
means native timings include PyO3 boundary and Python object conversion costs,
which is the relevant number for Python callers. Use the Criterion bench for
pure Rust hot-path timings.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from renderers import _native_router as router
from renderers.base import Message, ToolSpec, load_tokenizer
from renderers.qwen3 import Qwen3Renderer


MESSAGES: list[Message] = [
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
        "content": "I'll help. First, let me check the weather and find some restaurants.",
    },
    {"role": "user", "content": "Sounds good - go ahead."},
    {
        "role": "assistant",
        "content": (
            "Here's a plan: Friday evening tapas at Time Out Market, Saturday "
            "morning walk through Alfama, Saturday lunch at Ramiro (seafood), "
            "Saturday afternoon Belem pasteis, Sunday morning Sao Jorge castle, "
            "Sunday lunch at Cervejaria Trindade."
        ),
    },
]

NEW_MESSAGES: list[Message] = [
    {"role": "user", "content": "Add a kid-friendly option for Sunday morning."}
]


TOOLS = cast(
    list[ToolSpec],
    [
        {
            "type": "function",
            "function": {
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
        },
        {
            "type": "function",
            "function": {
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
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["city", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
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
        },
    ],
)


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

    @property
    def per_call_bytes(self) -> float:
        return self.peak_bytes / self.loops


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
        {
            "role": "user",
            "content": "Now produce the final plan with the best three swaps.",
        }
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


def _tool_cycle_messages() -> list[Message]:
    return [
        {
            "role": "system",
            "content": "You can call tools and then summarize the result.",
        },
        {"role": "user", "content": "Plan Sunday morning in Lisbon with weather."},
        {
            "role": "assistant",
            "content": "I will check weather and candidate places.",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Lisbon", "units": "celsius"},
                    },
                },
                {
                    "id": "call_places",
                    "type": "function",
                    "function": {
                        "name": "search_places",
                        "arguments": {
                            "city": "Lisbon",
                            "query": "kid friendly Sunday morning",
                            "filters": {
                                "kid_friendly": True,
                                "max_walk_minutes": 20,
                                "tags": ["parks", "pastries", "views"],
                            },
                        },
                    },
                },
            ],
        },
        {"role": "tool", "name": "get_weather", "content": '{"temp": 19, "rain": 0.1}'},
        {
            "role": "tool",
            "name": "search_places",
            "content": json.dumps(
                {
                    "places": [
                        {"name": "Jardim da Estrela", "walk_minutes": 12},
                        {"name": "Manteigaria", "walk_minutes": 18},
                    ]
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "assistant",
            "content": "Use Jardim da Estrela first, then pastries if the weather holds.",
        },
    ]


def _large_tool_only_messages() -> list[Message]:
    return [
        {
            "role": "system",
            "content": "You are a travel operations assistant.",
        },
        {
            "role": "user",
            "content": (
                "Use the available tools to build a food-first morning plan, "
                "but only call tools if missing information blocks the answer."
            ),
        },
    ]


def render_scenarios() -> list[RenderScenario]:
    return [
        RenderScenario("medium_gen_prompt", MESSAGES, add_generation_prompt=True),
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
    return [
        BridgeScenario(
            "medium_extend_user",
            MESSAGES[:-1],
            MESSAGES[-1],
            NEW_MESSAGES,
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
            _tool_cycle_messages()[:-1],
            _tool_cycle_messages()[-1],
            [
                {
                    "role": "tool",
                    "name": "book_table",
                    "content": '{"status": "waitlist", "eta_minutes": 15}',
                },
                {
                    "role": "user",
                    "content": "Adjust if the restaurant is waitlisted.",
                },
            ],
            tools=TOOLS,
        ),
    ]


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
    py_calls = py_value.tool_calls
    native_calls = native_value.tool_calls
    if len(py_calls) != len(native_calls):
        raise AssertionError("parse_response tool-call count parity failed")
    for py_call, native_call in zip(py_calls, native_calls, strict=True):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--min-time", type=float, default=0.35)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--memory-loops",
        type=int,
        default=1000,
        help=(
            "Iterations for tracemalloc peak measurement. This tracks Python "
            "heap allocations, including PyO3 boundary objects, not Rust malloc."
        ),
    )
    args = parser.parse_args()

    os.environ.pop("RENDERERS_NATIVE", None)
    tokenizer = load_tokenizer(args.model)
    tokenizer_path = router.resolve_tokenizer_path(tokenizer)

    native = router.load_native()
    if native is None:
        raise RuntimeError(
            "renderers_native is not built; run `uv run --with maturin maturin "
            "develop --manifest-path crates/renderers-py/Cargo.toml --release`"
        )

    py_renderer = Qwen3Renderer(tokenizer)
    native_renderer = native.Renderer.qwen3(tokenizer_path)

    cases: list[tuple[str, str, int, Callable[[Any], Callable[[], object]]]] = []

    for scenario in render_scenarios():
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
            raise AssertionError(f"{scenario.name} render_ids parity failed")
        cases.append(
            (
                "render_ids",
                scenario.name,
                len(py_ids),
                lambda r, scenario=scenario: (
                    lambda: r.render_ids(
                        scenario.messages,
                        tools=scenario.tools,
                        add_generation_prompt=scenario.add_generation_prompt,
                    )
                ),
            )
        )

    for scenario in parse_scenarios():
        py_completion_ids = _completion_ids(py_renderer, scenario)
        native_completion_ids = _completion_ids(native_renderer, scenario)
        if py_completion_ids != native_completion_ids:
            raise AssertionError(f"{scenario.name} completion parity failed")
        _assert_parsed_equal(
            py_renderer.parse_response(py_completion_ids),
            native_renderer.parse_response(py_completion_ids),
        )
        cases.append(
            (
                "parse_response",
                scenario.name,
                len(py_completion_ids),
                lambda r, ids=py_completion_ids: lambda: r.parse_response(ids),
            )
        )

    for scenario in bridge_scenarios():
        prev_prompt, prev_completion = _bridge_inputs(py_renderer, scenario)
        native_prev_prompt, native_prev_completion = _bridge_inputs(
            native_renderer, scenario
        )
        if (
            prev_prompt != native_prev_prompt
            or prev_completion != native_prev_completion
        ):
            raise AssertionError(f"{scenario.name} bridge input parity failed")
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
        if py_bridge is None or native_bridge is None:
            raise AssertionError(f"{scenario.name} bridge unexpectedly returned None")
        if list(py_bridge.token_ids) != list(native_bridge.token_ids):
            raise AssertionError(f"{scenario.name} bridge parity failed")
        cases.append(
            (
                "bridge_to_next_turn",
                scenario.name,
                len(py_bridge.token_ids),
                lambda r, scenario=scenario, pp=prev_prompt, pc=prev_completion: (
                    lambda: r.bridge_to_next_turn(
                        pp,
                        pc,
                        scenario.new_messages,
                        tools=scenario.tools,
                    )
                ),
            )
        )

    gc.collect()
    gc.disable()
    try:
        rows = []
        for operation, scenario, token_count, make in cases:
            py_timing = time_case(
                make(py_renderer), min_time_s=args.min_time, repeats=args.repeats
            )
            native_timing = time_case(
                make(native_renderer), min_time_s=args.min_time, repeats=args.repeats
            )
            py_memory = memory_case(make(py_renderer), loops=args.memory_loops)
            native_memory = memory_case(make(native_renderer), loops=args.memory_loops)
            rows.append(
                (
                    operation,
                    scenario,
                    token_count,
                    py_timing,
                    native_timing,
                    py_memory,
                    native_memory,
                )
            )
    finally:
        gc.enable()

    print(f"model={args.model}")
    print(f"tokenizer_path={tokenizer_path}")
    print()
    print(
        "| operation | scenario | tokens | python us | native us | speedup | "
        "python peak KiB | native peak KiB |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for (
        operation,
        scenario,
        token_count,
        py_timing,
        native_timing,
        py_memory,
        native_memory,
    ) in rows:
        speedup = py_timing.median_ns / native_timing.median_ns
        print(
            f"| `{operation}` | `{scenario}` | {token_count} | "
            f"{py_timing.median_us:.3f} | "
            f"{native_timing.median_us:.3f} | {speedup:.2f}x | "
            f"{py_memory.peak_kib:.1f} | {native_memory.peak_kib:.1f} |"
        )
    print()
    print(
        "memory note: peak KiB uses Python tracemalloc over "
        f"{args.memory_loops} calls; Rust allocator memory is not included."
    )


if __name__ == "__main__":
    main()
