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
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from renderers import _native_router as router
from renderers.base import load_tokenizer
from renderers.qwen3 import Qwen3Renderer


MESSAGES = [
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

NEW_MESSAGES = [
    {"role": "user", "content": "Add a kid-friendly option for Sunday morning."}
]


@dataclass(frozen=True)
class Timing:
    loops: int
    median_ns: float
    min_ns: float
    max_ns: float

    @property
    def median_us(self) -> float:
        return self.median_ns / 1_000.0


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--min-time", type=float, default=0.35)
    parser.add_argument("--repeats", type=int, default=7)
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

    py_ids = py_renderer.render_ids(MESSAGES, add_generation_prompt=True)
    native_ids = list(native_renderer.render_ids(MESSAGES, add_generation_prompt=True))
    if py_ids != native_ids:
        raise AssertionError("render_ids parity failed before benchmarking")

    prompt_messages = MESSAGES[:-1]
    assistant_message = MESSAGES[-1:]
    prev_prompt = py_renderer.render_ids(prompt_messages, add_generation_prompt=True)
    full = py_renderer.render_ids(prompt_messages + assistant_message)
    prev_completion = full[len(prev_prompt) :]
    if not prev_completion:
        raise AssertionError("benchmark fixture produced an empty completion")

    native_prev_prompt = list(
        native_renderer.render_ids(prompt_messages, add_generation_prompt=True)
    )
    if native_prev_prompt != prev_prompt:
        raise AssertionError("prompt parity failed before benchmarking")

    py_bridge = py_renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, NEW_MESSAGES
    )
    native_bridge = native_renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, NEW_MESSAGES
    )
    if py_bridge is None or native_bridge is None:
        raise AssertionError("bridge fixture unexpectedly returned None")
    if list(py_bridge.token_ids) != list(native_bridge.token_ids):
        raise AssertionError("bridge parity failed before benchmarking")

    parsed = py_renderer.parse_response(py_ids)
    native_parsed = native_renderer.parse_response(py_ids)
    if parsed.content != native_parsed.content:
        raise AssertionError("parse_response parity failed before benchmarking")

    cases: list[tuple[str, Callable[[object], Callable[[], object]]]] = [
        (
            "render_ids",
            lambda r: lambda: r.render_ids(MESSAGES, add_generation_prompt=True),
        ),
        ("parse_response", lambda r: lambda: r.parse_response(py_ids)),
        (
            "bridge_to_next_turn",
            lambda r: lambda: r.bridge_to_next_turn(
                prev_prompt, prev_completion, NEW_MESSAGES
            ),
        ),
    ]

    gc.collect()
    gc.disable()
    try:
        rows = []
        for name, make in cases:
            py_timing = time_case(
                make(py_renderer), min_time_s=args.min_time, repeats=args.repeats
            )
            native_timing = time_case(
                make(native_renderer), min_time_s=args.min_time, repeats=args.repeats
            )
            rows.append((name, py_timing, native_timing))
    finally:
        gc.enable()

    print(f"model={args.model}")
    print(f"tokenizer_path={tokenizer_path}")
    print()
    print("| operation | python us | native us | speedup |")
    print("|---|---:|---:|---:|")
    for name, py_timing, native_timing in rows:
        speedup = py_timing.median_ns / native_timing.median_ns
        print(
            f"| `{name}` | {py_timing.median_us:.3f} | "
            f"{native_timing.median_us:.3f} | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
