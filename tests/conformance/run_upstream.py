"""Run unmodified upstream tests against the installed, pinned runtime."""

import argparse
import importlib.metadata
import sys
from pathlib import Path

import pytest
import vllm.parser  # noqa: F401 -- fix the runtime path BEFORE adding upstream tests
from export_vllm import VLLM_VERSION, verify_source

options = argparse.ArgumentParser()
options.add_argument("source", type=Path)
options.add_argument(
    "--suite", choices=("engine", "reasoning", "tools"), default="engine"
)
args, extra = options.parse_known_args()
assert importlib.metadata.version("vllm") == VLLM_VERSION
source = args.source.resolve()
verify_source(source)
sys.path.insert(0, str(source))
if args.suite == "engine":
    directory = source / "tests/parser/engine"
    paths = [str(directory)]
else:
    directory = source / (
        "tests/reasoning" if args.suite == "reasoning" else "tests/tool_parsers"
    )
    families = (
        (
            "qwen3",
            "deepseekr1",
            "deepseekv3",
            "minimax_m2",
            "kimi_k2",
            "gptoss",
            "hy_v3",
            "nemotron_v3",
            "gemma4",
            "glm4_moe",
        )
        if args.suite == "reasoning"
        else (
            "qwen3coder",
            "hermes",
            "minimax_m2",
            "glm47_moe",
            "deepseekv3",
            "deepseekv4",
            "kimi_k2",
            "hy_v3",
            "gemma4",
            "poolside_v1",
        )
    )
    suffix = "reasoning_parser" if args.suite == "reasoning" else "tool_parser"
    paths = [str(directory / f"test_{name}_{suffix}.py") for name in families]
raise SystemExit(
    pytest.main(
        [
            *paths,
            f"--confcutdir={directory}",
            "-q",
            *extra,
        ]
    )
)
