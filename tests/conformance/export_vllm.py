"""Export upstream's SELF-VALIDATED corpus; run in the real pinned vLLM env.

uv run tests/conformance/export_vllm.py /path/to/vllm /tmp/vllm-corpus.json
No copies of parser logic, import stubs, or hand-authored expected outputs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

VLLM_VERSION = "0.31.0"
VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
FAMILIES = (
    "qwen3",
    "minimax_m2",
    "gemma4",
    "nemotron_v3",
    "deepseek_v4",
    "glm47_moe",
    "kimi_k2",
    "inkling",
)


def verify_source(source: Path):
    import vllm

    commit = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={source}",
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    assert commit == VLLM_COMMIT, (commit, VLLM_COMMIT)
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={source}",
            "-C",
            str(source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "tests",
            "vllm",
        ],
        check=True,
    )
    runtime = Path(vllm.__file__).parent
    hashes = {}
    for directory in ("parser", "reasoning", "tool_parsers", "tokenizers"):
        for file in sorted((source / "vllm" / directory).rglob("*.py")):
            relative = file.relative_to(source / "vllm")
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            assert (
                hashlib.sha256((runtime / relative).read_bytes()).hexdigest() == digest
            ), relative
            hashes[str(relative)] = digest
    return hashes


def export(source: Path) -> dict:
    assert importlib.metadata.version("vllm") == VLLM_VERSION
    # Import the installed runtime BEFORE adding the upstream tests to sys.path.
    import vllm.parser  # noqa: F401

    runtime_hashes = verify_source(source)
    sys.path.insert(0, str(source))
    from tests.parser.engine.trace_builder import build_samples
    from tests.tool_parsers.test_deepseekv3_tool_parser import TestDeepSeekV3ToolParser

    corpus = {}
    for family in FAMILIES:
        records = []
        # build_samples validates every sample using the actual upstream parser.
        for sample in build_samples(family):
            records.append(
                {
                    "id": sample.id,
                    "description": sample.description,
                    "text": "".join(text for _, text in sample.tokens),
                    "reasoning": sample.expected_reasoning,
                    "content": sample.expected_content,
                    "tool_calls": sample.expected_tool_calls,
                    "tools": [
                        t.model_dump(exclude_none=True) for t in sample.tools or []
                    ],
                    "chat_template_kwargs": sample.chat_template_kwargs or {},
                    "prompt_text": "".join(
                        next(text for text, tid in sample.vocab.items() if tid == token)
                        for token in sample.prompt_token_ids or []
                    ),
                }
            )
        corpus[family] = records
    # Invoke the ORIGINAL plain fixture function, without rewriting the test
    # or its data. The unchanged legacy suite also exercises this fixture.
    config = asdict(
        TestDeepSeekV3ToolParser.test_config.__wrapped__(TestDeepSeekV3ToolParser())
    )
    legacy = [
        {"id": f"deepseek_v3-{name}", "text": value}
        for name, value in config.items()
        if name.endswith("_output") and isinstance(value, str)
    ]
    legacy += [
        {"id": f"deepseek_v3-malformed-{i}", "text": value}
        for i, value in enumerate(config["malformed_input_outputs"])
    ]
    files = [
        "tests/parser/engine/trace_builder.py",
        "tests/parser/engine/replay_harness.py",
        "tests/tool_parsers/test_deepseekv3_tool_parser.py",
    ]
    return {
        "reference": {
            "version": VLLM_VERSION,
            "commit": VLLM_COMMIT,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in (
                    "transformers",
                    "openai-harmony",
                    "torch",
                    "huggingface-hub",
                    "xgrammar",
                )
            },
            "runtime_parser_files": runtime_hashes,
            "files": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in files
            },
        },
        "families": corpus,
        "legacy_tool_configs": {"deepseek_v3": config},
        "legacy_tool_traces": {"deepseek_v3": legacy},
    }


if __name__ == "__main__":
    output = export(Path(sys.argv[1]).resolve())
    Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + "\n")
