"""Manual audit: uv run pytest tests/audit_prefix_stability.py -q."""

import json
from difflib import unified_diff
from pathlib import Path

import pytest
from conftest import _load
from renderers import DefaultRenderer, create_renderer
from renderers.base import RENDERER_REGISTRY, _populate_registry
from renderers.configs import _config_class_for

_CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "prefix_stability.json").read_text()
)


def _prefix_diff(tokenizer, before, after):
    candidate = after[: len(before)]
    if before == candidate:
        return "No differing token: this witness now preserves the original prefix."
    index = next(
        (i for i, (left, right) in enumerate(zip(before, candidate)) if left != right),
        min(len(before), len(candidate)),
    )
    diff = "\n".join(
        unified_diff(
            tokenizer.decode(before).splitlines(),
            tokenizer.decode(candidate).splitlines(),
            fromfile="original render",
            tofile="same-length prefix after append",
            lineterm="",
        )
    )
    return (
        f"First differing token: {index}\n"
        f"Before: {before[index : index + 8]}\n"
        f"After:  {candidate[index : index + 8]}\n{diff}"
    )


@pytest.mark.parametrize("case", _CORPUS["cases"], ids=lambda case: case["id"])
def test_prefix_stability_witness(case):
    config = dict(case["config"])
    name = config.pop("name")
    config_cls = _config_class_for(name)
    tokenizer, _ = _load(case["model"], name)
    renderer = create_renderer(tokenizer, config_cls(**config))
    scenario = _CORPUS["scenarios"][case["scenario"]]
    messages = scenario["messages"]
    assert messages[-1]["role"] == "assistant" and scenario["append"]
    before = renderer.render_ids(messages, tools=scenario.get("tools"))
    after = renderer.render_ids(
        messages + scenario["append"], tools=scenario.get("tools")
    )
    extends = after[: len(before)] == before
    assert renderer.is_prefix_stable is case["stable"], case["reason"]
    assert extends is case["stable"], (
        f"{case['reason']}\n{_prefix_diff(tokenizer, before, after)}"
    )


def test_corpus_covers_builtin_renderers():
    _populate_registry()
    assert {case["config"]["name"] for case in _CORPUS["cases"]} | {"default"} == set(
        RENDERER_REGISTRY
    )
    assert len({case["id"] for case in _CORPUS["cases"]}) == len(_CORPUS["cases"])
    # An opaque template is unknown; inventing a counterexample would be wrong.
    assert DefaultRenderer(object()).is_prefix_stable is False


def test_stability_uses_auto_resolved_template_kwargs():
    tokenizer, _ = _load("Qwen/Qwen3.8-27B", "qwen3.8")
    assert create_renderer(tokenizer).is_prefix_stable is True
    assert (
        create_renderer(
            tokenizer, chat_template_kwargs={"preserve_thinking": False}
        ).is_prefix_stable
        is False
    )
