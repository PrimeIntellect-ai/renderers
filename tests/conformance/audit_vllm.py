"""Compare public renderers to real vLLM; emit every disagreement before failing.

uv run --with vllm==0.31.0 -m tests.conformance.audit_vllm \
    /tmp/vllm-corpus.json /tmp/renderer-fidelity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from renderers.base import ToolCallParseStatus
from tests.conformance.harness import (
    reference,
    renderer,
    tokenizer,
    unmodified_generation_reference,
)
from tests.conformance.report import POLICY, differences
from tests.conformance.test_rollouts import canonical_turn, matrix
from tests.conformance.test_tool_cycles import (
    cases as tool_cases,
)
from tests.conformance.test_tool_cycles import (
    reference_turn,
    tool_renderer,
    tool_turn,
)
from tests.conformance.vllm_reference import (
    PARSERS,
    parse_reference,
    projection,
    same_projection,
)
from tests.parity import MODEL_CATALOG, TOOLS

CORPUS_ROUTES = {
    "prime-qwen3": "qwen3",
    "qwen3.5": "qwen3",
    "qwen3.6": "qwen3",
    "qwen3.8": "qwen3",
    "glm-4.5": "glm47_moe",
    "glm-5": "glm47_moe",
    "glm-5.1": "glm47_moe",
    "glm-5.3": "glm47_moe",
    "gemma4": "gemma4",
    "deepseek-v4": "deepseek_v4",
    "minimax-m2": "minimax_m2",
    "nemotron-3": "nemotron_v3",
    "nemotron-3-ultra": "nemotron_v3",
    "nemotron-3.5": "nemotron_v3",
    "kimi-k2": "kimi_k2",
    "kimi-k2.5": "kimi_k2",
    "inkling": "inkling",
}


def native_projection(parsed):
    return projection(
        parsed.reasoning_content,
        parsed.content,
        [
            {"name": call.name, "arguments": call.arguments}
            for call in parsed.tool_calls
            if call.status == ToolCallParseStatus.OK
        ],
    )


def generation_records(models=None):
    records = []
    for parameter in matrix():
        case, kwargs = parameter.values
        if models and case.model not in models:
            continue
        messages = [{"role": "user", "content": "Test."}]
        actual = renderer(case, kwargs).render_ids(messages, add_generation_prompt=True)
        expected = unmodified_generation_reference(case, messages, kwargs)
        tok = tokenizer(case.model)
        records.append(
            {
                "model": case.model,
                "settings": kwargs,
                "agrees": actual == expected,
                "native_ids": actual,
                "reference_ids": expected,
                "native_tail": tok.decode(actual[-96:], skip_special_tokens=False),
                "reference_tail": tok.decode(expected[-96:], skip_special_tokens=False),
            }
        )
    return records


def inputs(corpus):
    # Every structured-inference configuration, including actual bridged
    # prompts and structured parser output reused as conversation history.
    for parameter in matrix():
        case, kwargs = parameter.values
        history = [{"role": "user", "content": "What is two plus two?"}]
        next_prompt = None
        for turn in range(3):
            try:
                native = renderer(case, kwargs)
                prompt, sampled, _, _, _ = canonical_turn(case, kwargs, native, history)
                if next_prompt is not None:
                    prompt = next_prompt.token_ids
                completion = (
                    sampled[:-1]
                    if sampled and sampled[-1] in native.get_stop_token_ids()
                    else sampled
                )
                yield (
                    case,
                    kwargs,
                    f"template-continuation/turn-{turn}",
                    prompt,
                    completion,
                    [],
                )
                parsed = native.parse_response(sampled, prompt_ids=prompt)
                assistant = {"role": "assistant", "content": parsed.content}
                if parsed.reasoning_content is not None:
                    assistant["reasoning_content"] = parsed.reasoning_content
                new = [{"role": "user", "content": f"Next query {turn}."}]
                bridge = native.bridge_to_next_turn(prompt, sampled, new)
                history += [assistant] + new
                next_prompt = (
                    bridge
                    if bridge is not None
                    else native.render(history, add_generation_prompt=True)
                )
            except Exception as exc:
                yield (
                    case,
                    kwargs,
                    f"fixture-error/{type(exc).__name__}: {exc}",
                    [],
                    [],
                    [],
                )
                break
    for parameter in tool_cases():
        (case,) = parameter.values
        kwargs = {"is_training": False} if case.resolved_renderer == "hy3" else {}
        native = tool_renderer(case, kwargs)
        for city in ("Paris", "", "  Paris\nMünchen ☁  "):
            history = [
                {"role": "system", "content": "Use the weather tool."},
                {"role": "user", "content": "Weather?"},
            ]
            next_prompt = None
            for turn in range(2):
                assistant, new = tool_turn(case, turn, city)
                prompt, sampled, _ = reference_turn(
                    case, native, history, assistant, new, kwargs
                )
                if next_prompt is not None:
                    prompt = next_prompt.token_ids
                completion = (
                    sampled[:-1]
                    if sampled and sampled[-1] in native.get_stop_token_ids()
                    else sampled
                )
                yield (
                    case,
                    kwargs,
                    f"template-tool-cycle/{city!r}/turn-{turn}",
                    prompt,
                    completion,
                    TOOLS,
                )
                parsed = native.parse_response(sampled, prompt_ids=prompt, tools=TOOLS)
                assistant = {
                    "role": "assistant",
                    "content": parsed.content,
                    "reasoning_content": parsed.reasoning_content,
                    "tool_calls": [
                        {
                            "id": call.id or new[0]["tool_call_id"],
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in parsed.tool_calls
                        if call.status == ToolCallParseStatus.OK
                    ],
                }
                new[0]["tool_call_id"] = assistant["tool_calls"][0]["id"]
                bridge = native.bridge_to_next_turn(prompt, sampled, new, tools=TOOLS)
                history += [assistant] + new
                next_prompt = (
                    bridge
                    if bridge is not None
                    else native.render(history, tools=TOOLS, add_generation_prompt=True)
                )
    # Legacy V3/R1 checkpoints have no structured tool-history template, but
    # their public completion parsers support the upstream wire protocol.
    for case in MODEL_CATALOG:
        if case.resolved_renderer not in {"deepseek-v3", "deepseek-r1"}:
            continue
        tok = tokenizer(case.model)
        prompt = reference(
            case, [{"role": "user", "content": "Test."}], {}, add_generation_prompt=True
        )
        for record in corpus["legacy_tool_traces"]["deepseek_v3"]:
            # R1's exact live prefix opens reasoning. Close that empty channel
            # before sampling the unmodified upstream tool-parser trace.
            text = (
                "</think>" if case.resolved_renderer == "deepseek-r1" else ""
            ) + record["text"]
            ids = tok.encode(text, add_special_tokens=False)
            yield case, {}, record["id"], prompt, ids, []
            if len(ids) > 1:
                yield (
                    case,
                    {},
                    record["id"] + "/truncated",
                    prompt,
                    ids[: len(ids) // 2],
                    [],
                )

    # Upstream's independently self-validated multi-format scenario corpus.
    for case in MODEL_CATALOG:
        family = CORPUS_ROUTES.get(case.resolved_renderer)
        if family is None:
            continue
        kwargs = {}
        fields = renderer(case, {}).config.template_field_names()
        for field in ("enable_thinking", "thinking"):
            if field in fields:
                kwargs[field] = True
        for record in corpus["families"][family]:
            settings = dict(kwargs)
            upstream_settings = dict(record["chat_template_kwargs"])
            if family == "deepseek_v4" and "thinking" in upstream_settings:
                settings["enable_thinking"] = upstream_settings.pop("thinking")
            settings.update(upstream_settings)
            tok = tokenizer(case.model)
            tools = record["tools"]
            prompt = (
                tok.encode(record["prompt_text"], add_special_tokens=False)
                if record["prompt_text"]
                else reference(
                    case,
                    [{"role": "user", "content": "Test."}],
                    settings,
                    tools=tools or None,
                    add_generation_prompt=True,
                )
            )
            ids = tok.encode(record["text"], add_special_tokens=False)
            yield case, settings, record["id"], prompt, ids, tools
            if record["text"].startswith("<think>"):
                for whitespace in ("\n", "\r\n"):
                    spaced = tok.encode(
                        whitespace + record["text"], add_special_tokens=False
                    )
                    yield (
                        case,
                        settings,
                        record["id"] + f"/leading-{whitespace!r}",
                        prompt,
                        spaced,
                        tools,
                    )
            # A token-budget stop halfway through the very SAME upstream output.
            # Compare partial serving intents too; executable validity remains
            # a separate renderer contract, not implied by streamed deltas.
            if len(ids) > 1:
                yield (
                    case,
                    settings,
                    record["id"] + "/truncated",
                    prompt,
                    ids[: len(ids) // 2],
                    tools,
                )


def audit(corpus, on_record, models=None):
    assert set(PARSERS) == {c.resolved_renderer for c in MODEL_CATALOG}
    selected = set(models) if models else {case.model for case in MODEL_CATALOG}
    assert selected <= {case.model for case in MODEL_CATALOG}, selected
    records = []
    for case, kwargs, scenario, prompt, completion, tools in inputs(corpus):
        if models and case.model not in models:
            continue
        tok = tokenizer(case.model)
        key = f"{case.model}/{scenario}/{json.dumps(kwargs, sort_keys=True)}"
        record = {
            "case": key,
            "model": case.model,
            "format": case.resolved_renderer,
            "prompt_profile": "forced-analysis"
            if case.oracle == "harmony"
            else "official-generation-prefill",
            "settings": kwargs,
            "scenario": scenario,
            "prompt_ids": prompt,
            "prompt_tail": tok.decode(prompt[-96:], skip_special_tokens=False),
            "completion_ids": completion,
            "text": tok.decode(completion, skip_special_tokens=False),
            "tools": tools,
        }
        try:
            if scenario.startswith("fixture-error/"):
                raise RuntimeError(scenario)
            native = tool_renderer(case, kwargs) if tools else renderer(case, kwargs)
            parsed = native.parse_response(
                completion, prompt_ids=prompt, tools=tools or None
            )
            record["native"] = native_projection(parsed)
            record["reasoning_complete"] = parsed.reasoning_complete
            record["attempts"] = [
                {"status": c.status.value, "raw": c.raw, "span": c.token_span}
                for c in parsed.tool_calls
            ]
            record["vllm"] = parse_reference(
                case, tok, prompt, completion, tools, kwargs
            )
            streams = list(record["vllm"]["stream"].values())
            expected = streams[0] if streams else record["vllm"]["complete"]
            record["agrees"] = (
                not record["vllm"].get("stream_errors")
                and same_projection(record["native"], expected)
                and all(same_projection(value, expected) for value in streams)
            )
            record["vllm_complete_agrees_with_stream"] = same_projection(
                record["vllm"]["complete"], expected
            )
            record["vllm_chunk_invariant"] = not record["vllm"].get(
                "stream_errors"
            ) and all(same_projection(value, expected) for value in streams)
            record["decision"] = {
                "authority": "vllm-stream-with-exact-prompt"
                if streams
                else (
                    "reference-phase-failure"
                    if expected is None
                    else "configured-identity-parser"
                ),
                "expected": expected,
                "reason": "The real generation prompt determines the initial channel. Streaming chunk size must not change the result. Raw attempts are retained separately.",
            }
        except Exception as exc:
            record["agrees"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["differences"] = differences(record)
        record["authority_decisions"] = {
            category: POLICY[category] for category in record["differences"]
        }
        signature = json.dumps(record, sort_keys=True).encode()
        record["fingerprint"] = hashlib.sha256(signature).hexdigest()
        records.append(record)
        on_record(record)
        print(f"{'PASS' if record['agrees'] else 'DIFF'} {key}", flush=True)
    assert {row["model"] for row in records} == selected
    assert {
        row["model"]
        for row in records
        if row["scenario"].startswith(("template-tool-cycle/", "deepseek_v3-"))
    } == selected
    prefills = generation_records(models)
    return {
        "reference": corpus["reference"],
        "tokenizers": json.loads(
            Path(__file__).with_name("tokenizers.json").read_text()
        ),
        "parser_profiles": PARSERS,
        "authority_policy": POLICY,
        "generation_prefills": prefills,
        "cases": records,
        "summary": {
            "total": len(records),
            "disagreements": sum(not r["agrees"] for r in records),
            "generation_prefill_disagreements": sum(not r["agrees"] for r in prefills),
        },
    }


if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument("corpus", type=Path)
    options.add_argument("report", type=Path)
    options.add_argument("--model", action="append")
    args = options.parse_args()
    with args.report.with_suffix(".jsonl").open("w") as progress:

        def save(record):
            progress.write(json.dumps(record) + "\n")
            progress.flush()

        result = audit(json.loads(args.corpus.read_text()), save, models=args.model)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
    raise SystemExit(
        bool(
            result["summary"]["disagreements"]
            or result["summary"]["generation_prefill_disagreements"]
        )
    )
