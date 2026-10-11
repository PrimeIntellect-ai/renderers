"""Summarize the raw comparison without changing any comparison result."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from tests.conformance.vllm_reference import same_projection
from tests.parity import MODEL_CATALOG


def differences(record):
    if "error" in record:
        return ["reference-or-fixture-error"]
    reference = record["vllm"]
    streams = list(reference["stream"].values())
    expected = streams[0] if streams else reference["complete"]
    categories = []
    if reference.get("complete_error") or reference.get("stream_errors"):
        categories.append("parser-phase-error")
    if reference.get("decode_agrees") is False:
        categories.append("serving-detokenization")
    if any(not same_projection(value, expected) for value in streams):
        categories.append("stream-chunk-dependence")
    if not same_projection(reference["complete"], expected):
        categories.append("complete-versus-stream")
    native = record["native"]
    if (
        expected is not None
        and record["scenario"].startswith("template-")
        and native["reasoning"] is not None
        and expected["reasoning"] is None
        and any(
            marker in expected["content"]
            for marker in ("<think>", "<think:opensource>")
        )
    ):
        categories.append("template-channel-configuration")
    if expected is not None and (native["reasoning"], native["content"]) != (
        expected["reasoning"],
        expected["content"],
    ):
        categories.append("channel-bytes")
    if expected is not None and not same_projection(
        native["tool_calls"], expected["tool_calls"]
    ):
        categories.append("tool-projection")
    if any(a["status"] != "ok" for a in record["attempts"]):
        categories.append("malformed-attempt-metadata")
    return categories


POLICY = {
    "template-channel-configuration": {
        "authority": "explicit-official-template-markers",
        "decision": "Honor explicit reasoning markers emitted by the official template. A disabled or incompatible serving parser profile must be aligned with that grammar, rather than displaying valid reasoning as final text with think tags.",
    },
    "generation-prefill": {
        "authority": "unmodified-official-encoder",
        "decision": "Use the encoder's actual generation prefix. Remove GPT-OSS's unconfigured forced-analysis suffix; do not change the oracle to imitate the renderer.",
    },
    "parser-phase-error": {
        "authority": "official-generation-prefill",
        "decision": "Keep complete and streaming phase failures visible. The actual prompt must be supported. GPT-OSS should use the official bare assistant prefill rather than an implicit forced-analysis profile that the serving parser does not initialize.",
    },
    "serialization": {
        "authority": "pinned-official-chat-template",
        "decision": "Match token IDs exactly, including generation prefills and turn closers. Bridges preserve sampled IDs and append only the encoder's boundary suffix.",
    },
    "channel-bytes": {
        "authority": "vllm-stream-with-exact-prompt",
        "decision": "Preserve nonempty channel bytes; do not strip whitespace to make a comparison pass. An explicit channel marker and the actual generation prefix determine reasoning state.",
    },
    "complete-versus-stream": {
        "authority": "live-stream-with-exact-prompt",
        "decision": "Prefer the prompt-aware streaming result for live completions. Keep the complete-parser result visible as a separate upstream discrepancy.",
    },
    "stream-chunk-dependence": {
        "authority": "template-protocol",
        "decision": "Reject chunk-dependent behavior. No arbitrary chunk size is a trustworthy oracle; resolve the upstream bug against its fixture and the official template before accepting native equivalence.",
    },
    "serving-detokenization": {
        "authority": "sampled-token-bytes",
        "decision": "Preserve sampled IDs. Report serving-inserted spaces or withheld incomplete Unicode separately; configure serving to avoid inserted spaces rather than teaching the native parser to invent bytes.",
    },
    "tool-projection": {
        "authority": "official-tool-format-and-schema",
        "decision": "Preserve string argument bytes and compare valid arguments as JSON values. Follow vLLM's grammar for valid calls, but distinguish streamed partial intents from complete executable calls.",
    },
    "malformed-attempt-metadata": {
        "authority": "complete-call-validity",
        "decision": "Retain malformed attempts and their raw text/status. Partial JSON, unclosed blocks, missing names, and unknown tools must not become executable merely because a serving parser emitted a delta.",
    },
    "reference-or-fixture-error": {
        "authority": "reproducible-reference",
        "decision": "Fail the audit; an unavailable or misconfigured reference is not evidence of fidelity.",
    },
}


def summarize(result):
    formats = {case.model: case.resolved_renderer for case in MODEL_CATALOG}
    rows = result["cases"]
    prefills = result.get("generation_prefills", [])
    counts = Counter(category for row in rows for category in differences(row))
    counts["generation-prefill"] = sum(not r["agrees"] for r in prefills)
    ref = result["reference"]
    lines = [
        "# Renderer fidelity audit",
        "",
        f"Reference: vLLM {ref['version']}, source `{ref['commit']}`.",
        "",
        f"{len(rows)} comparisons; {sum(not r['agrees'] for r in rows)} raw projection disagreements.",
        "A disagreement is reported even when the recommended contracts differ intentionally. No output is normalized or exempted from the strict comparison.",
        f"Unmodified encoder prefills: {len(prefills)} comparisons, {sum(not r['agrees'] for r in prefills)} disagreements.",
        "",
        "| Format | Comparisons | Raw disagreements |",
        "| --- | ---: | ---: |",
    ]
    for name in sorted(set(formats.values())):
        group = [r for r in rows if formats[r["model"]] == name]
        lines.append(
            f"| {name} | {len(group)} | {sum(not r['agrees'] for r in group)} |"
        )
    lines += ["", "## Authority decisions", ""]
    for category, policy in POLICY.items():
        lines += [
            f"- **{category}** ({counts.get(category, 0)} cases): {policy['decision']} Authority: `{policy['authority']}`."
        ]
    lines += ["", "## Representative inputs and outputs", ""]
    if any(not row["agrees"] for row in prefills):
        row = next(row for row in prefills if not row["agrees"])
        shown = {
            key: row[key]
            for key in ("model", "settings", "native_tail", "reference_tail")
        }
        lines += [
            "### generation-prefill",
            "",
            "```json",
            json.dumps(shown, indent=2),
            "```",
            "",
        ]
    for category in counts:
        if category == "generation-prefill":
            continue
        row = next(r for r in rows if category in differences(r))
        shown = {
            key: row[key]
            for key in (
                "case",
                "prompt_tail",
                "prompt_profile",
                "settings",
                "text",
                "native",
                "reasoning_complete",
                "attempts",
                "vllm",
                "error",
            )
            if key in row
        }
        lines += [
            f"### {category}",
            "",
            "```json",
            json.dumps(shown, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(summarize(result))
