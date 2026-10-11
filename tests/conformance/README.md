# Renderer conformance

The renderer is a black box. Tests use `create_renderer`, `render`, `render_ids`,
`parse_response`, `bridge_to_next_turn`, and the production `generate` client.
Only sampling is scripted. No native parser internals generate expectations.

The shared model/scenario catalog drives all 29 concrete renderer formats and
45 checkpoints. `tokenizers.json` pins the reviewed assets; a new renderer must
have a catalog checkpoint. Complete transcripts exercise every valid scenario
and finite template setting, including raw assistant serialization. Structured
inference settings also run three user turns, token-budget stops, and two tool
cycles with ordinary, empty, and whitespace/Unicode string arguments. Parsed
public-client messages are reused in subsequent turns. Raw assistant
passthrough and training serialization belong to the transcript matrix, rather
than the structured-message client loop.

Generation boundaries must come from the official generation prompt. Historical
assistant headers are never inserted into completion tokens. R1 and Harmony
have explicit live fixtures because their history format removes reasoning or
changes channels. Closed thinking prefills use the official finished body and
turn close. No `.strip()` is allowed on parsed nonempty channel contents.

A bridge must preserve every sampled token and append the exact template suffix.
The suffix is obtained by perturbing only the previous assistant body in two
reference renders and locating their common turn-close suffix. This still works
when a new query changes historical reasoning. A declined bridge takes a full
reference-checked render. Empty/absent wire reasoning and null/string content
are mapped to the renderer API; nonempty whitespace is never normalized.

## Independent references

Most encoders are the checkpoint's unmodified Jinja template. The isolated CI
jobs set `RENDERERS_UPSTREAM_ENCODER=1` to execute DeepSeek V4's actual Python
encoder at its pinned revision and SHA256. The host suite uses the existing
portable DeepSeek adapter. That adapter is additionally checked against the
actual encoder in CI. Harmony's existing reference profile uses the official
encoder with an explicit forced-analysis prefill. An additional normative
generation-prefix test calls the unmodified official encoder and exposes that
profile's divergence instead of silently assuming it is template fidelity.

vLLM is **0.31.0**, source commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6`, matching the production lockfile.
The reference job pins transformers 5.17.0 and openai-harmony 0.0.8 and records
the installed reference dependencies in its exported corpus.
`run_upstream.py` runs upstream engine tests plus selected unchanged reasoning
and tool parser suites without editing their expectations. `prepare_upstream.py`
pins their public tokenizer snapshots in an isolated HF cache; the unchanged
fixtures resolve `main` to those snapshots and run offline. The legacy gated
Llama fixture is not in that selection; Llama's actual public parser is covered
by native-versus-vLLM tool cycles using the reviewed unrestricted tokenizer.
`export_vllm.py` imports the upstream trace builder normally and exports its
self-validated scenarios. It also invokes the original DeepSeek V3 test-config
fixture function to export the unchanged legacy tool cases, including malformed
calls, for V3 and R1's completion parsers. Those checkpoints do not have a
structured tool-history template. Every checkpoint must have both reasoning
and tool-wire comparisons; missing coverage fails the audit. There are no import stubs,
AST extraction, copied parser implementations, or fake tokenizer IDs in the
native-versus-vLLM comparison.
The installed parser, reasoning, tool-parser, and tokenizer Python files must
hash-identically to the pinned source checkout before upstream tests run.

`audit_vllm.py` retokenizes that corpus with the real pinned checkpoint tokenizer,
adds truncations, and includes template-derived live completions for every
renderer. `vllm_reference.py` uses `ParserManager` and the serving parser names,
plus vLLM's own incremental detokenizer. Complete and streaming outputs are
recorded separately, with chunks of 1, 2, 5, and the whole completion.

Run the portable tests:

```sh
uv run pytest tests/conformance
```

Run the isolated reference job, with the pinned source checkout:

```sh
uv pip install --python .venv/bin/python vllm==0.31.0 transformers==5.17.0 openai-harmony==0.0.8
HF_HUB_CACHE=/tmp/vllm-tokenizers uv run --no-sync tests/conformance/prepare_upstream.py
uv run --no-sync tests/conformance/run_upstream.py /path/to/vllm
HF_HUB_CACHE=/tmp/vllm-tokenizers HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync tests/conformance/run_upstream.py /path/to/vllm --suite reasoning
HF_HUB_CACHE=/tmp/vllm-tokenizers HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync tests/conformance/run_upstream.py /path/to/vllm --suite tools
uv run --no-sync tests/conformance/export_vllm.py /path/to/vllm /tmp/vllm-corpus.json
RENDERERS_UPSTREAM_ENCODER=1 uv run --no-sync -m tests.conformance.audit_vllm \
  /tmp/vllm-corpus.json /tmp/renderer-fidelity.json
uv run --no-sync -m tests.conformance.report \
  /tmp/renderer-fidelity.json /tmp/renderer-fidelity.md
```

## Disagreement policy

Serialization follows the official template. Live channels follow the exact
generation prefix and vLLM's streaming projection when it is chunk invariant.
Chunk-dependent reference output is an upstream defect to resolve against the
template and fixture, not a reason to bless one arbitrary chunk size.
Complete-versus-streaming disagreement is reported; the prompt-aware live
stream owns the live contract. Each report includes inputs, settings,
raw text, both reference outputs, native output, an authority decision, and a
fingerprint. Any native-versus-authoritative or chunk-size disagreement fails
CI after writing the full report; there is no blanket skip/xfail allowlist.
`report.py` groups differences and recommends an authority per contract without
changing any comparison result. These recommendations do not turn known
disagreements green. The broader fidelity gate can remain red while the
profile/template/client tests pass; that is evidence of outstanding parser
work. The unmodified-prefill guard separately rejects a renderer that injects
an unconfigured channel prefill. It is not skipped or marked xfail.

Tool arguments compare as JSON values, preserving every string byte from the
official encoding. Serving-inserted spaces and withheld incomplete Unicode are
reported as detokenization differences, independently of parsing. Random
serving-generated call IDs are excluded from that projection. Kimi's native
qualified IDs are retained for history/tool result association. Successful
calls must match the grammar and schema; a serving parser that trims payload
strings loses to the template's original payload. Malformed attempts remain
extra renderer metadata and never become executable calls merely to match a
salvage parser's partial intent. Raw sampled
IDs remain intact irrespective of any structured projection.

The test suite checks text/token protocols. Existing processor parity tests
cover image/audio/video expansion; the new text suite does not claim to validate
GPU sampling, media processing, or the dashboard UI.
