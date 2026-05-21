# Native Runtime Performance Ideas

This document is the working plan for making the Rust/PyO3 renderers faster while
keeping parity visible at every step. The goal is not to guess where the speedup
comes from. Each change should land with a benchmark artifact that compares the
new commit against the previous baseline.

## Current Shape

The benchmark entry point is:

```bash
uv run maturin develop --manifest-path crates/renderers-py/Cargo.toml --release
uv run python benchmarks/native_vs_python_qwen3.py --families all --min-time 0.35 --repeats 7 --memory-loops 1000
```

The script already compares:

- Python renderer public APIs.
- Native list-returning APIs.
- Native NumPy-returning APIs where available.
- `render_ids`, `parse_response`, and `bridge_to_next_turn`.
- Multiple families: Qwen, GLM, DeepSeek, Kimi, MiniMax, and Nemotron.

The script now has progress and reproducibility support, so it can be used as
the optimization scoreboard before and after each runtime commit.

## Benchmark Harness First

Before optimizing runtime code, make the benchmark produce stable artifacts.
This lets every commit answer the same question: what got faster, what got
slower, and by how much?

### 1. Structured Output

Implemented flags in `benchmarks/native_vs_python_qwen3.py`:

```bash
--json-out benchmark-results/native-runtime/latest.json
--markdown-out benchmark-results/native-runtime/latest.md
--baseline benchmark-results/native-runtime/baseline.json
```

The JSON includes:

- Git commit SHA and dirty state.
- Python version, Rust version, platform, CPU model if available.
- Native extension build mode.
- Benchmark args: families, repeats, min time, memory loops.
- One row per family, operation, scenario, and API path.
- Median, min, max, loop count, token count, and memory peak.
- Per-family geomean and overall geomean.

The Markdown includes:

- A short summary table with overall list and NumPy geomean speedups.
- A per-family table.
- A worst regressions table versus baseline.
- A best improvements table versus baseline.
- Skipped cases and why they were skipped.

Raw terminal tables are still printed, but the JSON is the source of truth.

### 2. Live Progress

The full all-family benchmark is long enough that it renders progress as it
runs. Progress output goes to stderr.

Suggested progress lines:

```text
[1/120] qwen3 render_ids medium_gen_prompt: python
[1/120] qwen3 render_ids medium_gen_prompt: native list
[1/120] qwen3 render_ids medium_gen_prompt: native np
[1/120] qwen3 render_ids medium_gen_prompt: memory
```

The script also prints a compact family summary after each family finishes:

```text
family=qwen3 rows=12 list_geomean=1.81x np_geomean=2.03x elapsed=31.2s
```

This matters because performance work can fail halfway through a full
matrix. Partial progress should still be useful.

### 3. Compare Mode

Implemented comparison mode:

```bash
uv run python benchmarks/native_vs_python_qwen3.py \
  --families all \
  --baseline benchmark-results/native-runtime/baseline.json \
  --json-out benchmark-results/native-runtime/$SHA.json \
  --markdown-out benchmark-results/native-runtime/$SHA.md
```

Comparison rules:

- Compare matching `family + operation + scenario + path`.
- Report ratios against the baseline medians.
- Treat missing baseline rows as new coverage, not wins.
- Treat missing current rows as failures unless explicitly skipped.
- Flag any row slower than baseline by more than 5 percent.
- Flag any row faster than baseline by more than 5 percent.

The script exits non-zero only with an explicit flag such as:

```bash
--fail-on-regression 5
```

That keeps exploratory runs flexible while making CI or local gates strict when
we want them strict.

### 4. Add a Small/Fast Profile

Use a sub-minute profile before every larger run:

```bash
uv run python benchmarks/native_vs_python_qwen3.py \
  --families qwen3,qwen35,kimi_k2 \
  --min-time 0.02 \
  --repeats 3 \
  --memory-loops 20 \
  --json-out benchmark-results/native-runtime/smoke.json
```

The smoke profile catches broken benchmark plumbing and obvious parity failures.
Only after it passes should we run the full profile:

```bash
uv run python benchmarks/native_vs_python_qwen3.py \
  --families all \
  --min-time 0.35 \
  --repeats 7 \
  --memory-loops 1000 \
  --json-out benchmark-results/native-runtime/$SHA.json \
  --markdown-out benchmark-results/native-runtime/$SHA.md
```

## Commit Measurement Loop

Every runtime optimization commit should follow this loop:

1. Build native extension in release mode.

```bash
uv run maturin develop --manifest-path crates/renderers-py/Cargo.toml --release
```

2. Run correctness checks.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked
cargo test --workspace
uv run pytest -m parity tests/test_native_parity.py -q -rs
env RENDERERS_NATIVE=all uv run pytest \
  tests/test_render_ids.py \
  tests/test_bridge.py \
  tests/test_roundtrip.py \
  tests/test_message_indices.py \
  tests/test_native_router.py \
  tests/test_native_vision.py \
  tests/test_native_numpy.py \
  -q -rs
```

3. Run benchmark smoke.

```bash
uv run python benchmarks/native_vs_python_qwen3.py \
  --families qwen3,qwen35,kimi_k2 \
  --min-time 0.02 \
  --repeats 3 \
  --memory-loops 20
```

4. Run full benchmark and save artifacts.

```bash
SHA=$(git rev-parse --short HEAD)
uv run python benchmarks/native_vs_python_qwen3.py \
  --families all \
  --min-time 0.35 \
  --repeats 7 \
  --memory-loops 1000 \
  --baseline benchmark-results/native-runtime/baseline.json \
  --json-out benchmark-results/native-runtime/$SHA.json \
  --markdown-out benchmark-results/native-runtime/$SHA.md
```

5. Commit code and benchmark artifact together when the benchmark is part of the
claim. If artifacts are too noisy for git, commit the code and paste the saved
Markdown summary into the commit message body or PR description.

## Performance Work Queue

The highest-value work is reducing repeated Python object parsing, repeated tool
formatting, and repeated token list materialization. Single fresh calls still pay
for Python input objects and tokenizer work, so the 8x to 10x target is most
realistic for prepared, batched, or multiturn workloads.

### A. Prepared Tools

Problem:

The examples and benchmarks pass the same tool schema repeatedly. Today the
native path still receives Python objects, converts them to Rust structures, and
formats schema text for each render.

Idea:

Add a Python-visible prepared tool handle:

```python
prepared_tools = renderer.prepare_tools(TOOLS)
ids = renderer.render_ids(messages, tools=prepared_tools, add_generation_prompt=True)
```

Native side:

- Parse tool specs once.
- Normalize provider-specific tool shape once.
- Pre-render static tool instruction text once.
- Pre-tokenize static tool blocks where the family template allows it.
- Keep the original public `tools=list[dict]` path as fallback.

Benchmark cases:

- Existing `large_tools_gen_prompt`.
- Existing `tool_cycle_large_schema`.
- New repeated-tools scenario that renders the same tools across many short
  prompts.

Expected proof:

- `render_ids` with tools gets faster.
- No regression for no-tools scenarios.
- SGLang and vLLM examples can use it directly because they already reuse
  `TOOLS`.

Status:

- Implemented a Python-visible native `PreparedTools` handle.
- `Renderer.prepare_tools(TOOLS)` parses Python tool dictionaries once and can
  be passed to `render_ids`, `render_ids_np`, `render_batch_ids`, and
  `render_batch_ids_np_packed`.
- Benchmark rows now include `render_ids_prepared_tools`.
- Added a native `ToolTextCache` for repeated prepared-tool prompts. Qwen3,
  Qwen3.5/Qwen3.6, GLM, Nemotron 3, MiniMax M2, and Kimi K2 now cache the fully
  rendered and pre-tokenized system/tool text block keyed by the prepared tools
  and dynamic system text. Repeated prepared-tool renders skip both tool
  formatting and tokenization.

### B. Prepared Conversation or Session

Problem:

Multiturn examples repeatedly pass Python message lists. For bridge paths, we
also repeatedly pass prompt IDs, completion IDs, and new messages across PyO3.

Idea:

Add a native session object that owns parsed messages and token buffers:

```python
session = renderer.new_session(messages, tools=prepared_tools)
prompt_ids = session.render_ids(add_generation_prompt=True)
completion = engine_completion_ids(...)
bridged_ids = session.bridge_to_next_turn(completion, new_messages)
```

Native side:

- Store parsed messages in Rust.
- Store prepared tools by reference or shared handle.
- Store previous prompt and completion buffers.
- Append new messages without reparsing the whole conversation.
- Return list IDs for existing engine APIs, and NumPy IDs for callers that can
  keep arrays.

Benchmark cases:

- Existing `bridge_to_next_turn`.
- New `session_bridge_to_next_turn`.
- Long history plus one new user message.
- Tool response extension.

Expected proof:

- Big gains on `bridge_to_next_turn`.
- Lower memory pressure on Python heap.
- Minimal Python-side example change: replace renderer calls with a session.

Status:

- Implemented a Python-visible native `RendererSession`.
- `Renderer.new_session(messages, tools=prepared_tools)` stores parsed messages
  and prepared tools in Rust.
- `session.render_ids()`, `session.render_ids_np()`,
  `session.bridge_to_next_turn()`, and `session.bridge_to_next_turn_np()` are
  available.
- Session messages are stored behind `Arc<Vec<Message>>`, so repeated
  `session.render_ids()` calls clone only a pointer before releasing the GIL.
- Benchmark rows now include `session_render_ids`.
- Bridge implementations that only need token IDs now use token-id-only render
  buffers, avoiding per-token message-index allocation on the extension path.
- `RendererSession.bridge_to_next_turn(..., update=False)` and
  `bridge_to_next_turn_np(..., update=False)` allow repeatable measurement of
  an initialized session bridge without mutating the stored prompt between
  benchmark iterations.
- Benchmark rows now include `session_bridge_to_next_turn`.
- Implemented `RendererSession.fork()` so benchmarks and callers can cheaply
  reset an initialized session state without reparsing messages or tools.
- Benchmark rows now include `session_bridge_loop`, a multi-step bridge loop
  that advances the same session through several generated turns.

### C. Batched Render APIs

Problem:

Serving systems rarely render one prompt in isolation. Even if SGLang or vLLM
does the model batching, the renderer can batch preprocessing before requests
reach the engine.

Idea:

Add:

```python
batch = renderer.render_batch_ids(messages_batch, tools=prepared_tools)
batch_np = renderer.render_batch_ids_np(messages_batch, tools=prepared_tools)
```

Native side:

- Parse one Python outer list.
- Reuse prepared tools across the batch.
- Use Rayon only after measuring thread overhead.
- Return `list[list[int]]` for current SGLang/vLLM compatibility.
- Return a packed NumPy representation for internal pipelines:
  `ids: np.ndarray[uint32]` plus `offsets: np.ndarray[int64]`.

Benchmark cases:

- 8, 32, and 128 prompt batches.
- Short prompts with large tools.
- Long histories without tools.
- Mixed prompt lengths.

Expected proof:

- Batch throughput in prompts per second improves.
- Per-prompt median latency improves for realistic batch sizes.
- No change required at SGLang/vLLM engine boundary if we return lists.

Status:

- Implemented `Renderer.render_batch_ids(...)`.
- The native batch path uses Rayon for batches of 8 or more prompts.
- Benchmark rows now include `render_batch_ids`.

### D. Packed NumPy Token Buffers

Problem:

Returning Python lists creates one Python integer object per token. NumPy avoids
that, but current SGLang/vLLM HTTP-style boundaries usually still need lists.

Idea:

Keep NumPy for renderer-internal and client-side intermediate steps:

```python
prompt_np = renderer.render_ids_np(messages, tools=prepared_tools)
parsed = renderer.parse_response_np(completion_np)
bridged_np = renderer.bridge_to_next_turn_np(prompt_np, completion_np, new_messages)
```

Native side:

- Return `uint32` arrays for token IDs.
- Accept contiguous `uint32` arrays without copying.
- Add packed batch arrays with offsets.
- Avoid list conversion until the exact engine call that requires it.

SGLang/vLLM applicability:

- Useful before and after engine generation.
- Not true end-to-end zero-copy for JSON or APIs requiring `list[int]`.
- Still useful for offline pipelines, metrics, masks, and bridge-heavy loops.

Benchmark cases:

- Existing NumPy rows.
- Add explicit `.tolist()` boundary rows:
  `render_ids_np_then_tolist`.
- Add packed batch rows:
  `render_batch_ids_np_packed`.

Expected proof:

- NumPy path stays faster than list path inside renderer.
- `.tolist()` boundary cost is visible instead of hidden.
- We can decide which examples should use NumPy and which should stay list-only.

Status:

- Existing single-prompt NumPy paths remain covered.
- Implemented `Renderer.render_batch_ids_np_packed(...)`, returning
  `(ids: np.ndarray[uint32], offsets: np.ndarray[int64])`.
- Benchmark rows now use the packed batch path as the native NumPy batch path.
- Benchmark rows now include `render_ids_np_then_tolist` so the cost of crossing
  back to engine-compatible Python lists is visible instead of hidden.

### E. Template Constant Token Caches

Problem:

Family templates contain repeated literal tokens: role tags, separators,
generation prompts, reasoning delimiters, tool delimiters, image sentinels, and
end markers.

Idea:

Pre-tokenize constant fragments when constructing each native renderer.

Native side:

- Store static token slices per family.
- Append cached token slices instead of repeatedly encoding literals.
- Keep text-render parity tests strict because whitespace and delimiter changes
  are easy to miss.

Benchmark cases:

- No-tools short prompts.
- Long histories.
- Reasoning histories.
- Structured text parts.

Expected proof:

- Broad `render_ids` improvement across families.
- Stronger gains on many-turn conversations.

### F. Dynamic Text Encode Batching

Problem:

Rendering many message parts can call the tokenizer repeatedly. Tokenizer call
overhead can dominate short fragments.

Idea:

Batch dynamic text segments where the tokenizer supports it, then interleave the
encoded pieces with cached template tokens.

Native side:

- Collect dynamic text fragments during render planning.
- Encode them in one tokenizer batch.
- Reassemble tokens in original order.
- Preserve message index accounting.

Benchmark cases:

- Long history.
- Structured text parts.
- Many short user/assistant turns.

Expected proof:

- Long history render improves.
- Message indices remain identical.
- No parse or bridge regressions.

Status:

- Added `Tokenizer::encode_batch_no_special(...)`, backed by the tokenizer
  crate's batch-fast encoder.
- Added a token-only `TokenPlanBuf` that records literal-token and dynamic-text
  operations, batch-encodes text fragments, then materializes the final token
  stream in order.
- Qwen3 `render_ids` uses the planned batch-encode path only for long no-tool
  histories. Short prompts, tool-heavy prompts, attributed `render()`, and
  bridge paths stay on the lower-overhead direct buffer.
- Benchmark rows now expose the targeted long-history render gain while keeping
  short-prompt and tool-response bridge regressions visible.
- Remaining rollout work: apply the same conservative dispatch to additional
  families only when a family-specific benchmark shows a gain.

### G. Fast Input Shape

Problem:

OpenAI-style dict messages are flexible but expensive to parse. Hot callers can
use a stricter shape if it is optional.

Idea:

Add a compact input API without replacing existing public APIs:

```python
renderer.render_fast(
    roles=["system", "user", "assistant"],
    contents=["...", "...", "..."],
    tools=prepared_tools,
)
```

Native side:

- Validate parallel arrays once.
- Avoid generic `dict` and `Content` traversal.
- Keep support for structured parts in the generic path.

Benchmark cases:

- Short chat.
- Long chat.
- Tool-heavy prompt.

Expected proof:

- Fast shape wins when the caller can provide it.
- Existing API behavior is unchanged.

Status:

- Implemented `Renderer.render_fast_ids(roles, contents, ...)`.
- Implemented `Renderer.render_fast_ids_np(roles, contents, ...)`.
- Benchmark rows now include `render_fast_ids` where the scenario is compatible
  with plain string roles and contents.

### I. Cached Template Literal Tokens

Problem:

Several family renderers still encoded fixed literal fragments on every render:
newlines, role prefixes, generation prompts, and XML close fragments.

Status:

- Qwen3 already cached the highest-frequency literal fragments.
- Qwen3.5/Qwen3.6 now cache common literal tokens at construction time:
  newline, double newline, role prefixes, assistant generation prefix, and
  `</function>\n`.
- Text render, bridge, and multimodal user rendering use the cached token
  slices.
- Nemotron 3 now caches common standalone literal tokens at construction time:
  newline, role prefixes, assistant generation prefix, and `</function>\n`.
- GLM now caches standalone newline tokens used in GLM-4.5 generation prompts
  and tool-call separators.
- MiniMax M2 now caches standalone newline, `ai\n`, and `tool` tokens.
- Kimi K2 now caches standalone newline and `assistant` tokens.
- Kimi K2.5 now caches standalone newline, `assistant`, `<think>`, and
  `<think></think>` tokens for text, bridge, and multimodal paths.
- Prepared tool text blocks are now pre-rendered and pre-tokenized for Qwen3,
  Qwen3.5/Qwen3.6, GLM, Nemotron 3, MiniMax M2, and Kimi K2 through the shared
  native `ToolTextCache`.

### H. Parse Response Fast Path

Problem:

Parse can be sub-microsecond in simple cases, but tool calls and reasoning blocks
still require scanning and allocation.

Idea:

Optimize parsing around byte/token markers:

- Search token IDs for known delimiter IDs before decoding full text.
- Decode only spans that become content, reasoning, or JSON arguments.
- Avoid JSON parsing unless a tool call delimiter exists.
- Return borrowed or compact Python objects where PyO3 allows it.

Benchmark cases:

- Plain content.
- Reasoning plus content.
- Multi-tool call.
- Long content.

Expected proof:

- Parse geomean improves.
- Multi-tool parse improves without slowing plain content.

Status:

- Qwen3.5/Qwen3.6 no longer allocate a copied `Vec<u32>` for the no-thinking
  parse path. Plain content and tool-call parse now borrow the stripped token
  slice directly.
- GLM no longer allocates a copied token vector for the no-thinking parse path.
- Qwen3 now moves plain decoded content through the no-thinking split path
  instead of cloning it into a second `String`.
- Remaining deeper work: token-delimiter partial decode for more families, and
  avoiding regex/string work inside XML tool-call spans where possible.

## SGLang and vLLM Compatibility

The examples currently pass renderer-owned token IDs to engines:

- SGLang offline uses `engine.generate(input_ids=prompt_ids, ...)`.
- SGLang online sends `"input_ids": prompt_ids` over JSON.
- vLLM offline uses `{"prompt_token_ids": prompt_ids}`.

That means:

- Prepared tools are directly usable.
- Session rendering and session bridge are directly usable.
- Batched list output is directly usable.
- NumPy buffers are useful inside the renderer/client pipeline, but many engine
  calls still need `list[int]`.
- True zero-copy across HTTP JSON is not realistic without changing the server
  protocol.

The best PR path is to preserve the existing list APIs and add opt-in fast paths.
Examples can adopt fast paths only where the call site remains clear.

The SGLang and vLLM multiturn examples now keep that shape:

- Native runs call `prepare_tools(TOOLS)` once when the renderer exposes it.
- Native runs use `new_session(messages, tools=prepared_tools)` for the first
  render and the next-turn bridge, so repeated serving-loop calls do not parse
  the same prompt/tool dictionaries again.
- `render_fast_ids(...)` remains the lighter API for local loops that already
  hold parallel role/content arrays and do not need structured content parts.

## Native/PyO3 API Map

This is the concrete mapping from the performance ideas above to the current
native extension surface and verification hooks.

| Idea | PyO3/native API | Benchmark row | Test coverage |
|---|---|---|---|
| Prepared tools | `Renderer.prepare_tools(...)`, `PreparedTools` | `render_ids_prepared_tools`, `render_batch_ids:short_batch_prepared_tools` | `tests/test_native_numpy.py::test_prepared_tools_match_raw_tools`, parity tool rows |
| Native session | `Renderer.new_session(...)`, `RendererSession.render_ids(...)`, `RendererSession.render_ids_np(...)` | `session_render_ids` | `tests/test_native_numpy.py::test_session_render_and_bridge_match_renderer`, parity rows |
| Session bridge | `RendererSession.bridge_to_next_turn(...)`, `RendererSession.bridge_to_next_turn_np(...)` | `session_bridge_to_next_turn` | `tests/test_native_numpy.py::test_session_render_and_bridge_match_renderer`, `test_session_numpy_bridge_match_renderer` |
| Repeatable session loop | `RendererSession.fork()` plus `bridge_to_next_turn(update=True)` | `session_bridge_loop` | `tests/test_native_numpy.py::test_session_fork_preserves_prompt_state`, benchmark parity precheck |
| Batched render | `Renderer.render_batch_ids(...)` | `render_batch_ids` | `tests/test_native_numpy.py::test_render_batch_ids_matches_single_calls` |
| Packed NumPy batch | `Renderer.render_batch_ids_np_packed(...)` | `render_batch_ids` native NumPy path | `tests/test_native_numpy.py::test_render_batch_ids_np_packed_matches_single_calls` |
| Single-prompt NumPy | `render_ids_np(...)`, `parse_response_np(...)`, `bridge_to_next_turn_np(...)` | native NumPy path, `render_ids_np_then_tolist` | `test_render_ids_np_matches_list_api`, `test_parse_response_np_borrows_uint32_completion`, `test_bridge_to_next_turn_np_matches_list_api` |
| Fast input shape | `Renderer.render_fast_ids(...)`, `Renderer.render_fast_ids_np(...)` | `render_fast_ids` | `tests/test_native_numpy.py::test_render_fast_ids_matches_dict_messages` |
| Dynamic text batching | `Tokenizer::encode_batch_no_special(...)`, `TokenPlanBuf`, Qwen3, Qwen3.5/Qwen3.6, DeepSeek V3, MiniMax M2, and GLM long no-tool `render_ids` dispatch | long-history `render_ids` and `render_fast_ids` rows | full parity, native-forced render tests, benchmark parity precheck |
| Template literal caches | family constructors store pre-tokenized literals | normal render and bridge rows across families | full parity and native-forced render/bridge tests |
| Prepared tool text cache | `ToolTextCache` in core family renderers | prepared-tools rows across supported families | full parity and native-forced render/bridge tests |
| Parse fast paths | borrowed stripped slices in Qwen3.5/Qwen3.6 and GLM, moved decoded content in Qwen3 | `parse_response` rows | full parity parse rows and native-forced roundtrip tests |

## PR Implementation Order

1. Benchmark artifact and progress support.
2. Baseline benchmark artifact from the current branch.
3. Prepared tools.
4. Session object for multiturn render and bridge.
5. Packed NumPy batch output.
6. Template constant token caches.
7. Dynamic text encode batching.
8. Optional fast input shape.
9. Parse response fast paths.

Each item should have:

- A parity test.
- A benchmark row or scenario that isolates it.
- A benchmark summary against the previous baseline.
- No broad Python-side rewrite unless the benchmark shows the API is worth it.

## Success Criteria

Runtime work is ready for the PR when:

- Full parity passes.
- Full native-forced Python test subset passes.
- Full benchmark artifacts exist for baseline and final commits.
- The PR description shows per-family and overall geomean speedup.
- Any regression over 5 percent is explained or fixed.
- SGLang and vLLM examples still show the simple list-based path.
- Optional fast paths are documented by example, not required for normal use.
