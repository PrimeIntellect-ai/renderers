# renderers

Programmable chat templates for LLM training and inference. A renderer turns a model's chat template into a Python object that can render messages → token ids, parse completion ids → structured assistant messages, and extend a multi-turn rollout without re-rendering model-sampled history.

Standalone on PyPI, and portable across training and inference stacks (transformers, vLLM, SGLang, Tinker). Initially developed for RL training with [verifiers](https://github.com/PrimeIntellect-ai/verifiers) and `prime-rl` at Prime Intellect.

## Install

```bash
uv add renderers
```

## At a glance

```python
from transformers import AutoTokenizer
from renderers import create_renderer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
r = create_renderer(tok, renderer="auto")           # → Qwen3Renderer

prompt_ids = r.render_ids(
    [{"role": "user", "content": "hi"}],
    add_generation_prompt=True,
)
# Feed prompt_ids to a Token-In, Token-Out endpoint.
# It returns completion_ids sampled by the model.

parsed = r.parse_response(completion_ids)
# ParsedResponse(content=..., reasoning_content=..., tool_calls=...)
```

For the next turn, extend the previous sampled stream instead of re-rendering history:

```python
next_prompt_ids = r.bridge_to_next_turn(
    previous_prompt_ids=prompt_ids,
    previous_completion_ids=completion_ids,
    new_messages=[{"role": "tool", "content": "..."}],
)
```

Hand-coded renderers ship for `qwen3`, `qwen3-vl`, `qwen3.5`, `qwen3.6`, `glm-5`, `glm-5.1`, `glm-4.5`, `minimax-m2`, `deepseek-v3`, `kimi-k2`, `kimi-k2.5`, `nemotron-3`, `gpt-oss`. Anything else falls back to `DefaultRenderer`, a generic `apply_chat_template` wrapper.

## API

```python
class Renderer(Protocol):
    def render(messages, *, tools=None, add_generation_prompt=False) -> RenderedTokens: ...
    def render_ids(messages, *, tools=None, add_generation_prompt=False) -> list[int]: ...
    def parse_response(token_ids) -> ParsedResponse: ...
    def get_stop_token_ids() -> list[int]: ...
    def bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages, *, tools=None) -> list[int] | None: ...
```

- `RenderedTokens` carries `token_ids` **and** `message_indices` — one entry per token attributing each to its source message (`-1` for structural scaffolding). Lets `build_training_sample` build a per-token loss mask in one render.
- `ParsedResponse` is `(content, reasoning_content, tool_calls)`. It scans token ids for special-token boundaries (e.g. id `151657` for `<tool_call>` on Qwen3) — a literal `"<tool_call>"` in user content tokenizes to ordinary text ids and never matches.
- Round-trip: rendering `[user, assistant(content, reasoning, tool_calls)]`, slicing the assistant completion, and feeding it through `parse_response` returns an equivalent structured message. Tested per-renderer in `tests/test_roundtrip.py`.

### `bridge_to_next_turn` (the core contract)

Given `(prev_prompt_ids, prev_completion_ids)` and new environment messages, return ids for the next turn's prompt such that the result starts with `prev_prompt_ids + prev_completion_ids` byte-for-byte and continues with the new messages plus the next assistant opener. If that cannot be proven safe, return `None` and the caller falls back to a full render.

Each hand-coded bridge:
1. Anchors at the previous turn's canonical close token. On clean stops it's already in `prev_completion_ids`. On truncation, the renderer synthesizes the close as non-loss prompt context.
2. Refuses assistant content in `new_messages` — re-rendering sampled tokens would replace them with canonical template bytes.
3. Renders only the new messages in the framing the model family expects.

`DefaultRenderer.bridge_to_next_turn` returns `None` unconditionally — the template's close is unknown, so the contract can't be proven.

### Picking a renderer

```python
r = create_renderer(tok, renderer="auto")
```

Auto-detect matches `tokenizer.name_or_path` against `MODEL_RENDERER_MAP` by **exact match**. Prefix matching is intentionally off — same architecture can ship different chat templates (base vs instruct, fine-tune renames). Fine-tunes must pass `renderer=<name>` explicitly; unknown names fall back to `DefaultRenderer`.

### Pools

```python
from renderers import create_renderer_pool

pool = create_renderer_pool("Qwen/Qwen3-8B", renderer="auto", size=16)
with pool.checkout() as r:
    ids = r.render_ids(messages)
```

Each slot owns its own tokenizer copy. Construction fans out across a thread pool so a 32-slot pool doesn't serially eat ~10–15s of `from_pretrained` calls at startup.

## Why use a renderer

For RL the trainer must see the exact token ids the sampler saw. The standard alternative — let the inference engine apply the chat template, parse tool calls, parse reasoning, and re-render full history every turn — silently breaks token identity. These are the failure modes a renderer's `bridge_to_next_turn` sidesteps by never re-rendering prior turns:

- **Boolean round-trip.** Engine emits `false`; client parses to Python `bool(False)`; `apply_chat_template` re-renders via `str(False)` → `"False"`. Capital F. Reproducible on Qwen3.5-35B-A3B + mini-swe-agent-plus at ~50% break rate per rollout.
- **BPE retokenization drift.** The same substring tokenizes differently depending on neighbouring bytes. `json` + `p` + `enderer` (3 tokens) vs `jsonp` + `enderer` (2 tokens) when whitespace shifts by one character. Every subsequent token is shifted from there on.
- **Tool-call XML drift.** The engine emits a no-arg call with a stylistic empty `</parameter>`; the Jinja re-render of the reconstructed dict drops it. Extension property broken at every such call.
- **Thinking stripped from non-latest assistants.** Some templates strip `<think>…</think>` blocks from prior assistant turns when re-rendering. The recorded stream has the thinking; the next prompt does not.
- **Max-seq-len truncation zeroing the anchor.** Client-side `max_seq_len` enforcement zeros `completion_ids` when `prompt_len > max_seq_len`. The bridge anchor is empty, falling back to full re-render — triggering every mode above.
- **Scaffold-level history rewriting.** Some agent scaffolds (e.g. opencode's `experimental_repairToolCall`) rewrite tool calls before sending them back as history. The next turn's prompt contains a tool call the model never emitted. *A renderer cannot fix this — the drift happens before rendering.*

Empirical delta on Qwen3.5-35B-A3B + mini-swe-agent-plus, step 0:

| client path                            | breaks | training samples from 64 rollouts |
| -------------------------------------- | ------ | --------------------------------- |
| `apply_chat_template` (full re-render) | 32     | 77                                |
| renderers `bridge_to_next_turn`        | 0      | 64                                |

Each break fragments a rollout into multiple training samples — every fragment re-encodes its prefix, inflating compute roughly linearly with the number of breaks.

## Compaction overrides

`create_renderer` and `create_renderer_pool` accept two constructor-only flags:

```python
preserve_all_thinking: bool = False
preserve_thinking_between_tool_calls: bool = False
```

Defaults preserve byte-identity with the model's chat template. Flipping a flag at construction restores `reasoning_content` the template would otherwise drop:

- `preserve_all_thinking=True` — every past assistant's reasoning is kept.
- `preserve_thinking_between_tool_calls=True` — reasoning is kept on assistants in the in-flight tool cycle (no-op for current renderers; reserved for future templates that drop it).

The canonical use case is **compaction**. Injecting a `user` turn like *"summarize the work so far"* puts every prior assistant in a "past cycle", so template-default rules drop their `reasoning_content` before the summarizer sees it. Build the renderer with `preserve_all_thinking=True` to keep reasoning visible end-to-end on those flows. Both flags only ever *add* tokens vs the template default.

## `DefaultRenderer`

Fallback for unsupported models. Wraps `apply_chat_template` and accepts `tool_parser` / `reasoning_parser` kwargs (vLLM convention). `bridge_to_next_turn` returns `None` because the template's close is unknown, so multi-turn rollouts fall back to full re-render. Implementing a hand-coded renderer is a few hundred lines of Python (`render_ids` + `parse_response` + `bridge_to_next_turn`) and is the only path that closes the failure modes above by construction.

## Roadmap

- **VLM support.** `ContentPart` is text-only today; `Qwen3VLRenderer` ships only because Qwen3-VL's text-only chat template differs from Qwen3's. Plan: add `ImagePart` / `VideoPart`, multimodal bridges, validate against a Qwen3-VL RL run.
- **Patched chat templates.** Some shipped templates re-tokenize history, normalize JSON, or auto-strip thinking — each breaks the extension property. Plan: a `use_patched` opt-in per renderer that renders the same surface form while avoiding known-bad patterns.

## Testing

```bash
uv sync --group dev
uv run pytest
```

Round-trip parity (render → parse → original) and token-level parity against `apply_chat_template` are tested per renderer. End-to-end validation runs against Reverse-Text, Wordle, OpenCode-Math, and RLM-SWE environments.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
