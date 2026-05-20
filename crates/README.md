# `renderers` Rust port

Pure-Rust port of the `renderers` library, with a thin PyO3 wrapper so
existing Python callers can opt into the native path without code
changes.

## Workspace layout

| Crate            | Role                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `renderers-core` | Pure-Rust crate. Public `Renderer` / `MultimodalRenderer` traits, family implementations. |
| `renderers-py`   | PyO3 wrapper. Builds the `renderers._native` extension module via maturin.                |

The Rust crate is usable standalone (e.g. from an sglang-rs / vllm-rs
integration); the Python wrapper exists only to bridge into the
existing `renderers` package.

## Building the native extension

For development (editable install into the active venv):

```bash
maturin develop --manifest-path crates/renderers-py/Cargo.toml --release
```

This installs `renderers_native.<abi-tag>.so` into the venv's
`site-packages` so `import renderers_native` resolves. It's kept as a
top-level module (rather than `renderers._native`) so the maturin-built
wheel doesn't collide with the hatchling-built `renderers` wheel at
install time.

## Opting into the native path at runtime

The Python shims keep the pure-Python implementation as the default and
only route to the native module when `RENDERERS_NATIVE` selects the
family:

```bash
RENDERERS_NATIVE=qwen3 pytest tests/test_render_ids.py
RENDERERS_NATIVE=all   pytest tests/                       # everything ported
```

## Parity testing

Two complementary suites validate the port:

1. **`tests/test_render_ids.py` (and siblings)** — Python (or, when the
   env var routes, native) vs HuggingFace's `apply_chat_template`.
   Catches drift from the upstream reference. Run under the native
   path with `RENDERERS_NATIVE=qwen3 pytest tests/test_render_ids.py`.
2. **`tests/test_native_parity.py`** — Python vs native, holding the
   reference fixed. Catches drift between the two implementations even
   if HuggingFace changes its template. Cheaper because the HF call
   isn't on the path. Marker: `-m parity`.

The parity suite skips cleanly when the tokenizer.json isn't on disk
or the extension isn't built, so it's safe to import in CI without
gating on either.

Recognised values:

| `RENDERERS_NATIVE` | Behaviour                                                |
| ------------------ | -------------------------------------------------------- |
| unset / `0`        | Pure Python (default)                                    |
| `1` / `all`        | Route every supported family to the native module       |
| `qwen3`            | Route only Qwen3                                          |
| `qwen3,qwen35,...` | Route a comma-separated list of families                  |

If `RENDERERS_NATIVE` is set but the extension isn't installed, the
shim logs a one-shot info message and falls back to Python.

## Family coverage

| Family       | Status                                          |
| ------------ | ----------------------------------------------- |
| Qwen3        | ✅ ported (Phase 2)                              |
| Qwen3.5      | ✅ ported text-only (Phase 3) — multimodal Phase 5 |
| GLM 4.5 / 5  | planned (Phase 3)                                |
| DeepSeek V3  | planned (Phase 3)                                |
| Nemotron3    | planned (Phase 3)                                |
| Kimi K2      | planned (Phase 4)                                |
| Kimi K2.5    | planned (Phase 4 — text; multimodal Phase 5)    |
| MiniMax M2   | planned (Phase 4)                                |
| Qwen3.6      | planned (Phase 4)                                |
| Qwen3-VL     | planned (Phase 5 — multimodal incl. processor) |
| Qwen3.5 mm   | planned (Phase 5)                                |
| GPT-OSS      | planned (Phase 6 — via `openai-harmony` crate)  |
| Default      | planned (Phase 7 — via `minijinja`)             |

## Performance targets

Single-call latency (Qwen3.5, 1500-token prompt, 512-token completion):

| Phase                | Python (current) | Rust (target) | Speedup |
| -------------------- | ---------------: | ------------: | ------: |
| `render_ids`         |       0.5–1.0 ms |   0.15–0.3 ms |     3–5×|
| `parse_response`     |     0.05–0.15 ms | 0.05–0.15 ms¹ |    5–10×|
| `bridge_to_next_turn`|       0.3–0.6 ms |  0.05–0.15 ms |     4–6×|

¹ Speedup vs Python including FFI overhead, which is the actual gap;
absolute numbers depend on completion content shape.

Throughput on an 8-thread caller is expected to gain another ~5–8×
because every method releases the GIL (`py.allow_threads`) — the Python
pool model is obsolete in Rust.

## Crate-level invariants

- `#![forbid(unsafe_code)]` at the crate root of `renderers-core`.
- All hot-path scans use bounded `&[u32]` slices; no allocation in
  `find` / `find_from` / `find_any`.
- `RenderBuf` reserves capacity once based on `messages.len() * 256`.
- Special-token ids are resolved at renderer construction and cached
  on the struct.
- Tokenizer is held behind `Arc<...>` so a single instance serves any
  number of concurrent callers.
