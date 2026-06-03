# Renderer config

`renderers.RendererConfig` is the typed input to `create_renderer` and
`create_renderer_pool`. It pins the renderer choice and its template-control
kwargs at construction.

```python
from renderers import create_renderer, Qwen35RendererConfig

r = create_renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))
```

`RendererConfig` is a pydantic discriminated union (one variant per renderer,
dispatched on the `name` field). Selecting a variant exposes exactly the
fields that renderer's chat template honours; anything else raises a
`pydantic.ValidationError` at construction.

## Per-renderer configs

Each hand-coded renderer has a typed config class with the template kwargs
its Jinja chat template reads. For example:

| Renderer       | Config class             | Template fields                                                |
|----------------|--------------------------|----------------------------------------------------------------|
| Qwen3          | `Qwen3RendererConfig`    | `enable_thinking`                                              |
| Qwen3.5 / 3.6  | `Qwen35RendererConfig`   | `enable_thinking`, `add_vision_id`                             |
| Qwen3-VL       | `Qwen3VLRendererConfig`  | `add_vision_id`                                                |
| GLM-5 / 5.1    | `GLM5RendererConfig`     | `enable_thinking`, `clear_thinking`                            |
| GLM-4.5        | `GLM45RendererConfig`    | `enable_thinking`                                              |
| Nemotron-3     | `Nemotron3RendererConfig`| `enable_thinking`, `truncate_history_thinking`                 |
| Kimi K2.5      | `KimiK25RendererConfig`  | `thinking`                                                     |
| MiniMax-M2     | `MiniMaxM2RendererConfig`| `model_identity`                                               |
| Laguna-XS.2    | `LagunaXS2RendererConfig`| `enable_thinking`, `render_assistant_messages_raw`             |
| gpt-oss        | `GptOssRendererConfig`   | `reasoning_effort`, `conversation_start_date`                  |

Field names mirror the upstream Jinja variable names. Passing
`Qwen3RendererConfig(add_vision_id=True)` raises — Qwen3 is text-only, so
the field doesn't exist on its config. Use
`type(config).template_field_names()` to introspect the fields that mirror
chat-template kwargs (parity is verified against `apply_chat_template` in
`tests/test_renderer_config_parity.py`).

Configs are frozen. To override a field, construct a new instance or call
`config.model_copy(update={...})`.

## Auto-resolution

`create_renderer(tokenizer)` (no config) resolves the renderer from
`tokenizer.name_or_path` via `MODEL_RENDERER_MAP`:

```python
r = create_renderer(tokenizer)                                 # AutoRendererConfig() is the default
r = create_renderer(tokenizer, AutoRendererConfig(preserve_all_thinking=True))
```

`AutoRendererConfig` carries only the shared `preserve_*` flags. Template
kwargs depend on the renderer, so overriding them requires naming the
renderer explicitly:

```python
r = create_renderer(tokenizer, GLM5RendererConfig(clear_thinking=False))
```

Auto-resolution fails loudly for VLMs that miss the exact-match lookup —
`DefaultRenderer` only knows `apply_chat_template` + text tokens, so silently
falling back for a VLM would produce token streams the trainer can't
reconstruct. Text-only fine-tunes without a registered renderer fall back to
`DefaultRenderer` and log the choice at INFO.

## `preserve_*` flags

Every variant carries two renderer-agnostic flags on `_BaseRendererConfig`:

- `preserve_all_thinking: bool = False` — re-emit `reasoning_content` on
  every past assistant turn, even when the chat template would drop it.
- `preserve_thinking_between_tool_calls: bool = False` — re-emit
  `reasoning_content` only inside the in-flight tool cycle (the contiguous
  A-T-…-A block after the most recent `user` message, when it contains at
  least one `tool` response). A new user turn closes the block and drops
  its thinking.

These OR-compose with template-level toggles. GLM-5's `clear_thinking` and
Nemotron-3's `truncate_history_thinking` already gate past thinking; the
`preserve_*` flags add to that:

| `clear_thinking` | `preserve_all_thinking` | past thinking? |
|------------------|-------------------------|----------------|
| `True` (default — drop) | `False` (default) | dropped |
| `True`           | `True`                  | kept           |
| `False` (keep)   | `False`                 | kept           |
| `False`          | `True`                  | kept           |

`preserve_*` can only extend retention, never force a drop. The canonical
use case is **compaction**: injecting a `user` turn like *"summarize the work
so far"* puts every prior assistant in a past cycle, and
`preserve_all_thinking=True` keeps reasoning visible end-to-end.

## `DefaultRendererConfig` accepts arbitrary Jinja kwargs

`DefaultRenderer` wraps `tokenizer.apply_chat_template` for any model that
doesn't have a hand-coded renderer. Its config sets `extra="allow"`:

```python
from renderers import create_renderer, DefaultRendererConfig

r = create_renderer(
    tokenizer,
    DefaultRendererConfig(
        tool_parser="qwen3",                # registered in renderers.parsers
        reasoning_parser="think",
        enable_thinking=False,              # forwarded to apply_chat_template
        custom_jinja_kwarg=True,            # ditto
    ),
)
```

`tool_parser` and `reasoning_parser` are typed because they configure
`DefaultRenderer`'s own parsing pipeline. Every other field lands in
`model_extra` and `DefaultRenderer._apply` forwards `model_extra` verbatim
to `apply_chat_template`.

## Downstream integration

Downstream pydantic configs (`prime-rl` orchestrator, `verifiers`
`ClientConfig`) hold a single field typed as `RendererConfig`:

```python
from pydantic import BaseModel, Field
from renderers import AutoRendererConfig, RendererConfig

class ClientConfig(BaseModel):
    renderer: RendererConfig = Field(default_factory=AutoRendererConfig)
```

In TOML / YAML, the discriminator routes deserialization:

```toml
[client.renderer]
name = "qwen3.5"
enable_thinking = false
add_vision_id = true
preserve_all_thinking = true
```

Pydantic dispatches on `name = "qwen3.5"` to `Qwen35RendererConfig`. Bogus
combinations (e.g. `add_vision_id` under `name = "qwen3"`) raise at
config-load with a clear message naming the offending field and the variant
that rejected it.

To construct a config from a renderer name string (e.g. from a CLI flag):

```python
from renderers import config_from_name

cfg = config_from_name("glm-5")           # → GLM5RendererConfig() with defaults
cfg = config_from_name("auto")            # → None, the implicit "auto" form
```

## Renaming a renderer is a breaking change

The discriminator key is the renderer name string. Renaming `"qwen3.5"` to
something else would break any downstream config that references it by
name. Add new renderers; don't rename existing ones.
