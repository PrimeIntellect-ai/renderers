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
r = create_renderer(tokenizer, AutoRendererConfig(thinking_retention="all"))
```

`AutoRendererConfig` carries only the shared `thinking_retention` flag. Template
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

## `thinking_retention`

Every variant carries one renderer-agnostic flag on `_BaseRendererConfig`,
an ascending scale whose floor is the chat template's own decision:

- `thinking_retention: Literal["template", "tool_cycle", "all"] = "template"`
  - `"template"` (default) — defer entirely to the chat template.
  - `"tool_cycle"` — additionally re-emit `reasoning_content` inside the
    in-flight tool cycle (the contiguous A-T-…-A block after the most
    recent `user` message, when it contains at least one `tool` response).
    A new user turn closes the block and drops its thinking.
  - `"all"` — additionally re-emit `reasoning_content` on every past
    assistant turn, even when the chat template would drop it.

The levels are nested: `"all"` ⊇ `"tool_cycle"` ⊇ `"template"`, and the
level is honoured end-to-end — `render()` and `bridge_to_next_turn` both
consult it, so multi-turn rollouts reproduce the template's history handling
faithfully by default. GLM-5's `clear_thinking` and Nemotron-3's
`truncate_history_thinking` are byte-equivalent template kwargs (`False` ≡
`"all"`) gating the same past thinking; `thinking_retention` composes with
them as:

| `clear_thinking` | `thinking_retention` | past thinking? |
|------------------|----------------------|----------------|
| `True` (default — drop) | `"template"` (default) | dropped |
| `True`           | `"all"`              | kept           |
| `False` (keep)   | `"template"`         | kept           |
| `False`          | `"all"`              | kept           |

`thinking_retention` can only extend retention, never force a drop — the
template is the floor. Because the kwarg and `thinking_retention` name the
same thing, explicitly setting a keep-history kwarg to `False` *and* a
non-`"all"` `thinking_retention` is contradictory and raises at config-load
(set `thinking_retention="all"` instead). The canonical use case is **compaction**: injecting
a `user` turn like *"summarize the work so far"* puts every prior assistant
in a past cycle, and `thinking_retention="all"` keeps reasoning visible
end-to-end.

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
thinking_retention = "all"
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
