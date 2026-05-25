# Renderer config

How `renderers.configs` is shaped, and why.

## Discriminated union of typed configs

`renderers/configs.py` defines one pydantic model per renderer, joined into a
discriminated union on the `name` field:

```python
class _BaseRendererConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    preserve_all_thinking: bool = False
    preserve_thinking_between_tool_calls: bool = False

class Qwen3RendererConfig(_BaseRendererConfig):
    name: Literal["qwen3"] = "qwen3"
    enable_thinking: bool = True

class GLM5RendererConfig(_BaseRendererConfig):
    name: Literal["glm-5"] = "glm-5"
    enable_thinking: bool = True
    clear_thinking: bool = True

# … one per renderer

RendererConfig = Annotated[
    Union[AutoRendererConfig, DefaultRendererConfig, Qwen3RendererConfig, …],
    Field(discriminator="name"),
]
```

Three properties fall out of this shape:

1. **Strict per-variant validation.** `extra="forbid"` means
   `Qwen3RendererConfig(add_vision_id=True)` raises at construction — Qwen3 is
   text-only, so the field doesn't exist on its config. No separate allowlist
   has to track which kwargs each renderer accepts.
2. **Discriminator-driven kwarg narrowing for downstream consumers.** When
   prime-rl / verifiers hold a single field typed as `RendererConfig`,
   selecting `name="qwen3.5"` automatically exposes only Qwen3.5's
   template fields (`enable_thinking`, `add_vision_id`) — bogus combinations
   error at config-load with a clear pydantic message.
3. **Frozen configs.** `frozen=True` makes the config a value object. Renderers
   that need to resolve a default at construction (e.g. Qwen3.5's
   `enable_thinking=None` → auto-detected bool) call `cfg.model_copy(update=…)`
   to rebind, not mutate in place.

## `AutoRendererConfig` resolves via `MODEL_RENDERER_MAP`

`create_renderer(tokenizer)` (no config) is equivalent to
`create_renderer(tokenizer, AutoRendererConfig())`. `_resolve_auto` reads
`tokenizer.name_or_path`, looks it up in `MODEL_RENDERER_MAP`, and constructs
the matching typed config:

```python
def _resolve_auto(tokenizer, auto: AutoRendererConfig) -> Renderer:
    renderer_name = MODEL_RENDERER_MAP.get(tokenizer.name_or_path)
    if renderer_name is not None:
        cfg_cls = _config_class_for(renderer_name)
        return RENDERER_REGISTRY[renderer_name](
            tokenizer,
            cfg_cls(
                preserve_all_thinking=auto.preserve_all_thinking,
                preserve_thinking_between_tool_calls=auto.preserve_thinking_between_tool_calls,
            ),
        )
    # … VLM check, DefaultRenderer fallback
```

Auto carries only the shared `preserve_*` fields. Template kwargs require an
explicit renderer choice — the user has to write `Qwen35RendererConfig(...)`
to override `enable_thinking`, because template-dependent behaviour is
renderer-specific and we want that choice visible at the call site.

VLMs that miss the exact-match lookup fail loud. `DefaultRenderer` only knows
`apply_chat_template` + text tokens, so silently falling back for a VLM would
produce token streams the trainer can't reconstruct.

## `DefaultRenderer` accepts arbitrary Jinja kwargs

`DefaultRenderer` wraps `tokenizer.apply_chat_template` for any model that
doesn't have a hand-coded renderer. We can't enumerate the kwargs an unknown
Jinja template will honour, so `DefaultRendererConfig` opts into
`extra="allow"`:

```python
class DefaultRendererConfig(_BaseRendererConfig):
    model_config = ConfigDict(frozen=True, extra="allow")
    name: Literal["default"] = "default"
    tool_parser: str | None = None
    reasoning_parser: str | None = None
```

Unknown fields land in `model_extra` and `DefaultRenderer._apply` forwards
them verbatim to `apply_chat_template`. `tool_parser` / `reasoning_parser`
are typed because they configure DefaultRenderer's own parsing pipeline,
not the underlying template — they're listed in `_internal_fields` so the
parity-test discovery doesn't try to round-trip them through Jinja.

## `preserve_*` semantics: OR-composition with template kwargs

Several renderers carry a template-level toggle that gates historical
thinking (GLM-5 `clear_thinking`, Nemotron-3 `truncate_history_thinking`).
The renderer-agnostic `preserve_*` flags on `_BaseRendererConfig` compose
with those via OR — either flag saying "keep" wins:

```python
include_thinking = (
    msg_idx > last_user_index            # template default for the current cycle
    or preserve_thinking                 # preserve_* override said keep
    or not self.config.clear_thinking    # Jinja kwarg said keep
) and reasoning_content
```

The contract: **`preserve_*` are additive over template kwargs — they can only
ever extend retention, never override a template kwarg into a "drop"
decision.** Setting `preserve_all_thinking=True` always keeps thinking,
regardless of `clear_thinking=True`.

| `clear_thinking` | `preserve_all_thinking` | past thinking? |
|---|---|---|
| `True` (default — drop) | `False` (default) | dropped |
| `True` | `True` | kept (preserve_* added it back) |
| `False` (keep) | `False` | kept (template kwarg already says keep) |
| `False` | `True` | kept (both say keep) |

This contract lives in one place: `should_preserve_past_thinking` in
`base.py`. Each renderer that exposes a template-level toggle ORs the
helper's return value into its own "render thinking?" condition.

## `_internal_fields`: separating template kwargs from renderer state

Most typed-config fields mirror a Jinja chat-template kwarg one-to-one
(`enable_thinking`, `add_vision_id`, `clear_thinking`, …). A few don't:

- `image_cache_max` (Qwen3.5 / Qwen3.6 / Qwen3-VL / Kimi K2.5) bounds an
  in-memory image-processor cache. Renderer state, not a template kwarg.
- gpt-oss's `use_system_prompt` / `knowledge_cutoff` / `model_identity`
  control how the renderer builds the harmony `SystemContent` preamble.
  Typed config so users can set them, but no Jinja analogue exists.
- DeepSeek-V3's and Kimi K2's `enable_thinking` is renderer convention for
  the R1-distill family — the upstream Jinja template ignores it, so
  passing it to `apply_chat_template` is a no-op.

Each config class lists those in a `_internal_fields: ClassVar[frozenset[str]]`
override. `_BaseRendererConfig.template_field_names()` excludes them, and the
parity matrix in `tests/test_renderer_config_parity.py` only exercises the
remainder against `apply_chat_template`.

## Tradeoffs

- **Pydantic is a direct dep.** Already transitively present everywhere this
  package runs.
- **Discriminator key is the renderer name string.** Renaming a renderer is a
  breaking change for downstream configs. Already true of `MODEL_RENDERER_MAP`
  values; the typed shape doesn't make it worse.
- **Auto-resolution carries only `preserve_*`.** Template kwargs + auto means
  naming the renderer explicitly. Intentional — template kwargs depend on the
  renderer, so requiring an explicit choice keeps template-dependent behaviour
  visible at the call site.
