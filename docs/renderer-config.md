# Renderer config

`renderers.RendererConfig` is the typed input to `create_renderer` and
`create_renderer_pool`. It pins the renderer choice and its config at
construction time.

```python
from renderers import create_renderer, Qwen35RendererConfig

r = create_renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))
r = create_renderer(tokenizer, chat_template_kwargs={"enable_thinking": False})
```

`RendererConfig` is a pydantic discriminated union, one variant per renderer,
dispatched on the `name` field. Most variants reject unknown fields at
construction. A field can either mirror a chat-template kwarg or configure a
renderer-only behavior such as parsing, image caching, or Harmony preamble
construction.

## Per-renderer configs

Use `type(config).template_field_names()` to inspect the fields that mirror
chat-template kwargs. Those fields are covered by parity tests against
`apply_chat_template` in `tests/test_renderer_config_parity.py`.

| Renderer | Config class | Template fields | Renderer-only fields |
| --- | --- | --- | --- |
| Qwen3 | `Qwen3RendererConfig` | `enable_thinking` | - |
| Qwen3.5 | `Qwen35RendererConfig` | `enable_thinking`, `add_vision_id` | `image_cache_max` |
| Qwen3.6 | `Qwen36RendererConfig` | `enable_thinking`, `add_vision_id`, `preserve_thinking` | `image_cache_max` |
| Qwen3-VL | `Qwen3VLRendererConfig` | `add_vision_id` | `image_cache_max` |
| GLM-5 / 5.1 | `GLM5RendererConfig` / `GLM51RendererConfig` | `enable_thinking`, `clear_thinking` | - |
| GLM-4.5 | `GLM45RendererConfig` | `enable_thinking` | - |
| gpt-oss | `GptOssRendererConfig` | `reasoning_effort`, `conversation_start_date` | `use_system_prompt`, `knowledge_cutoff`, `model_identity`, `auto_drop_analysis` |
| Kimi K2 | `KimiK2RendererConfig` | - | `enable_thinking` |
| Kimi K2.5 / 2.6 | `KimiK25RendererConfig` | `thinking` | `image_cache_max` |
| Laguna XS.2 | `LagunaXS2RendererConfig` | `enable_thinking`, `render_assistant_messages_raw` | - |
| Llama 3 | `Llama3RendererConfig` | `date_string`, `tools_in_user_message` | - |
| MiniMax M2 | `MiniMaxM2RendererConfig` | `model_identity` | - |
| Nemotron-3 Nano / Super | `Nemotron3RendererConfig` | `enable_thinking`, `truncate_history_thinking`, `low_effort` | - |
| Nemotron-3 Ultra | `Nemotron3UltraRendererConfig` | `enable_thinking`, `truncate_history_thinking`, `medium_effort` | - |
| DeepSeek V3 | `DeepSeekV3RendererConfig` | - | - |
| DeepSeek R1 | `DeepSeekR1RendererConfig` | - | - |

Configs are frozen value objects. To override a field, construct a new instance
or call `config.model_copy(update={...})`.

## Auto-resolution

`create_renderer(tokenizer)` resolves the renderer from `tokenizer.name_or_path`
via `MODEL_RENDERER_MAP`:

```python
from renderers import AutoRendererConfig, GLM5RendererConfig

r = create_renderer(tokenizer)
r = create_renderer(tokenizer, AutoRendererConfig(thinking_retention="all"))
r = create_renderer(tokenizer, GLM5RendererConfig(clear_thinking=False))
```

`AutoRendererConfig` carries only the shared `thinking_retention` override.
Callers that receive run-scoped chat-template kwargs can pass them separately:

```python
r = create_renderer(
    tokenizer,
    chat_template_kwargs={"enable_thinking": False},
)
pool = create_renderer_pool(
    "Qwen/Qwen3-8B",
    chat_template_kwargs={"enable_thinking": False},
)
```

Renderers resolves auto configs before applying `chat_template_kwargs`, so the
kwargs validate against the concrete renderer config. Unknown kwargs, or kwargs
that conflict with an explicit `thinking_retention`, fail at construction.

Auto-resolution fails loudly for VLMs without an exact registered renderer.
Text-only unknown models fall back to `DefaultRenderer`, unless
`AutoRendererConfig(thinking_retention=...)` was set. The default renderer
cannot implement selective bridge retention, so that combination raises.
`AutoRendererConfig` with `chat_template_kwargs` also raises for unknown models,
because renderers cannot validate those kwargs without a concrete renderer.
Use an explicit model-specific config, or `DefaultRendererConfig(...)` when you
intentionally want opaque `apply_chat_template` kwargs.

## `thinking_retention`

Every typed renderer config carries one shared optional bridge-policy override:

```python
thinking_retention: Literal["tool_cycle", "all"] | None = None
```

| Value | Meaning |
| --- | --- |
| `None` | Derive the effective bridge policy from the renderer's template knobs and defaults. |
| `"tool_cycle"` | Bridge within the current tool cycle; re-render when the extension opens a new user query. |
| `"all"` | Allow bridging across user-query boundaries when the bridge is otherwise structurally valid. |

`thinking_retention` affects `bridge_to_next_turn`, not full `render()`.
A full render always follows the Python chat-template implementation. Only real
template fields, such as `clear_thinking`, `preserve_thinking`, or
`truncate_history_thinking`, can change full-render historical thinking.

Internally, renderers resolve an `effective_thinking_retention` at construction:

| Internal policy | Bridge behavior |
| --- | --- |
| `"template"` | Decline bridging; caller falls back to a full re-render. |
| `"tool_cycle"` | Bridge unless `new_messages` introduces a user query. |
| `"all"` | Do not block bridging for thinking retention. |

`"template"` is not a public config value. Leave `thinking_retention` unset to
get template-derived behavior.

## Derived retention defaults

When `thinking_retention` is unset, each renderer derives its bridge policy from
the knobs its template actually exposes:

| Renderer | Derived policy |
| --- | --- |
| Qwen3 | `enable_thinking=False -> all`, else `tool_cycle` |
| Qwen3.5 | `enable_thinking=False -> all`, else `tool_cycle` |
| Qwen3.6 | `preserve_thinking=True -> all`; else `enable_thinking=False -> all`; else `tool_cycle` |
| GLM-5 / 5.1 | `clear_thinking=False -> all`; else `enable_thinking=False -> all`; else `tool_cycle` |
| GLM-4.5 | `enable_thinking=False -> all`, else `tool_cycle` |
| gpt-oss | `auto_drop_analysis=False -> all`, else `tool_cycle` |
| Kimi K2.5 / 2.6 | `thinking=False -> all`, else `tool_cycle` |
| Nemotron-3 | `truncate_history_thinking=False -> all`; else `enable_thinking=False -> all`; else `tool_cycle` |
| DeepSeek R1 | `template` |
| MiniMax M2 | `tool_cycle` |
| DeepSeek V3, Qwen3-VL, Kimi K2, Laguna XS.2, Llama 3 | `all` |

Config construction raises when an explicit template knob directly contradicts
an explicit generic bridge policy. For example:

```python
GLM5RendererConfig(clear_thinking=False, thinking_retention="tool_cycle")
# ValueError: clear_thinking=False implies thinking_retention="all"
```

Generation-only no-thinking knobs, such as `enable_thinking=False`, do not
conflict with an explicit conservative `thinking_retention="tool_cycle"`. They
only change the derived default when `thinking_retention` is unset.

## `DefaultRendererConfig`

`DefaultRenderer` wraps `tokenizer.apply_chat_template` for unsupported
text-only models. Its config sets `extra="allow"` so unknown fields are
forwarded as Jinja kwargs:

```python
from renderers import create_renderer, DefaultRendererConfig

r = create_renderer(
    tokenizer,
    DefaultRendererConfig(
        tool_parser="qwen3",
        reasoning_parser="think",
        enable_thinking=False,
        custom_jinja_kwarg=True,
    ),
)
```

`tool_parser` and `reasoning_parser` configure `DefaultRenderer` itself. Every
other extra field lands in `model_extra` and is forwarded to
`apply_chat_template`.

`DefaultRenderer` rejects explicit `thinking_retention` and the removed
`preserve_*` flags. Its bridge always returns `None`, because the template's
turn-close structure is opaque to the renderer.

## Downstream integration

Downstream pydantic configs can hold a single field typed as `RendererConfig`:

```python
from pydantic import BaseModel, Field
from renderers import AutoRendererConfig, RendererConfig


class ClientConfig(BaseModel):
    renderer: RendererConfig = Field(default_factory=AutoRendererConfig)
```

In TOML or YAML, the `name` discriminator selects the variant:

```toml
[client.renderer]
name = "qwen3.5"
enable_thinking = false
add_vision_id = true
thinking_retention = "all"
```

Bogus combinations, such as `add_vision_id` under `name = "qwen3"`, raise at
config load with a pydantic validation error.

To construct a config from a renderer name string:

```python
from renderers import config_from_name

cfg = config_from_name("glm-5")  # GLM5RendererConfig()
cfg = config_from_name("auto")  # None, the implicit auto form
```

## Renaming a renderer is a breaking change

The discriminator key is the renderer name string. Renaming `"qwen3.5"` to
something else would break downstream configs that reference it by name. Add
new renderers instead of renaming existing ones.
