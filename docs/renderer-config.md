# Typed renderer configs

**Status:** in-progress refactor of PR #60. Supersedes the `chat_template_kwargs: dict[str, Any]` shape introduced in that PR.

## Motivation

PR #60 introduced `chat_template_kwargs` as a free-form dict on `create_renderer` and `create_renderer_pool`. To validate it, each renderer class grew a `CHAT_TEMPLATE_KWARGS` frozenset listing which keys it accepts, and the factory grew a reserved-constructor-kwarg reject set. That works, but it has three problems Mika surfaced in his review:

1. **Two namespaces for the same data.** Every template kwarg is both a renderer constructor argument (typed) *and* an entry in `CHAT_TEMPLATE_KWARGS` (untyped frozenset). The two have to be kept in sync by hand.
2. **No discoverability.** `chat_template_kwargs={"enable_thinking": False}` gives no IDE help, no autocomplete, no schema. The actual list of accepted keys is buried per-renderer.
3. **No interop with consumer configs.** prime-rl and verifiers want pydantic-typed renderer settings so users can put them in TOML/YAML and get strict validation at config-load time, and so picking a renderer in a discriminated union automatically narrows to the kwargs that renderer supports.

This document specifies the typed-config replacement.

## Design

### One pydantic config per renderer, unified by a discriminated union

```python
# renderers/configs.py

class _BaseRendererConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    preserve_all_thinking: bool = False
    preserve_thinking_between_tool_calls: bool = False


class AutoRendererConfig(_BaseRendererConfig):
    name: Literal["auto"] = "auto"
    # No template kwargs. Auto resolves to a concrete renderer via
    # MODEL_RENDERER_MAP at create_renderer() time, carrying the
    # preserve_* fields with it.


class DefaultRendererConfig(_BaseRendererConfig):
    model_config = {"frozen": True, "extra": "allow"}  # accept arbitrary Jinja kwargs
    name: Literal["default"] = "default"
    tool_parser: str | None = None
    reasoning_parser: str | None = None


class Qwen3RendererConfig(_BaseRendererConfig):
    name: Literal["qwen3"] = "qwen3"
    enable_thinking: bool = True


class Qwen35RendererConfig(_BaseRendererConfig):
    name: Literal["qwen3.5"] = "qwen3.5"
    enable_thinking: bool | None = None
    add_vision_id: bool = False
    image_cache_max: int = 256


class GLM5RendererConfig(_BaseRendererConfig):
    name: Literal["glm-5"] = "glm-5"
    enable_thinking: bool = True
    clear_thinking: bool = True


class GptOssRendererConfig(_BaseRendererConfig):
    name: Literal["gpt-oss"] = "gpt-oss"
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    conversation_start_date: str | None = None
    use_system_prompt: bool = True
    knowledge_cutoff: str | None = None
    model_identity: str | None = None


# ... one per renderer (Qwen3VL, Qwen36, GLM51, GLM45, KimiK2, KimiK25,
#     LagunaXS2, MiniMaxM2, Nemotron3, DeepSeekV3)

RendererConfig = Annotated[
    Union[
        AutoRendererConfig,
        DefaultRendererConfig,
        Qwen3RendererConfig,
        Qwen35RendererConfig,
        # ... every variant
    ],
    Field(discriminator="name"),
]
```

### No public "Jinja-kwargs view" helper

The first draft of this design exported a `template_kwargs(config)` helper that returned the dict-shaped subset of a typed config — the assumption was that downstream consumers (prime-rl, vLLM) needed to forward those kwargs to a *second* Jinja consumer alongside the renderer. That assumption is wrong for this codebase: prime-rl and verifiers use vLLM via the token-in endpoint, which never applies a chat template. The renderer is the sole consumer of these kwargs end-to-end.

Two internal sites still care about "the Jinja-applicable subset", and they reach for it via the `Config.template_field_names()` classmethod on the base — but it's not exported as a public helper:

- **`DefaultRenderer._apply`** calls `tokenizer.apply_chat_template(messages, **kwargs)` reading `config.model_extra` inline.
- **Parity tests** in `tests/test_chat_template_kwargs_parity.py` enumerate template fields via `Config.template_field_names()` to discover the matrix cells that must agree with `apply_chat_template`.

`template_field_names()` defaults to "every non-base field except `name` and the `_internal_fields` ClassVar override," so renderers with renderer-internal-only fields (GptOss's `use_system_prompt` / `knowledge_cutoff` / `model_identity`, KimiK2 / DeepSeekV3's `enable_thinking` no-op, the per-renderer `image_cache_max`) declare those explicitly to keep parity discovery accurate.

### Factory functions take the typed config

```python
def create_renderer(tokenizer, config: RendererConfig | None = None) -> Renderer:
    if config is None:
        config = AutoRendererConfig()
    if isinstance(config, AutoRendererConfig):
        config = _resolve_auto(tokenizer, config)
    cls = RENDERER_REGISTRY[config.name]
    return cls(tokenizer, config)


def create_renderer_pool(
    tokenizer_name_or_path: str,
    config: RendererConfig | None = None,
    *,
    size: int = 16,
) -> RendererPool:
    ...
```

`_resolve_auto` reads `tokenizer.name_or_path`, looks it up in `MODEL_RENDERER_MAP`, and constructs the matching config class with the auto config's `preserve_*` fields carried over. If no match and the model has a vision config, fail loud (same as today). Otherwise fall back to `DefaultRendererConfig`.

### Renderer classes store the typed config

```python
class Qwen3Renderer:
    def __init__(self, tokenizer, config: Qwen3RendererConfig):
        self.config = config
        self._im_start = self._token_id("<|im_start|>")
        ...

    # Reads go through self.config:
    def render(self, ...):
        if self.config.enable_thinking:
            ...
```

Field shadowing (`self._enable_thinking = enable_thinking`) goes away — every renderer reads `self.config.<field>` directly. The config is frozen, so this is safe.

**Exception:** runtime-injected dependencies (`processor` for Qwen3VL/Qwen3.5/KimiK25) stay as separate constructor kwargs, since they're not serializable and don't belong in a config:

```python
class Qwen3VLRenderer:
    def __init__(self, tokenizer, config: Qwen3VLRendererConfig, *, processor=None):
        self.config = config
        self._processor = processor
        ...
```

### `preserve_*` semantics: OR-composition with template kwargs

The current code in GLM5 (`renderers/glm5.py:481`) and Nemotron3 (`renderers/nemotron3.py:656`) composes `preserve_all_thinking` / `preserve_thinking_between_tool_calls` with the template kwarg (`clear_thinking` / `truncate_history_thinking`) via OR:

```python
include_thinking = (
    msg_idx > last_user_index            # template default for the current cycle
    or preserve_thinking                 # preserve_* override said keep
    or not self.config.clear_thinking    # Jinja kwarg said keep
) and reasoning_content
```

The contract: **`preserve_*` are additive over template kwargs — they can only ever extend retention, never override the kwarg into a "drop" decision.** Setting `preserve_all_thinking=True` always keeps thinking, regardless of `clear_thinking=True`.

| `clear_thinking` | `preserve_all_thinking` | past thinking? |
|---|---|---|
| `True` (default — drop) | `False` (default) | dropped |
| `True` | `True` | kept (preserve_* added it back) |
| `False` (keep) | `False` | kept (template kwarg already says keep) |
| `False` | `True` | kept (both say keep) |

This contract lives in one place: the `should_preserve_past_thinking` helper in `base.py:1660`. The typed-config refactor doesn't change it. It only moves the inputs from `__init__` parameters to `self.config.<field>`. The `_BaseRendererConfig` docstring documents the OR contract.

### What gets deleted

From the current PR #60:
- `CHAT_TEMPLATE_KWARGS = frozenset(...)` declarations on every renderer (12 sites).
- `_RENDERER_CONSTRUCTOR_KWARGS` set in `base.py`.
- `_reject_renderer_constructor_kwargs` helper.
- `_model_renderer_chat_template_kwargs` helper.
- Per-renderer `self._enable_thinking = enable_thinking`-style field shadowing (12 sites).
- The "reject if a chat_template_kwarg overlaps with a renderer constructor kwarg" guard.

### What gets added

- `renderers/configs.py` — ~13 small pydantic config classes (typically 4–8 lines each plus docstrings), the `AutoRendererConfig` + `DefaultRendererConfig` special cases, and the discriminated-union alias.
- One direct dependency on `pydantic>=2` in `pyproject.toml`. Already transitively present via `openai-harmony` and `transformers`.

Net LOC: roughly neutral. The structural win is that **the type system replaces all the runtime allowlist machinery** — invalid combinations can't be constructed, and there's no separate frozenset to drift from the field list.

## Public surface — before and after

### Before

```python
from renderers import create_renderer

renderer = create_renderer(
    tokenizer,
    renderer="qwen3.5",
    chat_template_kwargs={"enable_thinking": False, "add_vision_id": True},
    preserve_all_thinking=True,
)
```

### After

```python
from renderers import create_renderer, Qwen35RendererConfig

renderer = create_renderer(
    tokenizer,
    Qwen35RendererConfig(
        enable_thinking=False,
        add_vision_id=True,
        preserve_all_thinking=True,
    ),
)
```

Or, in a downstream pydantic config (prime-rl orchestrator TOML):

```toml
[orchestrator.student.renderer.settings]
name = "qwen3.5"
enable_thinking = false
add_vision_id = true
preserve_all_thinking = true
```

Pydantic dispatches on `name="qwen3.5"` to `Qwen35RendererConfig`. Bogus keys (e.g. `add_vision_id` under `name="qwen3"`) error at config-load with a clear message.

## Companion PRs

### prime-rl (#2605)

Replace `RendererConfig` in `packages/prime-rl-configs/.../shared.py` with composition over `renderers.RendererConfig`:

```python
from renderers import RendererConfig as RendererSettings, AutoRendererConfig

class RendererConfig(BaseConfig):
    settings: RendererSettings = Field(default_factory=AutoRendererConfig)
    pool_size: int | None = Field(None, ge=1)
```

In `orchestrator.py`:
- `setup_student_inference_pool` passes `config.renderer.settings` directly to `create_renderer(tokenizer, config.renderer.settings)`.
- The `validate_renderer_chat_template_kwargs` cross-env consistency validator goes away. There's only one place to set template kwargs now: the typed renderer config. Nothing leaks server-side because vLLM is invoked via the token-in endpoint and never applies a chat template.
- The `_chat_template_kwargs_from_extra_body` helper introduced in #2605 also goes away. The typed config is the source; `sampling.extra_body.chat_template_kwargs` stops being a config slot for renderer behaviour.

### verifiers (#1447)

Add a typed `renderer_config: RendererConfig | None` field to `ClientConfig` in `verifiers/types.py`. The existing flat fields (`renderer`, `tool_parser`, `reasoning_parser`, `preserve_*`) either get marked deprecated with a compatibility shim or removed outright (decided per-PR).

In `verifiers/clients/renderer_client.py`, `_get_renderer_or_pool` reads `self._config.renderer_config` and passes it through to `create_renderer_pool`. Cache key becomes a hash of the typed config (`model_dump_json()`). The `_pop_chat_template_kwargs` / `_freeze_json_like` helpers introduced in #1447 go away — sampling-side `chat_template_kwargs` extraction is no longer the renderer config path.

For vf-eval's argparse CLI: accept the typed config via `--renderer-config '<json>'` or rely on the existing YAML/TOML client-config file. Pure-CLI users lose flat-flag access in exchange for typed correctness.

## Tradeoffs

- **Pydantic becomes a direct dep on renderers.** It's already transitively present everywhere this package runs.
- **Discriminator key is the renderer name string.** Renaming a renderer is a breaking change for downstream configs. Already true of the `renderer=<name>` argument today; the typed shape doesn't make it worse.
- **Auto-resolution carries `preserve_*` only.** If you want template kwargs *and* auto-resolution, you have to name the renderer explicitly. This is intentional — template kwargs depend on the renderer, so requiring an explicit choice makes template-dependent behavior visible.
- **vf-eval pure-argparse users lose flat flags.** JSON-blob or config-file workaround is acceptable for the interim; long-term vf-eval should adopt dotted-path nested-pydantic CLI parsing.
- **Companion PRs need rebase.** Already accepted.

## Migration ordering

1. Land the typed-config refactor in `renderers` (this PR).
2. Update `prime-rl` companion (#2605) to consume the typed shape.
3. Update `verifiers` companion (#1447) to add `renderer_config` on `ClientConfig`.

The renderer-package PR can land first; the companion PRs are isolated and don't block each other.
