"""Typed renderer configs — one pydantic model per renderer, unified by a
discriminated union (``RendererConfig``).

Each renderer accepts its own typed config; bad combinations (e.g.
``add_vision_id`` under ``name="qwen3"``) fail at config-load time with a
pydantic ``ValidationError`` rather than at runtime via an allowlist
check. The shared ``thinking_retention`` flag is optional: ``None`` means
"derive bridge policy from this renderer's chat-template knobs"; an
explicit value is a bridge-policy override.

``AutoRendererConfig`` is a placeholder variant: ``create_renderer``
resolves it via ``MODEL_RENDERER_MAP`` and constructs the matching
typed config with the auto config's ``thinking_retention`` field carried
over when one was explicitly supplied.

``DefaultRendererConfig`` uses ``extra="allow"`` to accept arbitrary
Jinja kwargs as ``model_extra`` — ``DefaultRenderer`` doesn't know which
keys its tokenizer's template will honour, so it can't enumerate them.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import ConfigDict, Field, model_validator
from pydantic_config import BaseConfig


def _reject_thinking_retention_conflict(
    config: BaseConfig,
    kwarg_name: str,
    *,
    true_implies: "ResolvedThinkingRetention",
    false_implies: "ResolvedThinkingRetention",
) -> None:
    """Raise if explicit template and renderer retention knobs disagree."""
    fields_set = config.__pydantic_fields_set__
    requested = getattr(config, "thinking_retention", None)
    if kwarg_name in fields_set and requested is not None:
        implied = (
            false_implies if getattr(config, kwarg_name) is False else true_implies
        )
        if requested == implied:
            return
        raise ValueError(
            f"{kwarg_name}={getattr(config, kwarg_name)!r} implies "
            f"thinking_retention={implied!r}, which conflicts with explicit "
            f"thinking_retention={requested!r}."
        )


ThinkingRetention = Literal["tool_cycle", "all"]
"""User-facing historical thinking/analysis retention override."""

ResolvedThinkingRetention = Literal["template", "tool_cycle", "all"]
"""Internal bridge policy after template kwargs have been resolved."""


class BaseRendererConfig(BaseConfig):
    """Shared fields and config for every renderer config variant.

    Inherits from ``pydantic_config.BaseConfig`` so the typed-config
    surface stays uniform with prime-rl / verifiers config bases. The
    BaseConfig contract includes ``extra="forbid"`` (preserved here);
    this class adds ``frozen=True`` so configs are hashable value
    objects.

    ``thinking_retention`` is an optional renderer-level retention override.
    Leave it ``None`` to derive the effective policy from the renderer's own
    chat-template knobs. Set it explicitly to request retention beyond the
    template default; renderers fail loudly when an explicit template knob says
    the opposite thing.
    """

    model_config = ConfigDict(frozen=True)

    thinking_retention: ThinkingRetention | None = None
    """Explicit retention override, or ``None`` to derive from template knobs:

    - ``None`` — derive the effective bridge policy from this renderer's
      chat-template knobs while keeping full renders template-faithful.
    - ``"tool_cycle"`` — bridge within the current tool cycle; re-render when
      a new user query arrives.
    - ``"all"`` — allow bridges across user-query boundaries.

    This does not change full ``render()`` output; full renders stay faithful
    to the Python chat-template implementation and its explicit template
    kwargs."""

    # Fields that are renderer-internal — not forwarded to (or mirrored
    # by) ``apply_chat_template``. Override in subclasses that hold
    # non-template config (e.g. ``image_cache_max``, GptOss's
    # ``use_system_prompt`` / ``knowledge_cutoff`` / ``model_identity``,
    # or fields that exist as renderer conventions without a Jinja
    # analogue like DeepSeek V3 / Kimi K2 ``enable_thinking``).
    #
    # Used by parity tests to compute the field subset that, when
    # changed, must produce token streams matching
    # ``apply_chat_template`` — see :meth:`template_field_names`. The
    # renderer is the only end-to-end consumer of these fields, so this
    # is a renderer-side bookkeeping concern rather than a public API.
    _internal_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def template_field_names(cls) -> frozenset[str]:
        """Subset of fields that mirror Jinja chat-template kwargs.

        Default: every non-base field except ``name`` and any field
        listed in ``_internal_fields``. Used by the parity test matrix
        (``tests/test_renderer_config_parity.py``) to discover the
        cells that must agree with ``apply_chat_template``.
        """
        base = frozenset(BaseRendererConfig.model_fields)
        return frozenset(cls.model_fields) - base - {"name"} - cls._internal_fields


class AutoRendererConfig(BaseRendererConfig):
    """Resolve the renderer from ``tokenizer.name_or_path`` at construction
    time via ``MODEL_RENDERER_MAP``. Carries only the shared
    ``thinking_retention`` field when explicitly set; template kwargs require
    an explicit renderer choice so template-dependent behaviour stays visible
    at the call site."""

    name: Literal["auto"] = "auto"


class DefaultRendererConfig(BaseRendererConfig):
    """Config for ``DefaultRenderer`` — the fallback wrapping
    ``tokenizer.apply_chat_template``. Accepts arbitrary extra fields
    via ``extra="allow"`` because the underlying Jinja template's kwargs
    are unknown to us. ``DefaultRenderer`` forwards ``model_extra`` to
    ``apply_chat_template`` verbatim.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    name: Literal["default"] = "default"

    tool_parser: str | None = None
    """Name of a tool parser registered in ``renderers.parsers`` (e.g.
    ``"qwen3"``, ``"glm"``). Consumed only by ``DefaultRenderer``."""

    reasoning_parser: str | None = None
    """Name of a reasoning parser registered in ``renderers.parsers``
    (e.g. ``"think"``). Consumed only by ``DefaultRenderer``."""

    # tool_parser / reasoning_parser are renderer-internal — they configure
    # DefaultRenderer's parsing pipeline, not the underlying Jinja
    # template. Jinja kwargs live in ``model_extra`` (extra="allow").
    _internal_fields = frozenset({"tool_parser", "reasoning_parser"})

    @model_validator(mode="after")
    def _reject_legacy_preserve_flags(self):
        # ``extra="allow"`` would otherwise swallow the removed ``preserve_*``
        # bools into ``model_extra`` and forward them to apply_chat_template,
        # silently dropping the user's intent (DefaultRenderer can't
        # selectively re-emit reasoning_content). Reject them like every other
        # config's ``extra="forbid"`` does, pointing at the replacement.
        legacy = {
            "preserve_all_thinking",
            "preserve_thinking_between_tool_calls",
        } & set(self.model_extra or {})
        if legacy:
            raise ValueError(
                f"{sorted(legacy)} were replaced by thinking_retention. "
                "DefaultRenderer falls back to apply_chat_template and can't "
                "selectively re-emit reasoning_content — use thinking_retention "
                "on a model-specific renderer."
            )
        return self


class Qwen3RendererConfig(BaseRendererConfig):
    """Qwen3 (text-only) renderer config."""

    name: Literal["qwen3"] = "qwen3"

    enable_thinking: bool = True
    """When ``True``, the generation prompt includes ``<think>`` so the
    model continues into a thinking block. Mirrors the chat template's
    ``enable_thinking`` kwarg."""


class Qwen35RendererConfig(BaseRendererConfig):
    """Qwen3.5 renderer config."""

    name: Literal["qwen3.5"] = "qwen3.5"

    enable_thinking: bool | None = None
    """When ``True``, the generation prompt includes ``<think>``. ``None``
    auto-detects from the tokenizer's chat-template default — Instruct
    checkpoints default off, Thinking checkpoints default on. Mirrors
    the chat template's ``enable_thinking`` kwarg."""

    add_vision_id: bool = False
    """When ``True``, prefix each ``<|vision_start|>`` placeholder with
    ``"Picture N: "`` / ``"Video N: "`` where N is a 1-indexed counter
    running across the entire conversation. Mirrors the chat template's
    ``add_vision_id`` toggle."""

    image_cache_max: int = 256
    """FIFO bound on the per-renderer image processor cache. Renderer-
    internal — not a Jinja chat-template kwarg."""

    _internal_fields = frozenset({"image_cache_max"})


class Qwen36RendererConfig(BaseRendererConfig):
    """Qwen3.6 renderer config. Inherits Qwen3.5's template surface."""

    name: Literal["qwen3.6"] = "qwen3.6"

    enable_thinking: bool | None = None
    """See :class:`Qwen35RendererConfig.enable_thinking`."""

    add_vision_id: bool = False
    """See :class:`Qwen35RendererConfig.add_vision_id`."""

    preserve_thinking: bool = False
    """When ``True``, keep historical ``<think>`` blocks even before the
    last real user query. Mirrors the Qwen3.6 chat template's native
    ``preserve_thinking`` kwarg."""

    image_cache_max: int = 256
    """See :class:`Qwen35RendererConfig.image_cache_max`."""

    _internal_fields = frozenset({"image_cache_max"})

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        _reject_thinking_retention_conflict(
            self,
            "preserve_thinking",
            true_implies="all",
            false_implies="tool_cycle",
        )
        return self


class Qwen3VLRendererConfig(BaseRendererConfig):
    """Qwen3-VL renderer config."""

    name: Literal["qwen3-vl"] = "qwen3-vl"

    add_vision_id: bool = False
    """See :class:`Qwen35RendererConfig.add_vision_id`."""

    image_cache_max: int = 256
    """See :class:`Qwen35RendererConfig.image_cache_max`."""

    _internal_fields = frozenset({"image_cache_max"})


class GLM5RendererConfig(BaseRendererConfig):
    """GLM-5 renderer config."""

    name: Literal["glm-5"] = "glm-5"

    enable_thinking: bool = True
    """When ``True``, the generation prompt includes ``<think>``. Mirrors
    the chat template's ``enable_thinking`` kwarg."""

    clear_thinking: bool = True
    """When ``False``, the renderer keeps ``<think>{reasoning}</think>``
    on past-cycle assistant turns instead of dropping them. Mirrors the
    chat template's ``clear_thinking`` toggle and resolves bridge policy
    to ``"all"``."""

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        _reject_thinking_retention_conflict(
            self,
            "clear_thinking",
            true_implies="tool_cycle",
            false_implies="all",
        )
        return self


class GLM51RendererConfig(BaseRendererConfig):
    """GLM-5.1 renderer config — same template surface as GLM-5, distinct
    discriminator so the registry can route to ``GLM51Renderer``."""

    name: Literal["glm-5.1"] = "glm-5.1"

    enable_thinking: bool = True
    """See :class:`GLM5RendererConfig.enable_thinking`."""

    clear_thinking: bool = True
    """See :class:`GLM5RendererConfig.clear_thinking`."""

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        _reject_thinking_retention_conflict(
            self,
            "clear_thinking",
            true_implies="tool_cycle",
            false_implies="all",
        )
        return self


class GLM45RendererConfig(BaseRendererConfig):
    """GLM-4.5 Air renderer config."""

    name: Literal["glm-4.5"] = "glm-4.5"

    enable_thinking: bool = True
    """When ``True``, the generation prompt includes ``<think>``. Mirrors
    the chat template's ``enable_thinking`` kwarg."""


class GptOssRendererConfig(BaseRendererConfig):
    """OpenAI gpt-oss (harmony) renderer config.

    Several fields here are renderer-internal: ``use_system_prompt``,
    ``knowledge_cutoff``, and ``model_identity`` control how the renderer
    builds the harmony ``SystemContent`` preamble and don't have direct
    Jinja-kwarg analogues. They're typed config rather than Jinja kwargs
    because users still want to set them — the distinction only matters
    for downstream tooling that synthesises a Jinja-kwargs view (none
    today, since vLLM is invoked via the token-in endpoint).
    """

    name: Literal["gpt-oss"] = "gpt-oss"

    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    """Harmony reasoning-effort tag. Mirrors the ``apply_chat_template``
    ``reasoning_effort`` kwarg."""

    conversation_start_date: str | None = None
    """ISO date string for the harmony preamble. ``None`` defers to
    today's date at render time."""

    use_system_prompt: bool = True
    """Prepend the canonical harmony ``SystemContent`` preamble. Matches
    HF's ``apply_chat_template`` behaviour."""

    knowledge_cutoff: str | None = None
    """Override the model's knowledge-cutoff string in the preamble.
    ``None`` uses harmony's built-in default."""

    model_identity: str | None = None
    """Override the model-identity line in the preamble. ``None`` uses
    harmony's built-in default."""

    auto_drop_analysis: bool = True
    """Harmony ``RenderConversationConfig.auto_drop_analysis`` behaviour.
    ``True`` keeps live tool-cycle analysis but drops stale analysis from
    history; ``False`` keeps analysis in all history."""

    _internal_fields = frozenset(
        {
            "use_system_prompt",
            "knowledge_cutoff",
            "model_identity",
            "auto_drop_analysis",
        }
    )

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        _reject_thinking_retention_conflict(
            self,
            "auto_drop_analysis",
            true_implies="tool_cycle",
            false_implies="all",
        )
        return self


class KimiK2RendererConfig(BaseRendererConfig):
    """Kimi K2 renderer config.

    ``enable_thinking`` is renderer-internal here — Kimi K2's chat
    template doesn't reference any thinking variable, so it's a no-op
    against ``apply_chat_template`` parity. The field is kept for
    protocol uniformity with the rest of the renderer family.
    """

    name: Literal["kimi-k2"] = "kimi-k2"

    enable_thinking: bool = True
    """No-op for Kimi K2 (template doesn't gate on it). Stored for
    introspection / cross-renderer uniformity."""

    _internal_fields = frozenset({"enable_thinking"})


class KimiK25RendererConfig(BaseRendererConfig):
    """Kimi K2.5 renderer config."""

    name: Literal["kimi-k2.5"] = "kimi-k2.5"

    thinking: bool = True
    """When ``True``, the generation prompt prefills ``<think>``; when
    ``False`` it prefills ``<think></think>``. The kwarg is named
    ``thinking`` (not ``enable_thinking``) to match the upstream chat
    template's native variable name."""

    image_cache_max: int = 256
    """See :class:`Qwen35RendererConfig.image_cache_max`."""

    _internal_fields = frozenset({"image_cache_max"})


class LagunaXS2RendererConfig(BaseRendererConfig):
    """Laguna XS.2 renderer config."""

    name: Literal["laguna-xs.2"] = "laguna-xs.2"

    enable_thinking: bool = False
    """When ``True``, the generation prompt includes ``<think>``. Mirrors
    the chat template's ``enable_thinking`` kwarg. Default ``False``
    matches the upstream Jinja default for Laguna XS.2."""

    render_assistant_messages_raw: bool = False
    """When ``True``, assistant messages render as a passthrough: the
    content bytes are emitted verbatim (no reasoning extraction, no
    tool-call XML synthesis), and the ``<think>``/``</think>`` prefix
    and ``</assistant>`` suffix are only added when missing. Mirrors the
    chat template's ``render_assistant_messages_raw`` gate."""


class Llama3RendererConfig(BaseRendererConfig):
    """Llama-3.x Instruct renderer config.

    Llama-3 ships no reasoning channel, so the base ``thinking_retention``
    flag is a no-op: there's never any past-assistant thinking to retain
    or drop, so any level leaves the token stream unchanged (same contract
    as Kimi-K2 / Qwen3-VL). Both fields below mirror real
    ``apply_chat_template``
    kwargs.
    """

    name: Literal["llama-3"] = "llama-3"

    date_string: str = "26 Jul 2024"
    """``Today Date`` value injected into the system preamble. Pinned to
    the chat template's ``strftime`` fallback by default so output stays
    deterministic; override per instance for production runs that want
    today's date. Mirrors the chat template's ``date_string`` kwarg."""

    tools_in_user_message: bool = True
    """When ``True`` (default), tool descriptions + JSON signatures inject
    into the first user message; ``False`` routes them into the system
    block instead. Mirrors the chat template's ``tools_in_user_message``
    kwarg."""


class MiniMaxM2RendererConfig(BaseRendererConfig):
    """MiniMax M2 / M2.5 renderer config."""

    name: Literal["minimax-m2"] = "minimax-m2"

    model_identity: str = "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax."
    """Fallback persona used when no system message is supplied. Mirrors
    the chat template's ``model_identity`` Jinja variable."""


class Nemotron3RendererConfig(BaseRendererConfig):
    """Nemotron-3 **Nano / Super** renderer config.

    Nano and Super share one chat-template variant; the renderer routes both
    through :class:`renderers.nemotron3.Nemotron3Renderer`. The Ultra variant
    has its own template (different reasoning-block glue) and config —
    :class:`Nemotron3UltraRendererConfig` — and is reached via the
    ``nemotron-3-ultra`` discriminator.
    """

    name: Literal["nemotron-3"] = "nemotron-3"

    enable_thinking: bool = True
    """When ``True``, the generation prompt includes ``<think>``. Mirrors
    the chat template's ``enable_thinking`` kwarg."""

    truncate_history_thinking: bool = True
    """When ``False``, keep ``<think>{reasoning}</think>`` on past-cycle
    assistant turns instead of dropping them. Mirrors the chat
    template's ``truncate_history_thinking`` toggle and resolves bridge
    policy to ``"all"``."""

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        _reject_thinking_retention_conflict(
            self,
            "truncate_history_thinking",
            true_implies="tool_cycle",
            false_implies="all",
        )
        return self

    low_effort: bool = False
    """When ``True``, append ``\\n\\n{reasoning effort: low}`` to the last user
    message, nudging the model toward shorter reasoning. Mirrors the **Super**
    chat template's ``low_effort`` kwarg. A no-op on **Nano** (its template
    doesn't define it) — exactly as ``apply_chat_template`` ignores an undefined
    template variable; the renderer distinguishes the two by model name (see
    ``renderers.nemotron3._is_super``)."""


class Nemotron3UltraRendererConfig(BaseRendererConfig):
    """Nemotron-3 **Ultra** renderer config — distinct discriminator so the
    registry routes Ultra checkpoints to the Ultra template variant.

    Ultra's template differs from Nano/Super: the reasoning block is glued as
    ``<think>\\n{reasoning}</think>{content}`` (no ``\\n`` around ``</think>``)
    and truncated historical turns collapse to ``<think></think>{content}``
    (no ``\\n``). It shares the :class:`renderers.nemotron3.Nemotron3Renderer`
    implementation, which selects the variant from ``config.name``.
    """

    name: Literal["nemotron-3-ultra"] = "nemotron-3-ultra"

    enable_thinking: bool = True
    """See :class:`Nemotron3RendererConfig.enable_thinking`."""

    truncate_history_thinking: bool = True
    """See :class:`Nemotron3RendererConfig.truncate_history_thinking`."""

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        _reject_thinking_retention_conflict(
            self,
            "truncate_history_thinking",
            true_implies="tool_cycle",
            false_implies="all",
        )
        return self

    medium_effort: bool = False
    """When ``True``, append ``\\n\\n{reasoning effort: efficient}`` to the last
    user message. Mirrors the Ultra chat template's ``medium_effort`` kwarg."""


class DeepSeekV3RendererConfig(BaseRendererConfig):
    """DeepSeek-V3 renderer config (non-reasoning).

    DeepSeek-V3 has no thinking concept: the generation prompt is a bare
    ``<｜Assistant｜>`` and assistant content is emitted verbatim. For the
    reasoning variant use :class:`DeepSeekR1RendererConfig`.
    """

    name: Literal["deepseek-v3"] = "deepseek-v3"


class DeepSeekR1RendererConfig(BaseRendererConfig):
    """DeepSeek-R1 renderer config (reasoning).

    R1 always reasons — its chat template unconditionally prefills
    ``<think>\\n`` at the generation prompt and strips ``</think>`` from
    historical assistant turns. There is therefore no ``enable_thinking``
    knob (thinking is not optional). With ``thinking_retention=None`` the
    resolved bridge policy is ``"template"``; explicit ``"tool_cycle"`` /
    ``"all"`` are bridge-policy overrides. Applies to full
    ``deepseek-ai/DeepSeek-R1`` / ``-R1-0528``
    — NOT the R1-Distill-Qwen/Llama models, which use those base
    tokenizers and route to the Qwen3 / Llama-3 renderers.
    """

    name: Literal["deepseek-r1"] = "deepseek-r1"


RendererConfig = Annotated[
    Union[
        AutoRendererConfig,
        DefaultRendererConfig,
        Qwen3RendererConfig,
        Qwen35RendererConfig,
        Qwen36RendererConfig,
        Qwen3VLRendererConfig,
        GLM5RendererConfig,
        GLM51RendererConfig,
        GLM45RendererConfig,
        GptOssRendererConfig,
        KimiK2RendererConfig,
        KimiK25RendererConfig,
        LagunaXS2RendererConfig,
        Llama3RendererConfig,
        MiniMaxM2RendererConfig,
        Nemotron3RendererConfig,
        Nemotron3UltraRendererConfig,
        DeepSeekV3RendererConfig,
        DeepSeekR1RendererConfig,
    ],
    Field(discriminator="name"),
]
"""Discriminated union over every renderer config variant.

Downstream pydantic configs (prime-rl orchestrator, verifiers
``ClientConfig``) can hold a single field typed as ``RendererConfig``;
deserialization dispatches on ``name`` and exposes strictly the kwargs
that renderer supports. Bogus combinations (e.g. ``add_vision_id`` under
``name="qwen3"``) raise ``ValidationError`` at config-load time.
"""


# Map discriminator → config class. Used by ``create_renderer`` when
# resolving ``AutoRendererConfig`` against ``MODEL_RENDERER_MAP``: the
# resolved renderer name picks the corresponding typed config, and the
# auto config's ``thinking_retention`` field is carried over.
_CONFIG_BY_NAME: dict[str, type[BaseRendererConfig]] = {
    "auto": AutoRendererConfig,
    "default": DefaultRendererConfig,
    "qwen3": Qwen3RendererConfig,
    "qwen3.5": Qwen35RendererConfig,
    "qwen3.6": Qwen36RendererConfig,
    "qwen3-vl": Qwen3VLRendererConfig,
    "glm-5": GLM5RendererConfig,
    "glm-5.1": GLM51RendererConfig,
    "glm-4.5": GLM45RendererConfig,
    "gpt-oss": GptOssRendererConfig,
    "kimi-k2": KimiK2RendererConfig,
    "kimi-k2.5": KimiK25RendererConfig,
    "laguna-xs.2": LagunaXS2RendererConfig,
    "llama-3": Llama3RendererConfig,
    "minimax-m2": MiniMaxM2RendererConfig,
    "nemotron-3": Nemotron3RendererConfig,
    "nemotron-3-ultra": Nemotron3UltraRendererConfig,
    "deepseek-v3": DeepSeekV3RendererConfig,
    "deepseek-r1": DeepSeekR1RendererConfig,
}


def _config_class_for(name: str) -> type[BaseRendererConfig]:
    cls = _CONFIG_BY_NAME.get(name)
    if cls is None:
        raise ValueError(
            f"No renderer config registered for name={name!r}. "
            f"Known: {sorted(_CONFIG_BY_NAME)}"
        )
    return cls


def config_from_name(name: str) -> BaseRendererConfig | None:
    """Construct a default-valued config for the given renderer name.

    Convenience for callers that hold a renderer name as a string and
    want the matching typed config. ``"auto"`` returns ``None`` —
    :func:`renderers.create_renderer` interprets that as "run auto
    resolution against ``MODEL_RENDERER_MAP``", which is what callers
    expect from a bare-string name.
    """
    if name == "auto":
        return None
    return _config_class_for(name)()


__all__ = [
    "AutoRendererConfig",
    "BaseRendererConfig",
    "DefaultRendererConfig",
    "DeepSeekR1RendererConfig",
    "DeepSeekV3RendererConfig",
    "GLM45RendererConfig",
    "GLM51RendererConfig",
    "GLM5RendererConfig",
    "GptOssRendererConfig",
    "KimiK25RendererConfig",
    "KimiK2RendererConfig",
    "LagunaXS2RendererConfig",
    "Llama3RendererConfig",
    "MiniMaxM2RendererConfig",
    "Nemotron3RendererConfig",
    "Nemotron3UltraRendererConfig",
    "Qwen35RendererConfig",
    "Qwen36RendererConfig",
    "Qwen3RendererConfig",
    "Qwen3VLRendererConfig",
    "RendererConfig",
    "ResolvedThinkingRetention",
    "ThinkingRetention",
    "config_from_name",
]
