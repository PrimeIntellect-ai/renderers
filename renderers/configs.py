"""Typed renderer configs — one pydantic model per renderer, unified by a
discriminated union (``RendererConfig``).

Each renderer accepts its own typed config and declares an explicit
``_template_fields`` allowlist for fields that may arrive through
``chat_template_kwargs``. Bad combinations (e.g. ``add_vision_id`` under
``name="qwen3"``) fail before renderer construction. The shared
``thinking_retention`` flag is optional: ``None`` means
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

    # Every renderer-specific field must be classified exactly once: either
    # as a chat-template kwarg or as renderer-internal configuration.
    _template_fields: ClassVar[frozenset[str]] = frozenset()
    _internal_fields: ClassVar[frozenset[str]] = frozenset()
    _allow_opaque_template_kwargs: ClassVar[bool] = False

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        base_fields = frozenset(BaseRendererConfig.model_fields)
        renderer_fields = frozenset(cls.model_fields) - base_fields - {"name"}
        overlap = cls._template_fields & cls._internal_fields
        missing = renderer_fields - cls._template_fields - cls._internal_fields
        unknown = (cls._template_fields | cls._internal_fields) - renderer_fields
        if overlap or missing or unknown:
            raise TypeError(
                f"{cls.__name__} has an invalid renderer-field classification: "
                f"overlap={sorted(overlap)}, missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )

    @classmethod
    def template_field_names(cls) -> frozenset[str]:
        """Subset of fields that mirror Jinja chat-template kwargs.

        Used both as the runtime allowlist for ``chat_template_kwargs`` and
        by parity tests to discover the cells that must agree with
        ``apply_chat_template``.
        """
        return cls._template_fields


class AutoRendererConfig(BaseRendererConfig):
    """Resolve the renderer from ``tokenizer.name_or_path`` at construction
    time via ``MODEL_RENDERER_MAP``. Carries only the shared
    ``thinking_retention`` field when explicitly set; template kwargs require
    an explicit renderer choice so template-dependent behaviour stays visible
    at the call site."""

    name: Literal["auto"] = "auto"
    _template_fields = frozenset()


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
    _template_fields = frozenset()
    _allow_opaque_template_kwargs = True

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
    _template_fields = frozenset({"enable_thinking"})

    enable_thinking: bool = True
    """When ``True``, the generation prompt includes ``<think>`` so the
    model continues into a thinking block. Mirrors the chat template's
    ``enable_thinking`` kwarg.

    When ``False``, the renderer deliberately deviates from the template
    on historical assistant turns without ``reasoning_content``: the empty
    ``<think>\\n\\n</think>\\n\\n`` wrapper the generation prompt prefilled
    is re-emitted instead of stripped, keeping re-renders token-stable with
    the sampled stream (see ``renderers/qwen3.py`` module docstring)."""


class PrimeQwen3RendererConfig(BaseRendererConfig):
    """PrimeIntellect Qwen3 renderer config."""

    name: Literal["prime-qwen3"] = "prime-qwen3"
    _template_fields = frozenset()


class Qwen35RendererConfig(BaseRendererConfig):
    """Qwen3.5 renderer config."""

    name: Literal["qwen3.5"] = "qwen3.5"
    _template_fields = frozenset({"enable_thinking", "add_vision_id"})

    enable_thinking: bool | None = None
    """When ``True``, the generation prompt includes ``<think>``. ``None``
    auto-detects from the tokenizer's chat-template default — Instruct
    checkpoints default off, Thinking checkpoints default on. Mirrors
    the chat template's ``enable_thinking`` kwarg.

    When resolved ``False``, the renderer deliberately deviates from the
    template on historical assistant turns without ``reasoning_content``:
    the empty ``<think>\\n\\n</think>\\n\\n`` wrapper the generation prompt
    prefilled is re-emitted instead of stripped, keeping re-renders
    token-stable with the sampled stream (see ``renderers/qwen35.py``
    module docstring)."""

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
    _template_fields = frozenset(
        {"enable_thinking", "add_vision_id", "preserve_thinking"}
    )

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


class Qwen38RendererConfig(BaseRendererConfig):
    """Qwen3.8 renderer config.

    Qwen3.8 keeps Qwen3.6's multimodal/tool template and adds native
    reasoning-effort instructions. Historical thinking is preserved by
    default in the upstream template.
    """

    name: Literal["qwen3.8"] = "qwen3.8"
    _template_fields = frozenset(
        {
            "enable_thinking",
            "add_vision_id",
            "preserve_thinking",
            "reasoning_effort",
        }
    )

    enable_thinking: bool | None = None
    """See :class:`Qwen35RendererConfig.enable_thinking`."""

    add_vision_id: bool = False
    """See :class:`Qwen35RendererConfig.add_vision_id`."""

    preserve_thinking: bool = True
    """Keep historical ``<think>`` blocks. Mirrors Qwen3.8's default."""

    reasoning_effort: Literal["xhigh", "medium", "low"] = "xhigh"
    """Reasoning depth hint injected into the system prompt when enabled."""

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
    _template_fields = frozenset({"add_vision_id"})

    add_vision_id: bool = False
    """See :class:`Qwen35RendererConfig.add_vision_id`."""

    image_cache_max: int = 256
    """See :class:`Qwen35RendererConfig.image_cache_max`."""

    _internal_fields = frozenset({"image_cache_max"})


class Gemma4RendererConfig(BaseRendererConfig):
    """Gemma 4 renderer config."""

    name: Literal["gemma4"] = "gemma4"
    _template_fields = frozenset({"enable_thinking", "preserve_thinking"})

    enable_thinking: bool = False
    """Enable Gemma 4's thinking mode. Mirrors the canonical template kwarg."""

    preserve_thinking: bool = False
    """Keep thinking on historical tool-call turns when the template permits it."""

    image_cache_max: int = 256
    """FIFO bound on processed image entries. Renderer-internal."""

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


class GLM5RendererConfig(BaseRendererConfig):
    """GLM-5 renderer config."""

    name: Literal["glm-5"] = "glm-5"
    _template_fields = frozenset({"enable_thinking", "clear_thinking"})

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
    _template_fields = frozenset({"enable_thinking", "clear_thinking"})

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
    _template_fields = frozenset({"enable_thinking"})

    enable_thinking: bool = True
    """When ``True``, the generation prompt includes ``<think>``. Mirrors
    the chat template's ``enable_thinking`` kwarg."""


class Hy3RendererConfig(BaseRendererConfig):
    """Tencent Hunyuan Hy3 renderer config.

    Hy3 reasons via a ``reasoning_effort`` gate rather than a boolean
    ``enable_thinking``. ``"no_think"`` (the template default) prefills an
    empty ``<think></think>`` at the generation prompt so the model answers
    directly; ``"low"`` / ``"high"`` prefill only the ``<think>`` opener so
    the model streams reasoning up to a ``</think>`` it emits itself.

    ``preserved_thinking`` mirrors the template kwarg of the same name:
    ``True`` keeps ``<think>{reasoning}</think>`` on every historical
    assistant turn; ``False`` collapses past-cycle reasoning to
    ``<think></think>``, keeping it only on the in-flight turn (after the
    last user query). ``None`` (default) follows the template's own
    default — ``True`` when ``tools`` are supplied at render time, ``False``
    otherwise. Bridge policy tracks the same resolution: ``"all"`` whenever
    ``preserved_thinking`` resolves to ``True`` for the tools at hand,
    ``"tool_cycle"`` otherwise.
    """

    name: Literal["hy3"] = "hy3"
    _template_fields = frozenset(
        {
            "reasoning_effort",
            "preserved_thinking",
            "is_training",
            "raw_last_assistant",
            "fallback_strategy",
        }
    )

    reasoning_effort: Literal["no_think", "low", "high"] = "no_think"
    """Reasoning gate. Mirrors the chat template's ``reasoning_effort`` kwarg.
    ``"no_think"`` prefills ``<think></think>`` at the generation prompt;
    ``"low"`` / ``"high"`` prefill just ``<think>``."""

    preserved_thinking: bool | None = None
    """Keep historical assistant reasoning. Mirrors the template's
    ``preserved_thinking`` kwarg. ``None`` defers to the template default
    (``True`` with tools, ``False`` without). Bridge policy is ``"all"``
    whenever this resolves to ``True`` for the tools at hand."""

    is_training: bool = False
    """Mirrors the template's ``is_training`` kwarg. ``True`` renders SFT
    targets: reasoning is kept on every assistant turn (regardless of
    ``preserved_thinking`` / position) and the final assistant is terminated
    with ``<｜hy_eos｜>``. Leave ``False`` for inference-faithful renders;
    the training loss mask is normally derived via ``build_training_sample``
    rather than this flag."""

    raw_last_assistant: bool = False
    """Mirrors the template's ``raw_last_assistant`` kwarg. When ``True`` a
    trailing non-tool assistant message is emitted as raw visible content —
    no ``<think>`` wrap, no ``<｜hy_eos｜>`` — for prefill / continuation."""

    fallback_strategy: Literal["reasoning_toolcall_retry"] | None = None
    """Mirrors the template's ``fallback_strategy`` kwarg. The sole active
    value, ``"reasoning_toolcall_retry"``, forces ``reasoning_effort="high"``
    and suppresses the generation prompt (``add_generation_prompt=False``)."""

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        if self.preserved_thinking is not None and self.thinking_retention is not None:
            implied = "all" if self.preserved_thinking else "tool_cycle"
            if self.thinking_retention != implied:
                raise ValueError(
                    f"preserved_thinking={self.preserved_thinking!r} implies "
                    f"thinking_retention={implied!r}, which conflicts with "
                    f"explicit thinking_retention={self.thinking_retention!r}."
                )
        return self


# Inkling ``reasoning_effort`` label → float (the template's own effort_map).
INKLING_EFFORT_MAP: dict[str, float] = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "max": 0.99,
}


class InklingRendererConfig(BaseRendererConfig):
    """Renderer config for Inkling and Inkling-Small.

    Inkling gates reasoning depth via a ``reasoning_effort`` knob rather
    than a boolean ``enable_thinking``. It accepts either a string label
    from :data:`INKLING_EFFORT_MAP` (``none`` … ``max``) or a raw float in
    ``[0.0, 0.99]``; the renderer emits ``Thinking effort level: {N}`` in a
    leading system message exactly as the chat template does (label mapped
    to its float, ``0.0`` printed as ``"0"``). The template default —
    applied when no ``reasoning_effort`` is passed — is ``0.9``, which this
    config mirrors.

    Reasoning is preserved on every historical assistant turn (the template
    has no history-dropping knob), so the effective bridge policy is
    ``"all"``.
    """

    name: Literal["inkling"] = "inkling"
    _template_fields = frozenset({"reasoning_effort"})

    reasoning_effort: str | float = 0.9
    """Reasoning-effort gate. Mirrors the chat template's ``reasoning_effort``
    kwarg: a label in :data:`INKLING_EFFORT_MAP` or a float in ``[0.0, 0.99]``.
    Default ``0.9`` matches the template's own default (equivalent to
    ``"high"``)."""

    image_cache_max: int = 256
    """FIFO bound on the per-renderer image-processor cache. Renderer-
    internal — not a Jinja chat-template kwarg."""

    audio_cache_max: int = 256
    """FIFO bound on the per-renderer audio-processor cache. Renderer-
    internal — not a Jinja chat-template kwarg."""

    _internal_fields = frozenset({"image_cache_max", "audio_cache_max"})

    @model_validator(mode="after")
    def _check_reasoning_effort(self):
        eff = self.reasoning_effort
        if isinstance(eff, str):
            if eff.strip() not in INKLING_EFFORT_MAP:
                raise ValueError(
                    f"reasoning_effort={eff!r} is not a known label. "
                    f"Use one of {sorted(INKLING_EFFORT_MAP)} or a float in [0.0, 0.99]."
                )
        else:
            num = float(eff)
            if num < 0.0 or num > 0.99:
                raise ValueError(f"reasoning_effort={eff!r} must be in [0.0, 0.99].")
        return self


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
    _template_fields = frozenset({"reasoning_effort", "conversation_start_date"})

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
    _template_fields = frozenset()

    enable_thinking: bool = True
    """No-op for Kimi K2 (template doesn't gate on it). Stored for
    introspection / cross-renderer uniformity."""

    _internal_fields = frozenset({"enable_thinking"})


class KimiK25RendererConfig(BaseRendererConfig):
    """Kimi K2.5 renderer config."""

    name: Literal["kimi-k2.5"] = "kimi-k2.5"
    _template_fields = frozenset({"thinking"})

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
    _template_fields = frozenset({"enable_thinking", "render_assistant_messages_raw"})

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


class LagunaM1RendererConfig(BaseRendererConfig):
    """Laguna M.1 renderer config.

    Laguna M.1 shares Laguna XS.2's role and tool-call format, but its
    official checkpoint has a distinct chat template: it does not inject
    XS.2's fallback system message and it gives ``message.reasoning``
    precedence over ``message.reasoning_content``. Served by
    :class:`renderers.laguna_xs2.LagunaM1Renderer`.
    """

    name: Literal["laguna-m.1"] = "laguna-m.1"
    _template_fields = frozenset({"enable_thinking", "render_assistant_messages_raw"})

    enable_thinking: bool = False
    """When ``True``, the generation prompt includes ``<think>``. Mirrors
    the official template's ``enable_thinking`` kwarg and default."""

    render_assistant_messages_raw: bool = False
    """When ``True``, assistant messages use the official template's
    verbatim passthrough branch. See
    :class:`LagunaXS2RendererConfig.render_assistant_messages_raw`."""


class LagunaXS21RendererConfig(BaseRendererConfig):
    """Laguna XS-2.1 renderer config.

    XS-2.1's chat template reads a single kwarg, ``enable_thinking``,
    which gates both the generation prompt (``<think>`` vs ``</think>``)
    and whether assistant reasoning is rendered into the history at all.
    Served by :class:`renderers.laguna_xs2.LagunaXS21Renderer`.
    """

    name: Literal["laguna-xs-2.1"] = "laguna-xs-2.1"
    _template_fields = frozenset({"enable_thinking"})

    enable_thinking: bool = False
    """When ``True``, the generation prompt ends with ``<think>`` and
    every assistant turn renders ``<think>{reasoning}</think>``; when
    ``False``, turns open with a bare ``</think>`` and reasoning is not
    rendered. Mirrors the template's ``enable_thinking`` kwarg and its
    upstream default."""


class LagunaS21RendererConfig(BaseRendererConfig):
    """Laguna S-2.1 renderer config.

    S-2.1 is a larger sibling of XS-2.1 sharing its tokenizer (same vocab,
    special tokens, and merges), but its chat template is *not* byte-identical:
    ``enable_thinking`` defaults to ``True`` (XS-2.1 defaults ``False``), and a
    new ``preserve_thinking`` kwarg widens the reasoning-display gate to
    ``enable_thinking or preserve_thinking``. The token format is otherwise
    identical, so this is served by
    :class:`renderers.laguna_s21.LagunaS21Renderer`, a thin subclass of
    ``LagunaXS21Renderer`` that only overrides that gate.
    """

    name: Literal["laguna-s-2.1"] = "laguna-s-2.1"
    _template_fields = frozenset({"enable_thinking", "preserve_thinking"})

    enable_thinking: bool = True
    """When ``True``, the generation prompt ends with ``<think>`` and every
    assistant turn renders ``<think>{reasoning}</think>``; when ``False``,
    turns open with a bare ``</think>``. Mirrors the template's
    ``enable_thinking`` kwarg — note S-2.1's upstream default is ``True``,
    unlike XS-2.1's ``False``."""

    preserve_thinking: bool = False
    """When ``True``, assistant turns keep their ``<think>{reasoning}</think>``
    block even while ``enable_thinking`` is ``False`` — the template gates
    reasoning display on ``enable_thinking or preserve_thinking``. With the
    default ``False`` the gate collapses to ``enable_thinking`` and the
    renderer matches XS-2.1 turn-for-turn. Mirrors the template's
    ``preserve_thinking`` kwarg and its upstream default."""


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
    _template_fields = frozenset({"date_string", "tools_in_user_message"})

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
    _template_fields = frozenset({"model_identity"})

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
    _template_fields = frozenset(
        {"enable_thinking", "truncate_history_thinking", "low_effort"}
    )

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
    _template_fields = frozenset(
        {"enable_thinking", "truncate_history_thinking", "medium_effort"}
    )

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


class Nemotron35RendererConfig(BaseRendererConfig):
    """Nemotron-3.5 (Lightning) renderer config.

    Nemotron 3.5's chat template is the Ultra variant's minus the effort
    kwarg: same reasoning-block glue (no ``\\n`` around ``</think>``, no
    trim on truncated tool-call history), but its Jinja defines no
    reasoning-effort variable at all. It shares the
    :class:`renderers.nemotron3.Nemotron3Renderer` implementation via the
    :class:`renderers.nemotron3.Nemotron35Renderer` subclass and is reached
    via the ``nemotron-3.5`` discriminator.
    """

    name: Literal["nemotron-3.5"] = "nemotron-3.5"
    _template_fields = frozenset({"enable_thinking", "truncate_history_thinking"})

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


class DeepSeekV3RendererConfig(BaseRendererConfig):
    """DeepSeek-V3 renderer config (non-reasoning).

    DeepSeek-V3 has no thinking concept: the generation prompt is a bare
    ``<｜Assistant｜>`` and assistant content is emitted verbatim. For the
    reasoning variant use :class:`DeepSeekR1RendererConfig`.
    """

    name: Literal["deepseek-v3"] = "deepseek-v3"
    _template_fields = frozenset()


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
    _template_fields = frozenset()


class DeepSeekV4RendererConfig(BaseRendererConfig):
    """DeepSeek-V4-Flash-0731 reference-encoder configuration.

    The checkpoint ships a Python encoder rather than a Jinja template.  These
    fields mirror its public controls: chat vs thinking mode, historical
    reasoning dropping, and the opt-in thinking-effort prefix.
    """

    name: Literal["deepseek-v4"] = "deepseek-v4"
    _template_fields = frozenset(
        {"enable_thinking", "drop_thinking", "reasoning_effort"}
    )

    enable_thinking: bool = False
    """Select thinking mode.  ``False`` matches the official inference script."""

    drop_thinking: bool = True
    """Drop reasoning before the latest user query when no tools are present.

    The reference encoder automatically preserves all reasoning whenever tools
    are supplied, regardless of this value.
    """

    reasoning_effort: Literal["low", "high", "max"] = "low"
    """Thinking-only effort prefix; ``low`` adds no text.

    ``low`` is the checkpoint Python encoder's default. DeepSeek's hosted API
    independently defaults its thinking effort to ``high``.
    """

    @model_validator(mode="after")
    def _check_thinking_retention(self):
        _reject_thinking_retention_conflict(
            self,
            "drop_thinking",
            true_implies="tool_cycle",
            false_implies="all",
        )
        return self


RendererConfig = Annotated[
    Union[
        AutoRendererConfig,
        DefaultRendererConfig,
        Qwen3RendererConfig,
        PrimeQwen3RendererConfig,
        Qwen35RendererConfig,
        Qwen36RendererConfig,
        Qwen38RendererConfig,
        Qwen3VLRendererConfig,
        Gemma4RendererConfig,
        GLM5RendererConfig,
        GLM51RendererConfig,
        GLM45RendererConfig,
        GptOssRendererConfig,
        Hy3RendererConfig,
        InklingRendererConfig,
        KimiK2RendererConfig,
        KimiK25RendererConfig,
        LagunaXS2RendererConfig,
        LagunaM1RendererConfig,
        LagunaXS21RendererConfig,
        LagunaS21RendererConfig,
        Llama3RendererConfig,
        MiniMaxM2RendererConfig,
        Nemotron3RendererConfig,
        Nemotron3UltraRendererConfig,
        Nemotron35RendererConfig,
        DeepSeekV3RendererConfig,
        DeepSeekR1RendererConfig,
        DeepSeekV4RendererConfig,
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
    "prime-qwen3": PrimeQwen3RendererConfig,
    "qwen3.5": Qwen35RendererConfig,
    "qwen3.6": Qwen36RendererConfig,
    "qwen3.8": Qwen38RendererConfig,
    "qwen3-vl": Qwen3VLRendererConfig,
    "gemma4": Gemma4RendererConfig,
    "glm-5": GLM5RendererConfig,
    "glm-5.1": GLM51RendererConfig,
    "glm-4.5": GLM45RendererConfig,
    "gpt-oss": GptOssRendererConfig,
    "hy3": Hy3RendererConfig,
    "inkling": InklingRendererConfig,
    "kimi-k2": KimiK2RendererConfig,
    "kimi-k2.5": KimiK25RendererConfig,
    "laguna-xs.2": LagunaXS2RendererConfig,
    "laguna-m.1": LagunaM1RendererConfig,
    "laguna-xs-2.1": LagunaXS21RendererConfig,
    "laguna-s-2.1": LagunaS21RendererConfig,
    "llama-3": Llama3RendererConfig,
    "minimax-m2": MiniMaxM2RendererConfig,
    "nemotron-3": Nemotron3RendererConfig,
    "nemotron-3-ultra": Nemotron3UltraRendererConfig,
    "nemotron-3.5": Nemotron35RendererConfig,
    "deepseek-v3": DeepSeekV3RendererConfig,
    "deepseek-r1": DeepSeekR1RendererConfig,
    "deepseek-v4": DeepSeekV4RendererConfig,
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
    "DeepSeekV4RendererConfig",
    "GLM45RendererConfig",
    "GLM51RendererConfig",
    "GLM5RendererConfig",
    "Gemma4RendererConfig",
    "GptOssRendererConfig",
    "Hy3RendererConfig",
    "INKLING_EFFORT_MAP",
    "InklingRendererConfig",
    "KimiK25RendererConfig",
    "KimiK2RendererConfig",
    "LagunaM1RendererConfig",
    "LagunaS21RendererConfig",
    "LagunaXS2RendererConfig",
    "LagunaXS21RendererConfig",
    "Llama3RendererConfig",
    "MiniMaxM2RendererConfig",
    "Nemotron35RendererConfig",
    "Nemotron3RendererConfig",
    "Nemotron3UltraRendererConfig",
    "PrimeQwen3RendererConfig",
    "Qwen35RendererConfig",
    "Qwen36RendererConfig",
    "Qwen38RendererConfig",
    "Qwen3RendererConfig",
    "Qwen3VLRendererConfig",
    "RendererConfig",
    "ResolvedThinkingRetention",
    "ThinkingRetention",
    "config_from_name",
]
