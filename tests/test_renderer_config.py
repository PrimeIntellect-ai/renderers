"""Unit tests for the typed-config surface — discriminated union,
auto-resolution, and ``extra="forbid"`` enforcement on per-renderer
configs."""

from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from renderers import (
    AutoRendererConfig,
    DefaultRendererConfig,
    GLM5RendererConfig,
    GptOssRendererConfig,
    Nemotron3RendererConfig,
    Qwen3RendererConfig,
    Qwen35RendererConfig,
    Qwen36RendererConfig,
    RendererConfig,
    base,
    create_renderer,
    create_renderer_pool,
)


def test_per_renderer_config_rejects_unknown_fields():
    """``extra="forbid"`` on every typed variant catches bogus keys at
    construction: ``add_vision_id`` doesn't exist on ``Qwen3RendererConfig``
    (Qwen3 is text-only), so passing it must raise."""
    with pytest.raises(ValidationError, match="add_vision_id"):
        Qwen3RendererConfig(add_vision_id=True)


def test_discriminated_union_dispatches_on_name():
    """A dict shaped like ``{"name": "glm-5", ...}`` deserialises to the
    matching typed config; the union ``RendererConfig`` is what
    downstream consumers (prime-rl, verifiers) hold as a single field."""
    ta = TypeAdapter(RendererConfig)
    parsed = ta.validate_python(
        {"name": "glm-5", "enable_thinking": False, "clear_thinking": False}
    )
    assert isinstance(parsed, GLM5RendererConfig)
    assert parsed.enable_thinking is False
    assert parsed.clear_thinking is False


def test_discriminated_union_rejects_wrong_renderer_kwargs():
    """``add_vision_id`` under ``name="qwen3"`` is invalid at deserialise
    time — the discriminator narrows to ``Qwen3RendererConfig`` whose
    schema does not include that field."""
    ta = TypeAdapter(RendererConfig)
    with pytest.raises(ValidationError, match="add_vision_id"):
        ta.validate_python({"name": "qwen3", "add_vision_id": True})


def test_default_renderer_config_accepts_arbitrary_extras():
    """``DefaultRenderer`` wraps ``apply_chat_template`` for unknown
    templates, so its config uses ``extra="allow"`` and surfaces extras
    via ``model_extra``."""
    cfg = DefaultRendererConfig(
        tool_parser="qwen3", enable_thinking=False, custom_jinja_kwarg=True
    )
    assert cfg.tool_parser == "qwen3"
    assert cfg.model_extra == {
        "enable_thinking": False,
        "custom_jinja_kwarg": True,
    }


def test_create_renderer_forwards_typed_config_to_renderer(monkeypatch):
    """``create_renderer`` dispatches on ``config.name`` via
    ``RENDERER_REGISTRY``; the renderer stores the config it was given."""

    class _FakeRenderer:
        def __init__(self, tokenizer, config):
            self.tokenizer = tokenizer
            self.config = config

    monkeypatch.setitem(base.RENDERER_REGISTRY, "qwen3", _FakeRenderer)

    renderer = create_renderer(
        SimpleNamespace(name_or_path="unused"),
        Qwen3RendererConfig(enable_thinking=False),
    )
    assert isinstance(renderer.config, Qwen3RendererConfig)
    assert renderer.config.enable_thinking is False


def test_create_renderer_auto_resolves_via_model_map(monkeypatch):
    """``AutoRendererConfig`` (or ``config=None``) routes through
    ``MODEL_RENDERER_MAP`` to pick the matching renderer + typed config,
    carrying the shared ``thinking_retention`` field over from the auto config."""

    class _FakeQwen35:
        def __init__(self, tokenizer, config):
            self.tokenizer = tokenizer
            self.config = config

    monkeypatch.setitem(base.RENDERER_REGISTRY, "qwen3.5", _FakeQwen35)
    monkeypatch.setitem(base.MODEL_RENDERER_MAP, "fake/qwen35", "qwen3.5")

    renderer = create_renderer(
        SimpleNamespace(name_or_path="fake/qwen35"),
        AutoRendererConfig(thinking_retention="all"),
    )

    assert isinstance(renderer.config, Qwen35RendererConfig)
    assert renderer.config.thinking_retention == "all"
    # Template-level kwargs stay at their per-renderer defaults — auto
    # carries only the thinking_retention flag.
    assert renderer.config.add_vision_id is False


def test_create_renderer_auto_applies_chat_template_kwargs(monkeypatch):
    """Auto resolution happens before chat-template kwargs are validated."""

    class _FakeQwen3:
        def __init__(self, tokenizer, config):
            self.tokenizer = tokenizer
            self.config = config

    monkeypatch.setitem(base.RENDERER_REGISTRY, "qwen3", _FakeQwen3)
    monkeypatch.setitem(base.MODEL_RENDERER_MAP, "fake/qwen3", "qwen3")

    renderer = create_renderer(
        SimpleNamespace(name_or_path="fake/qwen3"),
        chat_template_kwargs={"enable_thinking": False},
    )

    assert isinstance(renderer.config, Qwen3RendererConfig)
    assert renderer.config.enable_thinking is False


def test_create_renderer_pool_forwards_chat_template_kwargs(monkeypatch):
    """Pool construction uses the same renderer-owned config resolution."""

    class _FakeQwen3:
        def __init__(self, tokenizer, config):
            self.config = config

    monkeypatch.setitem(base.RENDERER_REGISTRY, "qwen3", _FakeQwen3)
    monkeypatch.setitem(base.MODEL_RENDERER_MAP, "fake/qwen3", "qwen3")
    monkeypatch.setattr(
        base,
        "load_tokenizer",
        lambda name: SimpleNamespace(name_or_path=name),
    )

    pool = create_renderer_pool(
        "fake/qwen3",
        size=1,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert isinstance(pool._sole.config, Qwen3RendererConfig)
    assert pool._sole.config.enable_thinking is False


def test_auto_unknown_model_rejects_chat_template_kwargs():
    tok = SimpleNamespace(name_or_path="unknown/text-model")

    with pytest.raises(ValueError, match="chat_template_kwargs"):
        create_renderer(tok, chat_template_kwargs={"enable_thinking": False})


def test_chat_template_kwargs_validate_against_resolved_config(monkeypatch):
    class _FakeQwen3:
        def __init__(self, tokenizer, config):
            self.config = config

    monkeypatch.setitem(base.RENDERER_REGISTRY, "qwen3", _FakeQwen3)
    monkeypatch.setitem(base.MODEL_RENDERER_MAP, "fake/qwen3", "qwen3")

    with pytest.raises(ValidationError, match="enable_thinkng"):
        create_renderer(
            SimpleNamespace(name_or_path="fake/qwen3"),
            chat_template_kwargs={"enable_thinkng": False},
        )


def test_chat_template_kwargs_conflict_with_explicit_config():
    with pytest.raises(ValidationError, match="thinking_retention"):
        create_renderer(
            SimpleNamespace(name_or_path="fake/glm"),
            GLM5RendererConfig(thinking_retention="tool_cycle"),
            chat_template_kwargs={"clear_thinking": False},
        )


def test_chat_template_kwargs_preserve_default_field_unset_state(monkeypatch):
    class _FakeGlm:
        def __init__(self, tokenizer, config):
            self.config = config

    monkeypatch.setitem(base.RENDERER_REGISTRY, "glm-5", _FakeGlm)

    renderer = create_renderer(
        SimpleNamespace(name_or_path="fake/glm"),
        GLM5RendererConfig(thinking_retention="all"),
        chat_template_kwargs={"enable_thinking": False},
    )

    assert renderer.config.thinking_retention == "all"
    assert renderer.config.enable_thinking is False


def test_create_renderer_default_argument_is_auto():
    """Passing no config is equivalent to passing ``AutoRendererConfig()``
    — short form for the common case."""
    tok = SimpleNamespace(name_or_path="")  # no MODEL_RENDERER_MAP entry
    renderer = create_renderer(tok)
    # Falls through to DefaultRenderer when no match and no vision config.
    assert renderer.__class__.__name__ == "DefaultRenderer"


@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (GLM5RendererConfig, {"thinking_retention": "template"}),
        (
            GLM5RendererConfig,
            {"clear_thinking": False, "thinking_retention": "tool_cycle"},
        ),
        (GLM5RendererConfig, {"clear_thinking": True, "thinking_retention": "all"}),
        (
            Nemotron3RendererConfig,
            {
                "truncate_history_thinking": False,
                "thinking_retention": "tool_cycle",
            },
        ),
        (
            Nemotron3RendererConfig,
            {"truncate_history_thinking": True, "thinking_retention": "all"},
        ),
        (
            Qwen36RendererConfig,
            {"preserve_thinking": True, "thinking_retention": "tool_cycle"},
        ),
        (
            Qwen36RendererConfig,
            {"preserve_thinking": False, "thinking_retention": "all"},
        ),
        (
            GptOssRendererConfig,
            {"auto_drop_analysis": False, "thinking_retention": "tool_cycle"},
        ),
        (
            GptOssRendererConfig,
            {"auto_drop_analysis": True, "thinking_retention": "all"},
        ),
    ],
)
def test_thinking_retention_conflict_raises(config_cls, kwargs):
    """Explicit template and generic retention knobs must agree."""
    with pytest.raises(ValidationError, match="thinking_retention"):
        config_cls(**kwargs)


@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (GLM5RendererConfig, {"clear_thinking": False, "thinking_retention": "all"}),
        (
            GLM5RendererConfig,
            {"clear_thinking": True, "thinking_retention": "tool_cycle"},
        ),
        (GLM5RendererConfig, {"clear_thinking": False}),
        (GLM5RendererConfig, {"thinking_retention": "tool_cycle"}),
        (
            Nemotron3RendererConfig,
            {"truncate_history_thinking": False, "thinking_retention": "all"},
        ),
        (
            Nemotron3RendererConfig,
            {
                "truncate_history_thinking": True,
                "thinking_retention": "tool_cycle",
            },
        ),
        (
            Qwen36RendererConfig,
            {"preserve_thinking": True, "thinking_retention": "all"},
        ),
        (
            Qwen36RendererConfig,
            {"preserve_thinking": False, "thinking_retention": "tool_cycle"},
        ),
        (
            GptOssRendererConfig,
            {"auto_drop_analysis": False, "thinking_retention": "all"},
        ),
        (
            GptOssRendererConfig,
            {"auto_drop_analysis": True, "thinking_retention": "tool_cycle"},
        ),
    ],
)
def test_thinking_retention_consistent_pairs_are_accepted(config_cls, kwargs):
    config_cls(**kwargs)


def test_default_renderer_rejects_explicit_retention():
    """Opaque apply_chat_template fallback cannot implement bridge policy."""
    tok = SimpleNamespace(name_or_path="")
    create_renderer(tok, DefaultRendererConfig())

    for retention in ("tool_cycle", "all"):
        with pytest.raises(ValueError, match="DefaultRenderer"):
            create_renderer(tok, DefaultRendererConfig(thinking_retention=retention))


def test_default_renderer_config_rejects_legacy_preserve_flags():
    """``DefaultRendererConfig`` is ``extra="allow"``, so the removed
    ``preserve_*`` bools would otherwise slip into ``model_extra`` and be
    forwarded to ``apply_chat_template`` silently. A validator rejects them
    with a migration message; genuine Jinja kwargs still pass through."""
    with pytest.raises(ValidationError, match="thinking_retention"):
        DefaultRendererConfig(preserve_all_thinking=True)
    with pytest.raises(ValidationError, match="thinking_retention"):
        DefaultRendererConfig(preserve_thinking_between_tool_calls=True)

    cfg = DefaultRendererConfig(some_jinja_kwarg=True)
    assert cfg.model_extra["some_jinja_kwarg"] is True
