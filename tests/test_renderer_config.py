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
    Qwen3RendererConfig,
    Qwen35RendererConfig,
    RendererConfig,
    base,
    create_renderer,
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


def test_create_renderer_default_argument_is_auto():
    """Passing no config is equivalent to passing ``AutoRendererConfig()``
    — short form for the common case."""
    tok = SimpleNamespace(name_or_path="")  # no MODEL_RENDERER_MAP entry
    renderer = create_renderer(tok)
    # Falls through to DefaultRenderer when no match and no vision config.
    assert renderer.__class__.__name__ == "DefaultRenderer"


def test_thinking_retention_conflict_raises():
    """Explicit template and generic retention knobs must agree."""
    from renderers import Nemotron3RendererConfig

    with pytest.raises(ValidationError, match="thinking_retention"):
        GLM5RendererConfig(thinking_retention="template")
    with pytest.raises(ValidationError, match="thinking_retention"):
        GLM5RendererConfig(clear_thinking=False, thinking_retention="tool_cycle")
    with pytest.raises(ValidationError, match="thinking_retention"):
        GLM5RendererConfig(clear_thinking=True, thinking_retention="all")
    with pytest.raises(ValidationError, match="thinking_retention"):
        Nemotron3RendererConfig(
            truncate_history_thinking=False, thinking_retention="tool_cycle"
        )

    # Consistent pairs and single-field configs are accepted.
    GLM5RendererConfig(clear_thinking=False, thinking_retention="all")
    GLM5RendererConfig(clear_thinking=False)
    GLM5RendererConfig(thinking_retention="tool_cycle")


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
