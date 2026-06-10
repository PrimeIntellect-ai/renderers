"""Offline wiring tests for the Nemotron-3 variant split.

Assert the model→renderer mapping, the per-variant typed-config surface, and
the name-based ``low_effort`` gating WITHOUT loading any tokenizer (no
network). This pins the wiring the parity matrix can't reach — in particular
the FP8 Ultra entry, which no test loads a tokenizer for — so it can't
silently rot.

The two variants:

* ``nemotron-3`` — Nano / Super, shared template. Config exposes ``low_effort``
  (honoured on Super, a no-op on Nano).
* ``nemotron-3-ultra`` — Ultra, distinct ``</think>`` glue. Config exposes
  ``medium_effort``.

Both route to the one ``Nemotron3Renderer`` class, which selects the variant
from ``config.name``.
"""

from types import SimpleNamespace

from renderers.base import MODEL_RENDERER_MAP, RENDERER_REGISTRY, _populate_registry
from renderers.configs import (
    Nemotron3RendererConfig,
    Nemotron3UltraRendererConfig,
    _config_class_for,
)
from renderers.nemotron3 import Nemotron3Renderer, Nemotron3UltraRenderer, _is_super

_ULTRA_REPOS = [
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
]
_NANO_SUPER_REPOS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
]


def _fake_tok(name):
    return SimpleNamespace(name_or_path=name)


def test_models_map_to_their_variant():
    for repo in _ULTRA_REPOS:
        assert MODEL_RENDERER_MAP.get(repo) == "nemotron-3-ultra", repo
    for repo in _NANO_SUPER_REPOS:
        assert MODEL_RENDERER_MAP.get(repo) == "nemotron-3", repo


def test_each_discriminator_maps_to_its_config_and_renderer_class():
    # Config discriminator → config class.
    assert _config_class_for("nemotron-3") is Nemotron3RendererConfig
    assert _config_class_for("nemotron-3-ultra") is Nemotron3UltraRendererConfig
    # Registry → renderer class (Ultra is a sibling subclass, matching the
    # GLM-5/5.1 and Qwen3.5/3.6 house style — not one class under two names).
    _populate_registry()
    assert RENDERER_REGISTRY["nemotron-3"] is Nemotron3Renderer
    assert RENDERER_REGISTRY["nemotron-3-ultra"] is Nemotron3UltraRenderer
    assert issubclass(Nemotron3UltraRenderer, Nemotron3Renderer)


def test_variant_is_encoded_by_the_class():
    # The ``</think>`` glue is selected by the class hook, not config.name —
    # so the right renderer class must be constructed (create_renderer routes
    # config.name → class). Default config also follows the class.
    assert Nemotron3Renderer._ultra is False
    assert Nemotron3UltraRenderer._ultra is True
    assert Nemotron3Renderer._config_cls is Nemotron3RendererConfig
    assert Nemotron3UltraRenderer._config_cls is Nemotron3UltraRendererConfig


def test_template_fields_per_variant():
    # ``low_effort`` lives only on the Nano/Super config; ``medium_effort``
    # only on Ultra. Both ARE chat-template kwargs (unlike the removed ``ultra``
    # selector), so they appear in the template-field surface.
    assert Nemotron3RendererConfig.template_field_names() == frozenset(
        {"enable_thinking", "truncate_history_thinking", "low_effort"}
    )
    assert Nemotron3UltraRendererConfig.template_field_names() == frozenset(
        {"enable_thinking", "truncate_history_thinking", "medium_effort"}
    )


def test_configs_reject_the_other_variants_effort_kwarg():
    # Discriminated-union honesty: a bad combination fails at config-load.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Nemotron3RendererConfig(medium_effort=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Nemotron3UltraRendererConfig(low_effort=True)  # type: ignore[call-arg]
    # And the removed ``ultra`` selector is gone entirely.
    with pytest.raises(ValidationError):
        Nemotron3RendererConfig(ultra=True)  # type: ignore[call-arg]


def test_is_super_name_detection():
    assert _is_super(_fake_tok("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"))
    assert not _is_super(_fake_tok("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"))
    # Unknown / local-path checkpoints default to False → low_effort no-op.
    assert not _is_super(_fake_tok("/home/user/local-ckpt"))
    assert not _is_super(SimpleNamespace())  # no name_or_path attr
