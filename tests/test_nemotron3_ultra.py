"""Offline wiring tests for the Nemotron-3 Ultra template variant.

Assert the name-based ``ultra`` auto-selection, the model→renderer mapping,
and the typed-config surface WITHOUT loading any tokenizer (no network). This
pins the wiring the parity matrix can't reach — in particular the FP8 entry,
which no test loads a tokenizer for — so it can't silently rot.
"""

from types import SimpleNamespace

from renderers.base import MODEL_RENDERER_MAP
from renderers.configs import Nemotron3RendererConfig
from renderers.nemotron3 import _ULTRA_DEFAULTS, _default_ultra

_ULTRA_REPOS = [
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
]
_NON_ULTRA_REPOS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
]


def _fake_tok(name):
    return SimpleNamespace(name_or_path=name)


def test_ultra_and_non_ultra_models_map_to_nemotron3():
    for repo in _ULTRA_REPOS + _NON_ULTRA_REPOS:
        assert MODEL_RENDERER_MAP.get(repo) == "nemotron-3", repo


def test_default_ultra_resolves_by_name():
    # Ultra checkpoints (incl. the gated FP8 repo) resolve True.
    for repo in _ULTRA_REPOS:
        assert _ULTRA_DEFAULTS[repo] is True
        assert _default_ultra(_fake_tok(repo)) is True
    # Nano / Super resolve False (the shared Nano/Super template).
    for repo in _NON_ULTRA_REPOS:
        assert _default_ultra(_fake_tok(repo)) is False
    # Unknown / fine-tuned / local-path checkpoints fall back to False;
    # those must pass an explicit ultra= if they need the Ultra template.
    assert _default_ultra(_fake_tok("acme/my-nemotron-ultra-ft")) is False
    assert _default_ultra(_fake_tok("/home/user/local-ckpt")) is False
    assert _default_ultra(SimpleNamespace()) is False  # no name_or_path attr


def test_ultra_is_not_a_template_kwarg():
    fields = Nemotron3RendererConfig.template_field_names()
    assert "ultra" not in fields
    assert fields == frozenset({"enable_thinking", "truncate_history_thinking"})
    assert "ultra" in Nemotron3RendererConfig._internal_fields


def test_ultra_config_default_is_none_and_overridable():
    assert Nemotron3RendererConfig().ultra is None  # None => auto-detect by name
    assert Nemotron3RendererConfig(ultra=True).ultra is True
    assert Nemotron3RendererConfig(ultra=False).ultra is False
