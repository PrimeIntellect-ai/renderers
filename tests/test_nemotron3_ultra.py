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

from renderers.base import MODEL_RENDERER_MAP
from renderers.configs import (
    Nemotron3RendererConfig,
    Nemotron3UltraRendererConfig,
    _config_class_for,
)
from renderers.nemotron3 import Nemotron3Renderer, _is_super

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


def test_both_variants_resolve_to_one_renderer_class():
    # The registry routes both discriminators to the shared renderer class.
    assert _config_class_for("nemotron-3") is Nemotron3RendererConfig
    assert _config_class_for("nemotron-3-ultra") is Nemotron3UltraRendererConfig


def test_renderer_reads_variant_from_config_name():
    # No tokenizer needed for the ``_ultra`` flag — it comes off config.name.
    # Build with a fake tokenizer that has the special tokens stubbed out.
    class _Tok:
        name_or_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        unk_token_id = -1

        def convert_tokens_to_ids(self, tok):
            # Deterministic non-unk ids so construction succeeds offline.
            return abs(hash(tok)) % 100_000 + 1

    nano = Nemotron3Renderer(_Tok(), Nemotron3RendererConfig())
    ultra = Nemotron3Renderer(_Tok(), Nemotron3UltraRendererConfig())
    assert nano._ultra is False
    assert ultra._ultra is True


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
