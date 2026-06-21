"""Unit tests for ``renderers.base.load_tokenizer`` security policy.

The renderers package centralises ``trust_remote_code`` handling here:
default off, opt-in only for the Moonshot Kimi-K2 family, and even then
pinned to a reviewed revision so a future malicious push to the upstream
repo doesn't auto-propagate.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

from renderers import base
from renderers.base import TOKENIZER_SOURCE_OVERRIDES, TRUSTED_REVISIONS, load_tokenizer


# ---------------------------------------------------------------------------
# Allow-list shape
# ---------------------------------------------------------------------------


def test_trusted_revisions_only_kimi_family():
    """Only the Moonshot Kimi-K2 family is allowed to run repo-supplied
    Python at ``from_pretrained`` time. Adding a new entry here means
    the renderers package is opting into arbitrary-code execution for
    that model — should require deliberate review."""
    assert set(TRUSTED_REVISIONS) == {
        "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Kimi-K2.5",
        "moonshotai/Kimi-K2.6",
    }


def test_trusted_revisions_are_full_shas():
    """Every pinned revision must be a 40-char hex sha — never a branch
    name like ``main`` or ``HEAD``, which would defeat the purpose by
    auto-resolving to whatever ``HEAD`` points at."""
    sha_re = re.compile(r"^[0-9a-f]{40}$")
    for model, rev in TRUSTED_REVISIONS.items():
        assert sha_re.fullmatch(rev), (
            f"{model}: revision {rev!r} is not a 40-char hex sha — "
            f"branch names / tags can drift, only commit shas pin behaviour."
        )


# ---------------------------------------------------------------------------
# Call-shape: which kwargs reach AutoTokenizer.from_pretrained?
# ---------------------------------------------------------------------------


@patch("transformers.AutoTokenizer.from_pretrained")
def test_unlisted_model_loads_without_remote_code(mock_from_pretrained):
    """Default path: trust_remote_code=False, no revision pin."""
    load_tokenizer("Qwen/Qwen3-0.6B")
    args, kwargs = mock_from_pretrained.call_args
    assert args == ("Qwen/Qwen3-0.6B",)
    assert kwargs == {"trust_remote_code": False}


@patch("transformers.AutoTokenizer.from_pretrained")
def test_kimi_loads_with_pinned_revision(mock_from_pretrained):
    """Kimi-K2 family: trust_remote_code=True, revision pinned to the
    sha listed in TRUSTED_REVISIONS — never a branch name."""
    load_tokenizer("moonshotai/Kimi-K2.5")
    args, kwargs = mock_from_pretrained.call_args
    assert args == ("moonshotai/Kimi-K2.5",)
    assert kwargs == {
        "trust_remote_code": True,
        "revision": TRUSTED_REVISIONS["moonshotai/Kimi-K2.5"],
    }


@patch("transformers.AutoTokenizer.from_pretrained")
def test_meta_llama_loads_tokenizer_from_unsloth_mirror(mock_from_pretrained):
    """Canonical Meta Llama repos are gated; load their tokenizer/chat
    template from the audited unrestricted mirror while preserving the
    canonical name for renderer auto-resolution."""
    canonical = "meta-llama/Llama-3.2-1B-Instruct"
    mirror = "unsloth/Llama-3.2-1B-Instruct"
    mock_from_pretrained.return_value = SimpleNamespace(name_or_path=mirror)

    tok = load_tokenizer(canonical, use_fastokens=False)

    args, kwargs = mock_from_pretrained.call_args
    assert args == (mirror,)
    assert kwargs == {"trust_remote_code": False}
    assert tok.name_or_path == canonical


@patch("transformers.AutoTokenizer.from_pretrained")
def test_unknown_path_falls_through_to_no_remote_code(mock_from_pretrained):
    """Unknown / fine-tuned model paths — including ``moonshotai/Kimi-K2*``
    look-alikes that aren't in the allow-list — must fall through to
    ``trust_remote_code=False``. No prefix matching, no fuzzy fallback.
    Callers who legitimately need a custom-code tokenizer outside the
    allow-list call ``AutoTokenizer.from_pretrained`` themselves."""
    cases = [
        "some-org/random-finetune",
        "moonshotai/Kimi-K3",  # hypothetical future, NOT in allow-list
        "/local/path/to/tokenizer",
    ]
    for name in cases:
        mock_from_pretrained.reset_mock()
        load_tokenizer(name)
        args, kwargs = mock_from_pretrained.call_args
        assert args == (name,)
        assert kwargs == {"trust_remote_code": False}, (
            f"{name}: unlisted path leaked trust_remote_code=True"
        )


def test_tokenizer_source_overrides_are_exact_llama_mirrors():
    """Mirror overrides are intentionally narrow: only verified
    byte-identical Llama tokenizer/template mirrors should live here."""
    assert TOKENIZER_SOURCE_OVERRIDES == {
        "meta-llama/Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
    }


def test_offset_tokenizer_uses_unsloth_mirror_for_meta_llama(monkeypatch):
    """Offset-tokenizer reloads must use the same unrestricted source
    override, otherwise Llama rendering can hit the gated Meta repo after
    the initial tokenizer load succeeds."""

    class _NoOffsets:
        name_or_path = "meta-llama/Llama-3.2-1B-Instruct"

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("fastokens shim has no offsets")

    class _OffsetTokenizer:
        is_fast = True

        def __init__(self, name_or_path: str):
            self.name_or_path = name_or_path

        def __call__(self, *args, **kwargs):
            return {"offset_mapping": [(0, 1)]}

    calls = []

    def _fake_load(name_or_path, **kwargs):
        calls.append((name_or_path, kwargs))
        return _OffsetTokenizer(name_or_path)

    base._offset_tokenizers.clear()
    monkeypatch.setattr(base, "_load_tokenizer_via_auto", _fake_load)

    try:
        tok = base._get_offset_tokenizer(_NoOffsets())
    finally:
        base._offset_tokenizers.clear()

    assert calls == [("unsloth/Llama-3.2-1B-Instruct", {"trust_remote_code": False})]
    assert tok.name_or_path == "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Smoke: real tokenizer loads behave as expected
# ---------------------------------------------------------------------------


def test_load_tokenizer_real_qwen_works_without_remote_code():
    """End-to-end: an unlisted model loads successfully without
    trust_remote_code. Qwen tokenizers don't ship custom Python."""
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    assert tok is not None
    # Smoke: the tokenizer can encode a basic string.
    ids = tok.encode("hello", add_special_tokens=False)
    assert len(ids) > 0


def test_load_tokenizer_real_kimi_uses_pinned_revision():
    """End-to-end: Kimi-K2.5 loads via the pinned-revision path. The
    parity tests already exercise this path — this test pins the
    contract that ``load_tokenizer`` is the only sanctioned entry
    point for the trusted-revision allow-list."""
    tok = load_tokenizer("moonshotai/Kimi-K2.5")
    assert tok is not None
    ids = tok.encode("hello", add_special_tokens=False)
    assert len(ids) > 0
