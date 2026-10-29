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

import pytest

from renderers import base
from renderers.base import TOKENIZER_SOURCE_OVERRIDES, TRUSTED_REVISIONS, load_tokenizer
from tests.model_assets import load_test_tokenizer

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


@pytest.mark.parametrize("revision", [None, "a" * 40])
@patch("transformers.AutoTokenizer.from_pretrained")
def test_unlisted_model_loads_without_remote_code(mock_from_pretrained, revision):
    """Ordinary assets can be pinned without enabling remote code."""
    load_tokenizer("Qwen/Qwen3-0.6B", revision=revision)
    args, kwargs = mock_from_pretrained.call_args
    assert args == ("Qwen/Qwen3-0.6B",)
    expected = {"trust_remote_code": False}
    if revision is not None:
        expected["revision"] = revision
    assert kwargs == expected


@pytest.mark.parametrize(
    "revision", [None, TRUSTED_REVISIONS["moonshotai/Kimi-K2.5"], "a" * 40]
)
@patch("transformers.AutoTokenizer.from_pretrained")
def test_kimi_loads_with_pinned_revision(mock_from_pretrained, revision):
    """Kimi-K2 family: trust_remote_code=True, revision pinned to the
    sha listed in TRUSTED_REVISIONS — never a branch name."""
    if revision is not None and revision != TRUSTED_REVISIONS["moonshotai/Kimi-K2.5"]:
        with pytest.raises(ValueError, match="reviewed remote-code revision"):
            load_tokenizer("moonshotai/Kimi-K2.5", revision=revision)
        mock_from_pretrained.assert_not_called()
        return
    load_tokenizer("moonshotai/Kimi-K2.5", revision=revision)
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

    tok = load_tokenizer(canonical)

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


def test_offsetless_byo_preserves_ids_without_content_attribution():
    """Offsetless BYO tokenizers keep the joined BPE pass but omit attribution."""

    class _NoOffsets:
        name_or_path = "anywhere/anything"

        def encode(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return [len(text)]

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("BYO tokenizer has no offsets")

    tokenizer = _NoOffsets()
    assert base._get_offset_tokenizer(tokenizer) is None

    attributed = base.attribute_text_segments(
        tokenizer,
        [("user\n", False), ("hello", True)],
    )

    assert attributed == [(10, False)]
    assert attributed.has_content_attribution is False


# ---------------------------------------------------------------------------
# Smoke: real tokenizer loads behave as expected
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_load_tokenizer_real_qwen_works_without_remote_code():
    """End-to-end: an unlisted model loads successfully without
    trust_remote_code. Qwen tokenizers don't ship custom Python."""
    tok = load_test_tokenizer("Qwen/Qwen3-0.6B")
    assert tok is not None
    # Smoke: the tokenizer can encode a basic string.
    ids = tok.encode("hello", add_special_tokens=False)
    assert len(ids) > 0


@pytest.mark.model
def test_load_tokenizer_real_kimi_uses_pinned_revision():
    """End-to-end: Kimi-K2.5 loads via the pinned-revision path. The
    parity tests already exercise this path — this test pins the
    contract that ``load_tokenizer`` is the only sanctioned entry
    point for the trusted-revision allow-list."""
    tok = load_test_tokenizer("moonshotai/Kimi-K2.5")
    assert tok is not None
    ids = tok.encode("hello", add_special_tokens=False)
    assert len(ids) > 0
