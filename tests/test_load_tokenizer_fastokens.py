"""Coverage for the fastokens fast-path in ``renderers.base.load_tokenizer``.

``load_tokenizer`` defaults to routing every supported model through
``fastokens.patch_transformers()`` for ~10x faster encode. Models in
``FASTOKENS_INCOMPATIBLE`` skip the patch (DeepSeek's Metaspace
pretokenizer isn't supported). Callers can opt out per-call with
``use_fastokens=False``.

These tests pin the policy:

1. The denylist contains the empirically-verified incompat models —
   adding to it should be a deliberate review action.
2. With ``use_fastokens=True`` (the default) on a compatible model, the
   resulting tokenizer's backend is the fastokens shim. Encode output
   stays byte-identical to vanilla.
3. With ``use_fastokens=False``, the resulting tokenizer is vanilla.
4. For incompat models, the fast path is silently skipped and the
   tokenizer still loads + encodes correctly.
5. The fastokens patch is removed immediately after the load so it
   doesn't leak into the caller's process — subsequent
   ``AutoTokenizer.from_pretrained`` calls outside ``load_tokenizer``
   use vanilla.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from renderers.base import (
    FASTOKENS_INCOMPATIBLE,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# Denylist shape
# ---------------------------------------------------------------------------


def test_fastokens_incompatible_is_explicit_set():
    """The denylist is small and audited — pinning the exact contents
    catches accidental drift. Adding/removing entries should be a
    deliberate action with a parity probe."""
    assert FASTOKENS_INCOMPATIBLE == frozenset(
        {
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-V3-Base",
        }
    )


# ---------------------------------------------------------------------------
# Fast path (compatible model — Qwen3.5-9B as representative)
# ---------------------------------------------------------------------------


_FAST_MODEL = "Qwen/Qwen3.5-9B"


def _backend_class_name(tok) -> str:
    """Return the class name of the underlying backend object so tests
    can tell vanilla from fastokens-shimmed tokenizers."""
    backend = getattr(tok, "_tokenizer", None)
    return type(backend).__name__ if backend is not None else type(tok).__name__


def test_default_uses_fastokens_on_compatible_model():
    tok = load_tokenizer(_FAST_MODEL)
    # The shim type is named ``_TokenizerShim`` (see fastokens._compat);
    # match by name so we don't import private fastokens internals.
    assert "Shim" in _backend_class_name(tok), (
        f"Expected fastokens shim backend, got {_backend_class_name(tok)!r}"
    )


def test_explicit_off_returns_vanilla_backend():
    tok = load_tokenizer(_FAST_MODEL, use_fastokens=False)
    assert "Shim" not in _backend_class_name(tok), (
        f"Expected vanilla backend, got {_backend_class_name(tok)!r}"
    )


def test_fast_and_vanilla_encode_identically_on_compatible_model():
    fast = load_tokenizer(_FAST_MODEL)
    vanilla = load_tokenizer(_FAST_MODEL, use_fastokens=False)
    samples = [
        "Hello, world!",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "🌍 emoji + 中文 + tabs\there",
        " ".join([f"word_{i}" for i in range(50)]),
    ]
    for s in samples:
        assert fast.encode(s, add_special_tokens=False) == vanilla.encode(
            s, add_special_tokens=False
        ), f"encode diverged on {s!r}"


# ---------------------------------------------------------------------------
# Denylist: incompat models silently skip the patch and still load.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(FASTOKENS_INCOMPATIBLE))
def test_incompat_model_loads_via_vanilla_backend(model):
    """For models we know diverge / fail under fastokens, the fast path
    must be skipped so the load still succeeds with a vanilla backend."""
    if "DeepSeek" in model:
        # Skip if upstream gating / size makes the load impractical here.
        # We only care that the path doesn't try fastokens. Probe the
        # tokenizer_config to make sure the repo is reachable; if not,
        # skip rather than fail (CI without HF auth, network issues).
        from huggingface_hub import HfApi

        try:
            HfApi().repo_info(model)
        except Exception as e:
            pytest.skip(f"{model}: repo unreachable in this env ({e})")
    tok = load_tokenizer(model)
    assert "Shim" not in _backend_class_name(tok), (
        f"{model}: should NOT have been patched; got {_backend_class_name(tok)!r}"
    )
    # And it still encodes.
    ids = tok.encode("hello", add_special_tokens=False)
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# Patch must not leak: AutoTokenizer.from_pretrained calls OUTSIDE
# load_tokenizer should still produce a vanilla tokenizer.
# ---------------------------------------------------------------------------


def test_patch_is_unloaded_after_call():
    """``load_tokenizer`` brackets the fastokens patch. After it returns
    a fastokens-shimmed tokenizer, a fresh ``AutoTokenizer.from_pretrained``
    call must NOT pick up the patch — the user's process stays clean."""
    fast = load_tokenizer(_FAST_MODEL)
    assert "Shim" in _backend_class_name(fast), "preconditions: fast path active"

    # Now call AutoTokenizer.from_pretrained directly. It MUST be vanilla.
    direct = AutoTokenizer.from_pretrained(_FAST_MODEL, trust_remote_code=False)
    assert "Shim" not in _backend_class_name(direct), (
        f"fastokens patch leaked into user-side AutoTokenizer call: "
        f"got {_backend_class_name(direct)!r}"
    )


# ---------------------------------------------------------------------------
# Failure-mode fallback: if fastokens raises during the patched load,
# load_tokenizer falls back to vanilla without surfacing the error.
# ---------------------------------------------------------------------------


def test_fallback_on_fastokens_load_error(monkeypatch):
    """Simulate fastokens raising during patched load — load_tokenizer
    should fall back to vanilla and return a working tokenizer."""
    import renderers.base as rb

    def _boom(*args, **kwargs):
        raise ValueError("simulated fastokens failure: unsupported pre-tokenizer")

    monkeypatch.setattr(rb, "_patched_load", _boom)

    tok = load_tokenizer(_FAST_MODEL)  # default use_fastokens=True
    # The vanilla fallback ran — backend is not a fastokens shim.
    assert "Shim" not in _backend_class_name(tok)
    # Still works.
    assert len(tok.encode("hi", add_special_tokens=False)) > 0


# ---------------------------------------------------------------------------
# Print suppression: fastokens itself prints "[fastokens]
# patch_transformers: ..." on every patch/unpatch call. Building a
# RendererPool of size N would emit ~N lines (the pool factory calls
# load_tokenizer once per slot). load_tokenizer swallows that stdout
# chatter and emits a single INFO log on the first patch instead.
# ---------------------------------------------------------------------------


def test_no_fastokens_stdout_chatter(capsys, caplog):
    """``load_tokenizer`` must not leak ``[fastokens]`` prints onto
    stdout, and must emit exactly one INFO log per process announcing
    the fast path (not once per call)."""
    import logging

    import renderers.base as rb

    # Reset the process-wide "announced" flag so this test sees the
    # first-call log even if another test loaded a tokenizer earlier.
    rb._FASTOKENS_ANNOUNCED = False

    with caplog.at_level(logging.INFO, logger="renderers.base"):
        load_tokenizer(_FAST_MODEL)
        load_tokenizer(_FAST_MODEL)

    captured = capsys.readouterr()
    assert "[fastokens]" not in captured.out, (
        f"fastokens print leaked to stdout: {captured.out!r}"
    )
    assert "[fastokens]" not in captured.err, (
        f"fastokens print leaked to stderr: {captured.err!r}"
    )

    fastokens_info = [
        r for r in caplog.records if "fastokens enabled" in r.getMessage()
    ]
    assert len(fastokens_info) == 1, (
        f"expected exactly one fastokens INFO log across two loads, "
        f"got {len(fastokens_info)}"
    )
