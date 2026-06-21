"""Coverage for the fastokens fast-path in ``renderers.base.load_tokenizer``.

``load_tokenizer`` defaults to adapting every supported model's returned
backend with fastokens for ~10x faster encode. Models in
``FASTOKENS_INCOMPATIBLE`` skip adaptation (DeepSeek's Metaspace
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
5. Fastokens adaptation is scoped to the returned tokenizer, so concurrent
   and subsequent ``AutoTokenizer.from_pretrained`` calls stay vanilla.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast

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
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-R1-0528",
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
        fast_ids = fast.encode(s, add_special_tokens=False)
        vanilla_ids = vanilla.encode(s, add_special_tokens=False)
        assert fast_ids == vanilla_ids, f"encode diverged on {s!r}"
        assert fast.decode(fast_ids) == vanilla.decode(vanilla_ids), (
            f"decode diverged on {s!r}"
        )


# ---------------------------------------------------------------------------
# Denylist: incompat models silently skip adaptation and still load.
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
        f"{model}: should NOT have been adapted; got {_backend_class_name(tok)!r}"
    )
    # And it still encodes.
    ids = tok.encode("hello", add_special_tokens=False)
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# Adaptation must not leak: AutoTokenizer.from_pretrained calls OUTSIDE
# load_tokenizer should always produce a vanilla tokenizer.
# ---------------------------------------------------------------------------


def test_fastokens_is_scoped_to_loaded_tokenizer():
    """A fresh ``AutoTokenizer`` call stays vanilla after fast adaptation."""
    fast = load_tokenizer(_FAST_MODEL)
    assert "Shim" in _backend_class_name(fast), "preconditions: fast path active"

    # Now call AutoTokenizer.from_pretrained directly. It MUST be vanilla.
    direct = AutoTokenizer.from_pretrained(_FAST_MODEL, trust_remote_code=False)
    assert "Shim" not in _backend_class_name(direct), (
        f"fastokens leaked into user-side AutoTokenizer call: "
        f"got {_backend_class_name(direct)!r}"
    )


def test_fastokens_load_does_not_patch_transformers_concurrently(monkeypatch, tmp_path):
    """A slow renderer load must not expose fastokens to unrelated callers."""
    import renderers.base as rb

    backend = Tokenizer(models.BPE({"[UNK]": 0, "hello": 1}, [], unk_token="[UNK]"))
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    ).save_pretrained(tmp_path)

    started = threading.Event()
    release = threading.Event()
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)

    def slow_load(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return tokenizer

    monkeypatch.setattr(rb, "_load_tokenizer_via_auto", slow_load)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        loaded = executor.submit(load_tokenizer, "test-model")
        assert started.wait(timeout=5)
        try:
            direct = AutoTokenizer.from_pretrained(tmp_path)
            assert "Shim" not in _backend_class_name(direct)
        finally:
            release.set()

    assert "Shim" in _backend_class_name(loaded.result())


# ---------------------------------------------------------------------------
# Failure-mode fallback: if fastokens raises during per-tokenizer adaptation,
# load_tokenizer falls back to vanilla without surfacing the error.
# ---------------------------------------------------------------------------


def test_fallback_on_fastokens_adaptation_error(monkeypatch):
    """An adaptation error returns the already-loaded vanilla tokenizer."""
    import renderers.base as rb

    def _boom(*args, **kwargs):
        raise ValueError("simulated fastokens failure: unsupported pre-tokenizer")

    monkeypatch.setattr(rb, "_adapt_tokenizer_with_fastokens", _boom)

    tok = load_tokenizer(_FAST_MODEL)  # default use_fastokens=True
    # The vanilla fallback ran — backend is not a fastokens shim.
    assert "Shim" not in _backend_class_name(tok)
    # Still works.
    assert len(tok.encode("hi", add_special_tokens=False)) > 0


# ---------------------------------------------------------------------------
# Fastokens adaptation emits one INFO log per process, not once per pool slot.
# ---------------------------------------------------------------------------


def test_no_fastokens_stdout_chatter(capsys, caplog):
    """Fast adaptation stays quiet and announces its path once per process."""
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
