"""Unit tests for ``renderers.base._resolve_pool_init_workers``.

The pool defaults to serial construction (``workers=1``) because concurrent
``AutoTokenizer.from_pretrained`` calls have surfaced a rare but catastrophic
``NotImplementedError`` from the transformers Python tokenizer fallback path.
Users can opt back into parallel construction via the
``RENDERERS_POOL_INIT_WORKERS`` env var.
"""

from __future__ import annotations

import pytest

from renderers.base import _resolve_pool_init_workers


def test_default_is_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RENDERERS_POOL_INIT_WORKERS", raising=False)
    assert _resolve_pool_init_workers(32) == 1


def test_env_opts_into_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENDERERS_POOL_INIT_WORKERS", "4")
    assert _resolve_pool_init_workers(32) == 4


def test_clamped_to_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENDERERS_POOL_INIT_WORKERS", "16")
    assert _resolve_pool_init_workers(4) == 4


def test_clamped_to_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENDERERS_POOL_INIT_WORKERS", "32")
    assert _resolve_pool_init_workers(64) == 8


def test_zero_and_negative_fall_back_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENDERERS_POOL_INIT_WORKERS", "0")
    assert _resolve_pool_init_workers(32) == 1
    monkeypatch.setenv("RENDERERS_POOL_INIT_WORKERS", "-2")
    assert _resolve_pool_init_workers(32) == 1


def test_garbage_falls_back_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENDERERS_POOL_INIT_WORKERS", "not-an-int")
    assert _resolve_pool_init_workers(32) == 1
