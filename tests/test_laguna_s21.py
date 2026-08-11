"""Laguna-S-2.1 focused tests.

S-2.1 shares XS-2.1's tokenizer and token format, so the shared matrices
(conftest / config-parity) already assert byte parity on the common shapes
via :class:`LagunaS21Renderer`. This file pins the two behaviours that make
S-2.1 *differ* from XS-2.1 — and that the shared barrage can't reach because
it never sets the relevant kwargs:

- ``enable_thinking`` defaults to ``True`` (XS-2.1 defaults ``False``), so the
  auto-resolved renderer renders reasoning by default, and
- the new ``preserve_thinking`` kwarg keeps historical ``<think>`` blocks even
  while ``enable_thinking`` is ``False`` — the template gates reasoning display
  on ``enable_thinking or preserve_thinking``.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import LagunaS21RendererConfig
from renderers.laguna_s21 import LagunaS21Renderer

_MODEL = "poolside/Laguna-S-2.1"


@lru_cache(maxsize=None)
def _tok():
    return load_tokenizer(_MODEL)


def _renderer(**config_kwargs) -> LagunaS21Renderer:
    renderer = create_renderer(_tok(), LagunaS21RendererConfig(**config_kwargs))
    assert isinstance(renderer, LagunaS21Renderer)
    return renderer


def _expected(msgs, *, add_generation_prompt=False, **template_kwargs):
    return list(
        _tok().apply_chat_template(
            msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
    )


def test_auto_resolves_to_s21_renderer_with_thinking_on_by_default():
    """``poolside/Laguna-S-2.1`` resolves to :class:`LagunaS21Renderer`, and
    its config defaults ``enable_thinking=True`` / ``preserve_thinking=False``
    — matching S-2.1's template defaults (unlike XS-2.1's thinking-off)."""
    r = create_renderer(_tok())  # auto-resolve via MODEL_RENDERER_MAP
    assert isinstance(r, LagunaS21Renderer)
    assert r.config.name == "laguna-s-2.1"
    assert r.config.enable_thinking is True
    assert r.config.preserve_thinking is False


def test_default_renders_empty_think_wrapper():
    """With thinking on by default, a reasoning-free assistant turn still
    opens with the empty ``<think></think>`` wrapper (this is exactly the
    render_ids divergence from the XS-2.1 renderer)."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    r = _renderer()  # defaults: enable_thinking=True
    ours = r.render_ids(msgs)
    assert ours == _expected(msgs)
    assert "<assistant><think></think>4</assistant>" in _tok().decode(ours)


def test_preserve_thinking_keeps_history_when_thinking_off():
    """``enable_thinking=False`` alone drops reasoning, but with
    ``preserve_thinking=True`` the historical ``<think>{reasoning}</think>``
    survives — matching the template's ``enable_thinking or preserve_thinking``
    gate."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Simple arithmetic", "content": "4"},
    ]
    r = _renderer(enable_thinking=False, preserve_thinking=True)
    ours = r.render_ids(msgs)
    assert ours == _expected(msgs, enable_thinking=False, preserve_thinking=True)
    assert "<think>Simple arithmetic</think>" in _tok().decode(ours)


def test_no_preserve_thinking_drops_history_when_thinking_off():
    """With both flags off the gate collapses to ``enable_thinking`` and the
    turn opens with a bare ``</think>``, dropping the reasoning — identical to
    the XS-2.1 renderer's default."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Simple arithmetic", "content": "4"},
    ]
    r = _renderer(enable_thinking=False, preserve_thinking=False)
    ours = r.render_ids(msgs)
    assert ours == _expected(msgs, enable_thinking=False, preserve_thinking=False)
    assert "Simple arithmetic" not in _tok().decode(ours)


def test_all_thinking_flag_combos_match_template():
    """Byte parity against ``apply_chat_template`` for every
    ``enable_thinking`` × ``preserve_thinking`` combination, over a multi-turn
    conversation carrying historical reasoning."""
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "deduce", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "reasoning_content": "more", "content": "A2"},
    ]
    for enable_thinking, preserve_thinking in product((False, True), repeat=2):
        r = _renderer(
            enable_thinking=enable_thinking, preserve_thinking=preserve_thinking
        )
        for add_generation_prompt in (False, True):
            assert r.render_ids(
                msgs, add_generation_prompt=add_generation_prompt
            ) == _expected(
                msgs,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            ), (enable_thinking, preserve_thinking, add_generation_prompt)
