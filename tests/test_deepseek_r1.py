"""DeepSeek-R1 renderer: the reasoning variant of the DeepSeek format.

General byte parity lives in ``test_parity.py``. These tests pin the behaviors
that distinguish R1 from V3: the ``<think>`` generation prompt and the
stripping of ``</think>`` from historical assistant turns.
"""

from functools import lru_cache

from renderers import (
    DeepSeekR1Renderer,
    DeepSeekV3Renderer,
    create_renderer,
)
from renderers.base import load_tokenizer


@lru_cache
def _r1():
    tok = load_tokenizer("deepseek-ai/DeepSeek-R1")
    return tok, create_renderer(tok)


@lru_cache
def _v3():
    tok = load_tokenizer("deepseek-ai/DeepSeek-V3")
    return tok, create_renderer(tok)


def test_auto_detection_picks_the_right_renderer():
    _, r1 = _r1()
    _, v3 = _v3()
    assert isinstance(r1, DeepSeekR1Renderer)
    assert isinstance(v3, DeepSeekV3Renderer)


def test_generation_prompt_differs():
    """R1 prefills ``<think>`` to trigger reasoning; V3 does not."""
    msgs = [{"role": "user", "content": "hi"}]
    tr1, r1 = _r1()
    tv3, v3 = _v3()

    r1_text = tr1.decode(r1.render_ids(msgs, add_generation_prompt=True))
    v3_text = tv3.decode(v3.render_ids(msgs, add_generation_prompt=True))

    # R1 prefills <think> to trigger reasoning; V3 emits a bare assistant
    # turn. (Exact byte parity vs apply_chat_template is the gen_prompt case
    # in the parity matrix above; here we just pin the V3/R1 distinction —
    # decode trims the trailing "\n" so we don't match on it.)
    assert "<think>" in r1_text and r1_text.rstrip().endswith("<think>")
    assert "<think>" not in v3_text


def test_r1_strips_reasoning_from_history():
    """A historical assistant turn carrying an inline ``<think>…</think>``
    trace renders only the post-``</think>`` answer, byte-identical to the R1
    chat template's ``content.split('</think>')[-1]``.
    """
    tok, r1 = _r1()
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>private reasoning</think>The answer."},
        {"role": "user", "content": "q2"},
    ]
    got = r1.render_ids(msgs)
    expected = list(tok.apply_chat_template(msgs, tokenize=True, return_dict=False))

    assert got == expected
    # Reasoning must not survive into the rendered history.
    assert "private reasoning" not in tok.decode(got)


def test_v3_emits_content_verbatim_ignoring_reasoning():
    """V3 (non-reasoning) ignores ``reasoning_content`` — matching its
    template, which only reads ``content``."""
    tok, v3 = _v3()
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "reasoning_content": "should be ignored", "content": "4"},
    ]
    got = v3.render_ids(msgs)
    expected = list(tok.apply_chat_template(msgs, tokenize=True, return_dict=False))

    assert got == expected
    assert "should be ignored" not in tok.decode(got)
