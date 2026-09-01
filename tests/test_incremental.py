"""Unit tests for ``trim_to_turn_close`` and the bridge contract invariants
that every renderer's ``bridge_to_next_turn`` must uphold.

Cross-renderer parity is validated in test_parity.py and test_roundtrip.py;
this file exercises the shared primitive + protocol-level guarantees with a
fake renderer so we get fast, deterministic coverage of the tricky corners
(truncation opt-in, assistant-in-extension rejection, empty inputs).
"""

import numpy as np

from renderers.base import (
    ParsedResponse,
    RenderedConversation,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.token_arrays import (
    FixedWidthArrayBuilder,
    LOGPROBS_DTYPE,
    TOKEN_IDS_DTYPE,
    empty_array,
)


def _tokens(values: str) -> np.ndarray:
    token_ids = np.fromstring(values, sep=" ", dtype=TOKEN_IDS_DTYPE)
    token_ids.flags.writeable = False
    return token_ids


def _logprobs(values: str) -> np.ndarray:
    logprobs = np.fromstring(values, sep=" ", dtype=LOGPROBS_DTYPE)
    logprobs.flags.writeable = False
    return logprobs


def test_rendered_conversation_keeps_exact_token_tape():
    parsed = ParsedResponse(content="done")
    conv = RenderedConversation(
        prompt_ids=_tokens("1 2"), messages=[{"role": "user", "content": "hi"}]
    )

    next_conv = conv.with_completion(
        _tokens("3 99"),
        completion_logprobs=_logprobs("-0.1 -0.2"),
        parsed_completion=parsed,
    )

    assert np.array_equal(next_conv.token_ids, _tokens("1 2 3 99"))
    assert np.array_equal(next_conv.completion_logprobs, _logprobs("-0.1 -0.2"))
    assert next_conv.parsed_completion is parsed
    assert conv.completion_ids.size == 0


# ---------------------------------------------------------------------------
# trim_to_turn_close
# ---------------------------------------------------------------------------


def test_trim_to_turn_close_trims_to_last_close_in_completion():
    # prev = [1, 2] + [3, 99, 30] (stop token 99). Trim to the 99 boundary,
    # drop the [30] that the model sampled after the stop token.
    result = trim_to_turn_close(_tokens("1 2"), _tokens("3 99 30"), {99})
    assert np.array_equal(result, _tokens("1 2 3 99"))


def test_trim_to_turn_close_ignores_close_in_prompt():
    # A stop-token id that happens to appear in prev_prompt (as structural
    # template scaffolding) must not be treated as a turn boundary.
    result = trim_to_turn_close(_tokens("99 1"), _tokens("3 4 5"), {99})
    assert result is None


def test_trim_to_turn_close_synthesises_when_truncated():
    # Truncation: no stop token in completion. With synthesize_close=99,
    # append the synthetic close and return prev + [99].
    result = trim_to_turn_close(
        _tokens("1 2"), _tokens("3 4 5"), {99}, synthesize_close=99
    )
    assert np.array_equal(result, _tokens("1 2 3 4 5 99"))


def test_trim_to_turn_close_returns_none_on_truncation_without_synth():
    # Truncation without synth opt-in → caller falls back to fresh render.
    result = trim_to_turn_close(_tokens("1 2"), _tokens("3 4 5"), {99})
    assert result is None


def test_trim_to_turn_close_accepts_multiple_close_tokens():
    # Multiple close tokens: pick the LAST one that appears in completion.
    result = trim_to_turn_close(
        _tokens("1"), _tokens("3 50 4 99 30"), {50, 99}
    )
    assert np.array_equal(result, _tokens("1 3 50 4 99"))


# ---------------------------------------------------------------------------
# reject_assistant_in_extension
# ---------------------------------------------------------------------------


def test_reject_assistant_in_extension_true_when_assistant_present():
    assert reject_assistant_in_extension(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
    )


def test_reject_assistant_in_extension_false_for_tool_user_only():
    assert not reject_assistant_in_extension(
        [{"role": "tool", "content": "result"}, {"role": "user", "content": "next"}]
    )


# ---------------------------------------------------------------------------
# Contract tests against a minimal fake renderer
# ---------------------------------------------------------------------------


class _FakeRenderer:
    """Minimal Renderer whose bridge exercises the contract:

    - Extension token is a single sentinel ID 42.
    - ``<|im_end|>`` ≡ 99 is the canonical close.
    """

    def __init__(self):
        self._im_end = 99

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        raise NotImplementedError

    def parse_response(self, token_ids):
        return ParsedResponse(content="")

    def get_stop_token_ids(self):
        return _tokens(str(self._im_end))

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        if previous_prompt_ids.size == 0 or not new_messages:
            return None
        if reject_assistant_in_extension(new_messages):
            return None
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._im_end},
            synthesize_close=self._im_end,
        )
        if previous_ids is None:
            return None
        builder = FixedWidthArrayBuilder(
            TOKEN_IDS_DTYPE, initial_capacity=previous_ids.size + 1
        )
        builder.extend(previous_ids)
        builder.append(42)
        return builder.finish()


def test_fake_bridge_extends_verbatim_on_clean_stop():
    renderer = _FakeRenderer()
    prev_prompt = _tokens("1 2")
    prev_completion = _tokens("3 99")
    result = renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, [{"role": "user", "content": "next"}]
    )
    assert np.array_equal(result, _tokens("1 2 3 99 42"))
    expected_prefix = np.concatenate((prev_prompt, prev_completion))
    assert np.array_equal(result[: expected_prefix.size], expected_prefix)


def test_fake_bridge_synthesises_on_truncation():
    renderer = _FakeRenderer()
    result = renderer.bridge_to_next_turn(
        _tokens("1 2"),
        _tokens("3 4 5"),
        [{"role": "user", "content": "next"}],
    )
    # Truncated prev; synth-close appends 99 then extension 42.
    assert np.array_equal(result, _tokens("1 2 3 4 5 99 42"))


def test_fake_bridge_rejects_assistant_in_extension():
    renderer = _FakeRenderer()
    result = renderer.bridge_to_next_turn(
        _tokens("1"), _tokens("99"), [{"role": "assistant", "content": "x"}]
    )
    assert result is None


def test_fake_bridge_rejects_empty_inputs():
    renderer = _FakeRenderer()
    assert (
        renderer.bridge_to_next_turn(
            empty_array(TOKEN_IDS_DTYPE),
            _tokens("99"),
            [{"role": "user", "content": "x"}],
        )
        is None
    )
    assert renderer.bridge_to_next_turn(_tokens("1"), _tokens("99"), []) is None
