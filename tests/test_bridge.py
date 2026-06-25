"""Cross-renderer bridge contract tests.

Verifies for every hand-coded renderer that ``bridge_to_next_turn``:

  1. Extends ``prev_prompt_ids + prev_completion_ids`` verbatim.
  2. Refuses assistant-role messages in ``new_messages``.
  3. Synthesises the canonical turn close on truncation.
  4. On clean stop, the extension is compatible with a fresh render:
     decoding the extension should contain the new-message content and a
     generation-prompt-looking tail.

DefaultRenderer is excluded because it intentionally returns None (it
doesn't know its template's close). That path is exercised by the caller
fallback in ``test_renderer_e2e.py``.
"""

from __future__ import annotations

from functools import lru_cache

import pytest


# (HF model name, renderer name) — one representative per renderer class.
_BRIDGE_MODELS = [
    ("Qwen/Qwen3-8B", "auto"),
    ("Qwen/Qwen3.5-9B", "auto"),
    ("Qwen/Qwen3.6-35B-A3B", "auto"),
    ("zai-org/GLM-5", "auto"),
    ("zai-org/GLM-5.1", "auto"),
    ("THUDM/GLM-4.5-Air", "auto"),
    ("MiniMaxAI/MiniMax-M2.5", "auto"),
    ("moonshotai/Kimi-K2-Instruct", "auto"),
    ("moonshotai/Kimi-K2.5", "auto"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "auto"),
    ("unsloth/Llama-3.2-1B-Instruct", "llama-3"),
    ("openai/gpt-oss-20b", "gpt-oss"),
]


@lru_cache(maxsize=None)
def _load(model_name: str, renderer_name: str):
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import config_from_name

    tok = load_tokenizer(model_name)
    return tok, create_renderer(tok, config_from_name(renderer_name))


def pytest_generate_tests(metafunc):
    if "br_model" in metafunc.fixturenames:
        metafunc.parametrize(
            "br_model,br_renderer_name",
            _BRIDGE_MODELS,
            ids=[m for m, _ in _BRIDGE_MODELS],
        )


@pytest.fixture
def br_tokenizer(br_model, br_renderer_name):
    return _load(br_model, br_renderer_name)[0]


@pytest.fixture
def br_renderer(br_model, br_renderer_name):
    return _load(br_model, br_renderer_name)[1]


@pytest.fixture
def br_renderer_all(br_model, br_renderer_name):
    """Renderer forced to ``thinking_retention="all"``.

    The verbatim-extension mechanic tests below cross a user-query
    boundary. For thinking models the template drops a past block's
    thinking there, so the faithful bridge declines (covered by
    ``test_bridge_declines_across_user_query_when_template_drops_thinking``).
    ``"all"`` keeps thinking on every path, isolating the pure extension
    mechanic from the retention policy across all renderers.
    """
    from renderers import create_renderer

    tok, base = _load(br_model, br_renderer_name)
    cfg = base.config.model_copy(update={"thinking_retention": "all"})
    return create_renderer(tok, cfg)


def _simulate_prior_turn(renderer, assistant=None):
    """Build a (prev_prompt, prev_completion) pair that a real rollout
    would produce for a one-turn prior with a clean stop.

    Strategy: render ``[system, user]`` with gen_prompt=True to get
    prev_prompt, then render ``[system, user, assistant]`` without
    gen_prompt, and take the diff as prev_completion. We then trim
    prev_completion to the last close token so it matches what vLLM
    actually hands back (vLLM stops at the close token and excludes the
    trailing template scaffolding). Pass ``assistant`` to override the
    default no-thinking turn (e.g. one carrying ``reasoning_content``).
    """
    prior = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ]
    if assistant is None:
        assistant = [{"role": "assistant", "content": "Hello!"}]

    prev_prompt = renderer.render_ids(prior, add_generation_prompt=True)
    full_with_assistant = renderer.render_ids(
        prior + assistant, add_generation_prompt=False
    )
    prev_completion = list(full_with_assistant[len(prev_prompt) :])

    # Trim past any trailing scaffolding the template emits AFTER the
    # close (e.g. chatml's trailing ``\n``). vLLM only returns tokens up
    # to and including the close itself.
    stop_ids = set(renderer.get_stop_token_ids())
    last_close = -1
    for i in range(len(prev_completion) - 1, -1, -1):
        if prev_completion[i] in stop_ids:
            last_close = i
            break
    if last_close >= 0:
        prev_completion = prev_completion[: last_close + 1]

    return prev_prompt, prev_completion


def test_bridge_extends_prev_verbatim_on_clean_stop(br_renderer_all, br_model):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer_all)
    new_messages = [{"role": "user", "content": "What's 2+2?"}]

    bridged = br_renderer_all.bridge_to_next_turn(
        prev_prompt, prev_completion, new_messages
    )
    assert bridged is not None, f"{br_model}: bridge returned None on clean stop"
    bridged_ids = bridged.token_ids

    prev = prev_prompt + prev_completion
    assert bridged_ids[: len(prev)] == prev, (
        f"{br_model}: bridged does NOT extend prev_prompt + prev_completion"
    )
    assert len(bridged_ids) > len(prev), (
        f"{br_model}: bridge did not emit any extension tokens"
    )


def test_bridge_rejects_assistant_in_extension(br_renderer):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer)
    assert (
        br_renderer.bridge_to_next_turn(
            prev_prompt,
            prev_completion,
            [{"role": "assistant", "content": "forbidden"}],
        )
        is None
    )


def test_bridge_rejects_empty_prev_or_new(br_renderer):
    _, prev_completion = _simulate_prior_turn(br_renderer)
    assert (
        br_renderer.bridge_to_next_turn(
            [], prev_completion, [{"role": "user", "content": "x"}]
        )
        is None
    )
    prev_prompt, _ = _simulate_prior_turn(br_renderer)
    assert br_renderer.bridge_to_next_turn(prev_prompt, prev_completion, []) is None


def test_bridge_synthesises_close_on_truncation(br_renderer_all, br_model):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer_all)
    # Drop the final close token to simulate a max_tokens truncation.
    prev_completion_trunc = prev_completion[:-1] if prev_completion else prev_completion
    if len(prev_completion_trunc) == 0:
        pytest.skip(
            f"{br_model}: simulated prior had no completion tokens — can't truncate"
        )

    bridged = br_renderer_all.bridge_to_next_turn(
        prev_prompt,
        prev_completion_trunc,
        [{"role": "user", "content": "What's 2+2?"}],
    )
    assert bridged is not None, (
        f"{br_model}: bridge returned None on truncation; expected synth-close"
    )
    bridged_ids = bridged.token_ids
    prev_trunc = prev_prompt + prev_completion_trunc
    assert bridged_ids[: len(prev_trunc)] == prev_trunc, (
        f"{br_model}: truncated-prior bridge did not keep prev tokens verbatim"
    )
    assert len(bridged_ids) > len(prev_trunc), (
        f"{br_model}: synth-close produced no extra tokens"
    )


def test_bridge_extension_includes_new_message_text(
    br_renderer_all, br_tokenizer, br_model
):
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer_all)
    new_messages = [{"role": "user", "content": "HELLO_SENTINEL_XYZ"}]

    bridged = br_renderer_all.bridge_to_next_turn(
        prev_prompt, prev_completion, new_messages
    )
    assert bridged is not None
    ext = bridged.token_ids[len(prev_prompt) + len(prev_completion) :]
    decoded = br_tokenizer.decode(ext, skip_special_tokens=False)
    assert "HELLO_SENTINEL_XYZ" in decoded, (
        f"{br_model}: new-message content missing from extension; got {decoded!r}"
    )


def test_bridge_declines_across_user_query_when_template_drops_thinking():
    """Qwen3's template drops a past block's thinking once a new user turn
    arrives. The resolved ``tool_cycle`` bridge policy therefore treats a
    new user query as a hard re-render boundary, independent of whether the
    prior token stream happens to contain sampled thinking:

      - new user query + retention="tool_cycle" -> decline,
        and the fallback re-render equals ``apply_chat_template``.
      - thinking_retention="all" keeps thinking on every path -> extend.
      - a tool response (in-flight cycle, no new query) keeps thinking in
        the template too -> extend.
      - no prior thinking + new user query -> decline; no marker lookback.
    """
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import Qwen3RendererConfig

    tok = load_tokenizer("Qwen/Qwen3-8B")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    u1 = {"role": "user", "content": "What is 2+2?"}
    u2 = {"role": "user", "content": "Multiply that by 3."}
    tool = {"role": "tool", "content": "ok"}
    think = "<think>\n2 plus 2 is 4.\n</think>\n\n4"

    def prior(r, asst_text):
        p = r.render_ids([u1], add_generation_prompt=True)
        completion = tok.encode(asst_text, add_special_tokens=False) + [im_end]
        return p, completion

    # new user query + prior thinking + default retention -> decline
    r = create_renderer(tok, Qwen3RendererConfig())
    p, comp = prior(r, think)
    assert r.bridge_to_next_turn(p, comp, [u2]) is None
    # ...and the caller's faithful re-render matches the chat template
    hist = [u1, {"role": "assistant", "content": think}, u2]
    rendered = tok.decode(r.render_ids(hist, add_generation_prompt=True))
    assert rendered == tok.apply_chat_template(
        hist, tokenize=False, add_generation_prompt=True
    )

    # thinking_retention="all" keeps thinking everywhere -> extend
    r_all = create_renderer(tok, Qwen3RendererConfig(thinking_retention="all"))
    p, comp = prior(r_all, think)
    assert r_all.bridge_to_next_turn(p, comp, [u2]) is not None

    # tool response continues the in-flight cycle (no new query) -> extend
    p, comp = prior(r, think)
    assert r.bridge_to_next_turn(p, comp, [tool]) is not None

    # tool_cycle is a user-query-boundary policy; it does not scan prior tokens.
    p, comp = prior(r, "4")
    assert r.bridge_to_next_turn(p, comp, [u2]) is None


# Renderers whose default/effective bridge policy declines at a new user-query
# boundary. The exact query-boundary predicate can still be renderer-specific
# (for example Qwen's folded ``<tool_response>`` user messages).
# Non-thinking models (llama, deepseek-v3) are out of scope for this check.
_GUARDED_THINKING_MODELS = {
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-35B-A3B",
    "zai-org/GLM-5",
    "zai-org/GLM-5.1",
    "THUDM/GLM-4.5-Air",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "MiniMaxAI/MiniMax-M2.5",
    "moonshotai/Kimi-K2.5",
    "openai/gpt-oss-20b",
}


def test_bridge_declines_across_user_turn_when_thinking_present(br_renderer, br_model):
    """Across every guarded thinking renderer: a prior turn carrying
    ``reasoning_content`` makes the bridge decline a new *user* query (the
    template would strip that thinking) but still extend a *tool* response
    (in-flight cycle keeps it). Non-guarded models are skipped."""
    if br_model not in _GUARDED_THINKING_MODELS:
        pytest.skip(f"{br_model}: no bridge thinking-guard")

    asst = [
        {
            "role": "assistant",
            "reasoning_content": "Let me think.",
            "content": "",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
        }
    ]
    prev_prompt, prev_completion = _simulate_prior_turn(br_renderer, asst)

    declined = br_renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, [{"role": "user", "content": "next"}]
    )
    assert declined is None, f"{br_model}: expected faithful decline across a user turn"

    extended = br_renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, [{"role": "tool", "content": "result"}]
    )
    assert extended is not None, f"{br_model}: should still bridge within a tool cycle"


def test_bridge_keeps_thinking_when_history_kwarg_disables_truncation():
    """GLM ``clear_thinking=False`` / Nemotron ``truncate_history_thinking=
    False`` keep all past thinking (the template doesn't strip it), so the
    bridge must NOT decline across a user turn — declining would re-render and
    re-tokenize model-sampled thinking bytes."""
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import GLM5RendererConfig, Nemotron3RendererConfig

    asst = [
        {"role": "assistant", "reasoning_content": "Let me think.", "content": "Hi"}
    ]
    cases = [
        ("zai-org/GLM-5", GLM5RendererConfig(clear_thinking=False)),
        (
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            Nemotron3RendererConfig(truncate_history_thinking=False),
        ),
    ]
    for model, cfg in cases:
        r = create_renderer(load_tokenizer(model), cfg)
        prev_prompt, prev_completion = _simulate_prior_turn(r, asst)
        bridged = r.bridge_to_next_turn(
            prev_prompt, prev_completion, [{"role": "user", "content": "next"}]
        )
        assert bridged is not None, (
            f"{model}: bridge must keep verbatim when the template keeps all thinking"
        )
