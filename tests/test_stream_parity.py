"""Every renderer's response stream against its own parser, on every checkpoint.

Reference continuations come from each checkpoint's chat template (or native
encoder), so they are the completions the model is trained to produce. Random
completions cover everything else a sampler can emit.
"""

from __future__ import annotations

import random
from functools import lru_cache

import pytest
from parity import MODEL_CATALOG, TOOLS, kwarg_combinations
from reference_continuations import ANSWER, REASONING, reference_continuation

from renderers import create_renderer
from renderers.configs import DefaultRendererConfig, config_from_name
from tests.model_assets import load_test_tokenizer
from tests.streaming_checks import (
    fuzz_vocabulary,
    prefix_prompts,
    prompt_variants,
    stream_and_check,
    structural_pieces,
)

pytestmark = pytest.mark.model

MESSAGES = [{"role": "user", "content": "Check the weather."}]

# Checkpoints are visited in groups; reuse within each group only.
_tokenizer = lru_cache(maxsize=1)(load_test_tokenizer)

# The default renderer has no tool parser configured in the catalog.
_REFERENCE_CASES = [
    pytest.param(case, kwargs, calls, id=f"{case.model}-{kwargs}-calls={calls}")
    for case in MODEL_CATALOG
    for kwargs in kwarg_combinations(case)
    for calls in (
        [0]
        if case.resolved_renderer == "default"
        else [0, 1]
        if case.resolved_renderer in {"llama-3", "gpt-oss"}
        else [0, 1, 2]
    )
]


# Parser rules that reinterpret text a stream has already sent. An engine
# never samples a role header mid-completion; Llama's parser restarts the
# body after one. Muse Glimmer rewrites content to the raw completion when a
# tool message carries no complete invocation.
_UNSTREAMABLE_TOKENS = {"llama-3": ("<|end_header_id|>",)}
_REWRITES_CONTENT = {
    "muse-glimmer": lambda parsed: "<atem:" in parsed.content,
}


def _renderer(case, kwargs=None):
    return create_renderer(
        _tokenizer(case.model),
        config_from_name(case.resolved_renderer),
        chat_template_kwargs=kwargs or None,
    )


def _sampled_prefix(renderer, sampled):
    stops = renderer.get_stop_token_ids()
    end = next((i + 1 for i, t in enumerate(sampled) if t in stops), len(sampled))
    return sampled[:end]


@pytest.mark.parametrize("case", MODEL_CATALOG, ids=lambda case: case.model)
def test_every_catalog_renderer_streams(case):
    renderer = _renderer(case)
    prompt = renderer.render_ids(MESSAGES, add_generation_prompt=True)
    assert renderer.response_stream(prompt_ids=prompt) is not None


@pytest.mark.parametrize("case,kwargs,calls", _REFERENCE_CASES)
def test_reference_continuation_streams_all_of_its_text(case, kwargs, calls):
    tok = _tokenizer(case.model)
    renderer = _renderer(case, kwargs)
    tools = TOOLS if calls else None
    prompt, sampled, reasoning = reference_continuation(
        tok, case, kwargs, MESSAGES, tools, calls=calls
    )
    sampled = _sampled_prefix(renderer, sampled)

    live, parsed = stream_and_check(renderer, tok, prompt, sampled, tools=tools)
    assert len(parsed.tool_calls) == calls
    # Everything the model wrote as text is live; only tool calls wait.
    assert live == {"reasoning": reasoning, "content": "" if calls else ANSWER}

    if reasoning:
        cut = next(
            i
            for i in range(1, len(sampled) + 1)
            if tok.decode(sampled[:i]).rstrip().endswith(REASONING)
        )
        live, parsed = stream_and_check(renderer, tok, prompt, sampled[:cut])
        assert parsed.reasoning_complete is False
        assert live == {"reasoning": REASONING, "content": ""}


def _reference_completions(tok, case, kwargs, renderer):
    counts = (
        [0]
        if case.resolved_renderer == "default"
        else [0, 1]
        if case.resolved_renderer in {"llama-3", "gpt-oss"}
        else [0, 1, 2]
    )
    prompt, completions = None, []
    for calls in counts:
        prompt, sampled, _ = reference_continuation(
            tok, case, kwargs, MESSAGES, TOOLS if calls else None, calls=calls
        )
        completions.append(_sampled_prefix(renderer, sampled))
    return prompt, completions


@pytest.mark.parametrize("case", MODEL_CATALOG, ids=lambda case: case.model)
def test_random_completions_never_stream_text_the_parser_disowns(case):
    tok = _tokenizer(case.model)
    without = _UNSTREAMABLE_TOKENS.get(case.resolved_renderer, ())
    banned = {tok.convert_tokens_to_ids(token) for token in without}
    rewrites = _REWRITES_CONTENT.get(case.resolved_renderer, lambda parsed: False)
    opener = tok.encode("<think>", add_special_tokens=False)
    live_chars = final_chars = 0
    # The default configuration in depth, every other one more briefly.
    for kwargs in kwarg_combinations(case):
        renderer = _renderer(case, kwargs)
        reference_prompt, completions = _reference_completions(
            tok, case, kwargs, renderer
        )
        # Generic fragments, and pieces of this format's own completions.
        generic = fuzz_vocabulary(renderer, tok, without=without)
        shaped = [
            piece
            for piece in structural_pieces(tok, completions)
            if not banned & set(piece)
        ]
        prompts = [
            *prompt_variants(renderer, tok, MESSAGES),
            reference_prompt,
            *prefix_prompts(reference_prompt, completions),
        ]
        for seed in range(600 if not kwargs else 40):
            rng = random.Random(f"{case.model}-{kwargs}-{seed}")
            token_ids = [
                token_id
                for _ in range(rng.randint(1, 14))
                for token_id in rng.choice(shaped if rng.random() < 0.6 else generic)
            ]
            if rng.random() < 0.2:
                token_ids = [*opener, *token_ids]
            prompt = rng.choice(prompts)
            tools = TOOLS if rng.random() < 0.5 else None
            live, parsed = stream_and_check(
                renderer,
                tok,
                prompt,
                token_ids,
                seed=seed,
                tools=tools,
                random_plans=1,
                rewrites_content=rewrites,
            )
            if not parsed.tool_calls and not rewrites(parsed):
                live_chars += sum(map(len, live.values()))
                final_chars += len((parsed.reasoning_content or "").strip())
                final_chars += len((parsed.content or "").strip())
    # Without tool calls, most text arrives before the result. The rest is
    # what a stream has to withhold: a possible marker prefix, a replayed
    # header, or (Llama, Inkling) text that may still become a tool call.
    assert live_chars >= 0.55 * final_chars


def _one(renderer_name):
    case = next(c for c in MODEL_CATALOG if c.resolved_renderer == renderer_name)
    return _tokenizer(case.model), _renderer(case)


def test_harmony_without_framing_streams_as_plain_content():
    tok, renderer = _one("gpt-oss")
    token_ids = tok.encode("Plain words, no channels.", add_special_tokens=False)
    live, parsed = stream_and_check(renderer, tok, None, token_ids)
    assert live == {"reasoning": "", "content": "Plain words, no channels."}
    assert parsed.content == "Plain words, no channels."


def test_gemma_replayed_role_header_is_left_to_the_parser():
    tok, renderer = _one("gemma4")
    body = tok.encode("Sunny.", add_special_tokens=False)
    token_ids = [renderer._turn_start, *renderer._model_prefix, *body]
    live, parsed = stream_and_check(renderer, tok, None, token_ids)
    assert parsed.content == "Sunny."
    assert live == {"reasoning": "", "content": ""}


def test_llama_replayed_role_header_is_left_to_the_parser():
    tok, renderer = _one("llama-3")
    token_ids = [
        renderer._start_header,
        *tok.encode("assistant", add_special_tokens=False),
        renderer._end_header,
        *tok.encode("\n\nSunny.", add_special_tokens=False),
    ]
    live, parsed = stream_and_check(renderer, tok, None, token_ids)
    assert parsed.content == "Sunny."
    assert live == {"reasoning": "", "content": ""}


def test_muse_glimmer_keeps_whitespace_before_a_bare_recipient():
    tok, renderer = _one("muse-glimmer")

    def text(value):
        return tok.encode(value, add_special_tokens=False)

    token_ids = [
        renderer._message,
        *text("First. to=user"),
        renderer._message,
        *text("Second."),
        renderer._eom,
    ]
    live, parsed = stream_and_check(renderer, tok, None, token_ids)
    assert parsed.content == "First. Second."
    assert live == {"reasoning": "", "content": "First. Second."}


@pytest.mark.parametrize("label", ["note", "assis"])
def test_muse_glimmer_start_marker_without_a_header_is_body_text(label):
    """``assis`` could still become ``assistant`` until the message marker."""
    tok, renderer = _one("muse-glimmer")

    def text(value):
        return tok.encode(value, add_special_tokens=False)

    token_ids = [
        renderer._message,
        *text("First."),
        renderer._start,
        *text(f"{label} to=user"),
        renderer._message,
        *text("Second."),
        renderer._eom,
    ]
    live, parsed = stream_and_check(renderer, tok, None, token_ids)
    assert parsed.content == f"First.<|start|>{label} Second."
    assert live == {"reasoning": "", "content": parsed.content}


def test_muse_glimmer_joins_separate_reasoning_messages_with_a_newline():
    tok, renderer = _one("muse-glimmer")

    def text(value):
        return tok.encode(value, add_special_tokens=False)

    token_ids = [
        *text(" to=self"),
        renderer._message,
        *text("One."),
        renderer._eom,
        renderer._start,
        *text("assistant to=self"),
        renderer._message,
        *text("Two."),
        renderer._eom,
    ]
    live, parsed = stream_and_check(renderer, tok, None, token_ids)
    assert parsed.reasoning_content == "One.\nTwo."
    assert live == {"reasoning": "One.\nTwo.", "content": ""}


@pytest.mark.parametrize(
    "model, tool_parser",
    [
        ("Qwen/Qwen3-0.6B", "qwen3"),
        ("Qwen/Qwen3.5-0.8B", "qwen3.5"),
        ("THUDM/GLM-4.5-Air", "glm"),
        ("deepseek-ai/DeepSeek-V3", "deepseek-v3"),
    ],
)
@pytest.mark.parametrize("reasoning_parser", [None, "think"])
def test_default_renderer_streams_with_each_bundled_parser(
    model, tool_parser, reasoning_parser
):
    tok = _tokenizer(model)
    renderer = create_renderer(
        tok,
        DefaultRendererConfig(
            tool_parser=tool_parser, reasoning_parser=reasoning_parser
        ),
    )
    vocabulary = fuzz_vocabulary(renderer, tok)
    prompts = prompt_variants(renderer, tok, MESSAGES)
    assert renderer.response_stream(prompt_ids=prompts[0]) is not None
    streamed_text = 0
    for seed in range(250):
        rng = random.Random(f"{model}-{tool_parser}-{seed}")
        token_ids = [
            token_id
            for _ in range(rng.randint(1, 14))
            for token_id in rng.choice(vocabulary)
        ]
        live, _ = stream_and_check(
            renderer, tok, rng.choice(prompts), token_ids, seed=seed, random_plans=1
        )
        streamed_text += sum(map(len, live.values()))
    assert streamed_text > 1000
