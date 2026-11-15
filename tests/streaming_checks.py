"""Shared checks that a renderer's response stream agrees with its parser."""

from __future__ import annotations

import random
import re

from renderers.streaming import TextDelta

CHANNELS = ("reasoning", "content")


def joined(deltas: list[TextDelta], channel: str) -> str:
    return "".join(d.text for d in deltas if d.channel == channel)


def chunkings(token_ids: list[int], seed, *, random_plans: int = 3):
    """One token at a time, all at once, and a few random splits."""
    rng = random.Random(seed)
    plans = [[[t] for t in token_ids], [token_ids]]
    for _ in range(random_plans):
        cuts = sorted(rng.sample(range(len(token_ids) + 1), min(4, len(token_ids) + 1)))
        plans.append(
            [token_ids[a:c] for a, c in zip([0, *cuts], [*cuts, len(token_ids)])]
        )
    return plans


def stream_and_check(
    renderer,
    tokenizer,
    prompt_ids,
    token_ids,
    *,
    seed=0,
    tools=None,
    random_plans=3,
    rewrites_content=lambda parsed: False,
):
    """Stream ``token_ids`` under several chunkings and check the contract.

    Returns the text streamed before ``finish`` and the parsed response.

    Two parser rules look back at text a stream has already sent, so content
    is not checked for them. A completion that ends inside reasoning parses to
    empty content by contract, whatever came before. ``rewrites_content``
    names a format-specific rule of the same kind.
    """
    parsed = renderer.parse_response(token_ids, prompt_ids=prompt_ids, tools=tools)
    checked = (
        ("reasoning",)
        if parsed.reasoning_complete is False or rewrites_content(parsed)
        else CHANNELS
    )
    final = {
        "reasoning": (parsed.reasoning_content or "").strip(),
        "content": (parsed.content or "").strip(),
    }
    context = (
        f"{type(renderer).__name__} "
        f"prompt_tail={tokenizer.decode((prompt_ids or [])[-6:])!r} "
        f"completion={tokenizer.decode(token_ids)!r} ids={token_ids}"
    )
    live = None
    for plan in chunkings(token_ids, seed, random_plans=random_plans):
        stream = renderer.response_stream(prompt_ids=prompt_ids)
        deltas = [delta for chunk in plan for delta in stream.feed(chunk)]
        assert all(delta.text for delta in deltas), context
        streamed = stream.streamed
        for channel in CHANNELS:
            assert joined(deltas, channel) == streamed[channel], context
        for channel in checked:
            # Nothing streamed may ever need to be taken back.
            assert final[channel].startswith(streamed[channel]), (
                channel,
                streamed[channel],
                final[channel],
                context,
            )
        tail = stream.finish(
            reasoning_content=parsed.reasoning_content, content=parsed.content
        )
        for channel in checked:
            assert stream.streamed[channel] == final[channel], context
            assert joined(deltas + tail, channel) == final[channel], context
        # Chunk boundaries are invisible in the text.
        live = live or streamed
        assert streamed == live, context
    return live, parsed


_TEXTS = [
    "Hello",
    " world",
    "\n\n",
    "  ",
    "\n",
    "<",
    "</",
    " < 3",
    "日本語",
    "🙂",
    "é",
    '{"name": "get_weather", "arguments": {"city": "Paris"}}',
    "<th",
    "tool_call",
    "analysis",
    "final",
    "commentary",
    " to=functions.get_weather",
    "thought\n",
    "get_weather",
    "functions.get_weather:0",
    'call:get_weather{city:<|"|>Paris<|"|>}',
    "<function=get_weather>",
    "<parameter=city>",
    "</parameter>",
    "</function>",
    "<arg_key>city</arg_key>",
    "<arg_value>Paris</arg_value>",
    '<invoke name="get_weather">',
    '<parameter name="city">Paris</parameter>',
    "</invoke>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_calls>",
    "</tool_calls>",
    "<minimax:tool_call>",
    "</minimax:tool_call>",
    "<｜DSML｜tool_calls>",
    "</｜DSML｜tool_calls>",
    '<｜DSML｜invoke name="get_weather">',
    "</｜DSML｜invoke>",
    "｜DSML｜",
    "<|python_tag|>",
    '{"name": "get_weather", "parameters": {"city": "Paris"}}',
    "[TOOL_CALLS]",
    "<atem:function_calls>",
    '<atem:invoke name="get_weather">',
    '<atem:parameter name="city">Paris</atem:parameter>',
    "</atem:invoke>",
    "<|tool_call_section_begin|>",
    "<|tool_call_section_end|>",
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>',
    " to=self",
    " to=user",
    " to=get_weather",
    "assistant",
    "```json\n",
    "```",
]
_MARKER_WORDS = re.compile(
    r"think|tool|call|channel|message|start|end|return|eot|eom|python|section|invoke"
    r"|arg|function|param|DSML|constrain|reason|response|turn|im_|sep|bos|eos|User"
    r"|Assistant|answer|observation|content|text|head|role",
    re.IGNORECASE,
)
PREFILLS = [
    "<think>",
    "<think>\n",
    "<think>\n\n</think>\n\n",
    "<|channel|>analysis<|message|>",
    "<|channel|>final<|message|>",
    "<|channel>thought\n",
]


def fuzz_vocabulary(renderer, tokenizer, *, without=()) -> list[list[int]]:
    """Text fragments, every format's tool syntax, and this tokenizer's markers.

    ``without`` names marker tokens to leave out.
    """

    def encode(text):
        return list(tokenizer.encode(text, add_special_tokens=False))

    vocabulary = [encode(text) for text in _TEXTS]
    # Think markers that are not single tokens only matter in exact sequence.
    vocabulary += [encode("<think>"), encode("</think>")] * 4
    added = sorted(
        token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if _MARKER_WORDS.search(token) and token not in without
    )
    if len(added) > 60:
        added = random.Random(0).sample(added, 60)
    # Markers are over-represented so most samples cross several of them.
    vocabulary += [[token_id] for token_id in added] * 2
    vocabulary += [[token_id] for token_id in renderer.get_stop_token_ids()]
    # A lone leading token of a multi-token character, as truncation leaves it.
    for text in ("🙂", "日本語"):
        ids = encode(text)
        if len(ids) > 1:
            vocabulary.append(ids[:1])
    return [ids for ids in vocabulary if ids]


def prompt_variants(renderer, tokenizer, messages) -> list[list[int]]:
    """The generation prompt, alone and with each format's reasoning prefill."""
    prompt = renderer.render_ids(messages, add_generation_prompt=True)
    variants = [prompt]
    for prefill in PREFILLS:
        candidate = [*prompt, *tokenizer.encode(prefill, add_special_tokens=False)]
        if candidate not in variants:
            variants.append(candidate)
    return variants


def structural_pieces(tokenizer, completions: list[list[int]]) -> list[list[int]]:
    """Cut real completions at their marker tokens.

    Each marker becomes a piece, as does each run of text between markers and,
    for short runs, each of its tokens. Recombined at random, these produce
    the format's own headers and delimiters in every order.
    """
    markers = set(tokenizer.get_added_vocab().values())
    pieces: list[list[int]] = []
    for completion in completions:
        run: list[int] = []
        for token_id in [*completion, None]:
            if token_id is not None and token_id not in markers:
                run.append(token_id)
                continue
            if run:
                pieces.append(run)
                if len(run) <= 4:
                    pieces += [[t] for t in run]
                run = []
            if token_id is not None:
                pieces.append([token_id])
    unique: list[list[int]] = []
    for piece in pieces:
        if piece not in unique:
            unique.append(piece)
    return unique


def prefix_prompts(prompt: list[int], completions: list[list[int]]) -> list[list[int]]:
    """``prompt`` extended by the first few tokens of each real completion."""
    return [
        [*prompt, *completion[:size]]
        for completion in completions
        for size in range(1, min(6, len(completion)))
    ]
