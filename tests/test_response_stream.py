"""Incremental reasoning/content splitting of an arriving completion.

The first half drives :class:`ResponseStream` with a byte-level fake
tokenizer, so every hold-back rule is exercised without model assets. The
second half checks the real Qwen3 and DeepSeek V4 grammars against their own
``parse_response`` over hand-written and randomly generated completions.
"""

from __future__ import annotations

import logging
import random
from functools import lru_cache

import pytest

from renderers import DeepSeekV4Renderer, DeepSeekV4RendererConfig, Qwen3Renderer
from renderers.streaming import (
    ResponseStream,
    StreamGrammar,
    TextDelta,
    remaining_deltas,
)
from tests.model_assets import load_test_tokenizer

# ── A byte-level tokenizer: ids < 256 are raw bytes, the rest are markers ──

THINK, END_THINK, TOOL, END_TOOL, EOS = 300, 301, 302, 303, 304
_SPECIALS = {
    THINK: "<think>",
    END_THINK: "</think>",
    TOOL: "<tool_call>",
    END_TOOL: "</tool_call>",
    EOS: "<eos>",
}


class ByteTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        out, run = [], bytearray()
        for token_id in token_ids:
            if token_id < 256:
                run.append(token_id)
                continue
            out.append(run.decode("utf-8", errors="replace"))
            run.clear()
            out.append(_SPECIALS[token_id])
        out.append(run.decode("utf-8", errors="replace"))
        return "".join(out)


def b(text: str) -> list[int]:
    return list(text.encode("utf-8"))


REASONING_FIRST = StreamGrammar(
    stop_ids=frozenset({EOS}),
    think_open_ids=frozenset({THINK}),
    think_close_ids=frozenset({END_THINK}),
    tool_ids=frozenset({TOOL}),
)
REENTRANT = StreamGrammar(
    stop_ids=frozenset({EOS}),
    think_open_ids=frozenset({THINK}),
    think_close_ids=frozenset({END_THINK}),
    tool_markers=("<|calls|>", "<|invoke"),
    reentrant_reasoning=True,
)


def new_stream(grammar=REASONING_FIRST, *, reasoning=False) -> ResponseStream:
    return ResponseStream(ByteTokenizer(), grammar, reasoning=reasoning)


def feed_each(stream: ResponseStream, token_ids: list[int]) -> list[TextDelta]:
    """Feed one token at a time; return every delta in order."""
    return [delta for token_id in token_ids for delta in stream.feed([token_id])]


def joined(deltas: list[TextDelta], channel: str) -> str:
    return "".join(d.text for d in deltas if d.channel == channel)


def test_content_arrives_as_soon_as_each_token_settles_it():
    stream = new_stream()
    assert stream.feed(b("Hel")) == [TextDelta("content", "Hel")]
    assert stream.feed(b("lo")) == [TextDelta("content", "lo")]
    assert stream.streamed == {"reasoning": "", "content": "Hello"}


def test_one_feed_merges_adjacent_text_into_one_delta_per_channel_run():
    stream = new_stream(reasoning=True)
    deltas = stream.feed([*b("why"), END_THINK, *b("so"), *b(" be it")])
    assert deltas == [TextDelta("reasoning", "why"), TextDelta("content", "so be it")]


def test_leading_whitespace_is_dropped_and_trailing_whitespace_waits():
    stream = new_stream()
    assert stream.feed(b("\n\n  ")) == []
    assert stream.feed(b("a ")) == [TextDelta("content", "a")]
    assert stream.feed(b(" \n")) == []
    # The withheld whitespace is released once text follows it.
    assert stream.feed(b("b")) == [TextDelta("content", "  \nb")]
    assert stream.feed(b("  ")) == []
    assert stream.finish(reasoning_content=None, content="a  \nb") == []


def test_prompt_opened_reasoning_then_content():
    stream = new_stream(reasoning=True)
    deltas = feed_each(stream, [*b(" think \n"), END_THINK, *b("\n\nanswer"), EOS])
    assert joined(deltas, "reasoning") == "think"
    assert joined(deltas, "content") == "answer"
    # Reasoning is complete before the first content delta.
    channels = [d.channel for d in deltas]
    assert channels == sorted(channels, key=["reasoning", "content"].index)


def test_initial_opener_after_whitespace_starts_reasoning():
    stream = new_stream()
    deltas = feed_each(stream, [*b("\n"), THINK, *b("r"), THINK, *b("r"), END_THINK])
    deltas += feed_each(stream, b("c"))
    # A repeated opener inside the region is dropped from its text.
    assert joined(deltas, "reasoning") == "rr"
    assert joined(deltas, "content") == "c"


def test_reasoning_first_think_markers_after_content_are_literal():
    stream = new_stream()
    deltas = feed_each(stream, [*b("a"), THINK, *b("b"), END_THINK, *b("c")])
    assert joined(deltas, "reasoning") == ""
    assert joined(deltas, "content") == "a<think>b</think>c"


def test_reasoning_first_close_without_a_region_is_literal_content():
    stream = new_stream()
    deltas = feed_each(stream, [END_THINK, *b("x"), THINK])
    assert joined(deltas, "content") == "</think>x<think>"


def test_reasoning_first_markers_after_the_region_closes_are_literal():
    stream = new_stream(reasoning=True)
    deltas = feed_each(stream, [*b("r"), END_THINK, *b("a"), THINK, END_THINK])
    assert joined(deltas, "reasoning") == "r"
    assert joined(deltas, "content") == "a<think></think>"


def test_reentrant_reasoning_reopens_and_drops_its_markers():
    stream = new_stream(REENTRANT)
    deltas = feed_each(
        stream,
        [*b("A "), THINK, *b("r1"), THINK, *b(" r2"), END_THINK, END_THINK, *b("B")],
    )
    assert joined(deltas, "reasoning") == "r1 r2"
    # Whitespace withheld before the detour is released when content resumes.
    assert joined(deltas, "content") == "A B"


@pytest.mark.parametrize("state", ["content", "reasoning"])
def test_atomic_tool_opener_silences_the_stream(state):
    stream = new_stream(reasoning=state == "reasoning")
    before = feed_each(stream, [*b("before \n"), TOOL])
    after = feed_each(stream, [*b('{"name": "f"}'), END_TOOL, *b("after"), EOS])
    assert joined(before, state) == "before"
    assert {d.channel for d in before} == {state}
    assert after == []


def test_finish_delivers_what_the_parse_holds_beyond_the_stream():
    stream = new_stream(reasoning=True)
    feed_each(stream, [*b("draft "), TOOL, *b("{}"), END_TOOL, END_THINK, *b("done")])
    # A close after the tool markup makes all of it reasoning.
    tail = stream.finish(
        reasoning_content="draft <tool_call>{}</tool_call>", content="done"
    )
    assert tail == [
        TextDelta("reasoning", " <tool_call>{}</tool_call>"),
        TextDelta("content", "done"),
    ]
    assert stream.streamed == {
        "reasoning": "draft <tool_call>{}</tool_call>",
        "content": "done",
    }


def test_stop_token_ends_the_stream():
    stream = new_stream()
    assert feed_each(stream, [*b("a"), EOS, *b("b"), THINK]) == [
        TextDelta("content", "a")
    ]
    assert stream.feed(b("c")) == []


@pytest.mark.parametrize("text", ["é", "日本語", "🙂", "a🙂b", "👩‍👩‍👧"])
def test_multibyte_characters_are_never_emitted_in_pieces(text):
    stream = new_stream()
    deltas = feed_each(stream, b(text))
    assert all("\ufffd" not in d.text for d in deltas)
    assert joined(deltas, "content") == text
    # One delta per completed code point, none for its leading bytes.
    assert len(deltas) == len(text)


def test_an_incomplete_character_at_the_end_is_left_to_the_final_parse():
    stream = new_stream()
    assert feed_each(stream, [*b("ok"), *b("🙂")[:2]]) == [
        TextDelta("content", "o"),
        TextDelta("content", "k"),
    ]
    assert stream.finish(reasoning_content=None, content="ok\ufffd") == [
        TextDelta("content", "\ufffd")
    ]


def test_a_withheld_byte_before_an_opener_is_text_so_the_opener_is_literal():
    stream = new_stream()
    deltas = feed_each(stream, [*b("\n"), 0xF0, THINK, *b("r"), END_THINK])
    assert joined(deltas, "reasoning") == ""
    assert joined(deltas, "content") == "\ufffd<think>r</think>"


def test_a_withheld_byte_is_released_when_its_channel_ends():
    stream = new_stream(reasoning=True)
    deltas = feed_each(stream, [*b("r"), 0xF0, END_THINK, *b("c")])
    assert joined(deltas, "reasoning") == "r\ufffd"
    assert joined(deltas, "content") == "c"


class WordTokenizer:
    """SentencePiece-style: a token's leading space depends on its position."""

    WORDS = {1: "▁Hello", 2: "▁wor", 3: "ld", 4: "▁again"}

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = "".join(self.WORDS.get(t) or _SPECIALS[t] for t in token_ids)
        return pieces.replace("▁", " ").removeprefix(" ")


def test_tokens_are_decoded_in_the_context_of_their_run():
    stream = ResponseStream(WordTokenizer(), REASONING_FIRST, reasoning=True)
    deltas = feed_each(stream, [1, 2, 3, END_THINK, 4, 1])
    assert [d.text for d in deltas] == ["Hello", " wor", "ld", "again", " Hello"]
    # A run's first token has no leading space, as in the parser's decode.
    assert stream.streamed == {"reasoning": "Hello world", "content": "again Hello"}


def test_invalid_bytes_are_emitted_once_valid_text_follows():
    stream = new_stream()
    deltas = feed_each(stream, [*b("a"), 0xFF, *b("b")])
    assert joined(deltas, "content") == "a\ufffdb"


def test_text_tool_marker_split_across_tokens_stops_the_stream():
    stream = new_stream(REENTRANT)
    deltas = feed_each(stream, b("see \n<|calls|><|invoke name=x>") + b("tail"))
    assert deltas == [TextDelta("content", c) for c in "see"]


def test_text_tool_marker_prefix_is_withheld_until_it_resolves():
    stream = new_stream(REENTRANT)
    assert joined(feed_each(stream, b("a <|")), "content") == "a"
    # Not a marker after all: the withheld text is released.
    assert joined(feed_each(stream, b("x")), "content") == " <|x"
    assert joined(feed_each(stream, b(" <|inv")), "content") == ""
    assert feed_each(stream, b("oke")) == []
    assert stream.feed(b("more")) == []
    assert stream.streamed["content"] == "a <|x"


def test_longest_possible_marker_prefix_is_withheld():
    stream = new_stream(REENTRANT)
    # "<|" starts both markers; "<|c" only the longer one.
    assert joined(stream.feed(b("q<|c")), "content") == "q"
    assert joined(stream.feed(b("alm")), "content") == "<|calm"


def test_text_tool_marker_does_not_match_across_a_think_marker():
    stream = new_stream(REENTRANT, reasoning=True)
    deltas = feed_each(stream, [*b("r<|"), END_THINK, *b("calls|>c")])
    assert joined(deltas, "reasoning") == "r<|"
    assert joined(deltas, "content") == "calls|>c"


def test_text_tool_marker_in_reasoning_stops_the_stream():
    stream = new_stream(REENTRANT, reasoning=True)
    deltas = feed_each(stream, [*b("r <|invoke"), END_THINK, *b("c")])
    assert deltas == [TextDelta("reasoning", "r")]


def test_chunking_does_not_change_the_streamed_text():
    token_ids = [
        *b(" \n"),
        THINK,
        *b("plan: 日本 🙂 \n"),
        END_THINK,
        *b("\n\nfirst <| then <|c "),
        THINK,
        *b("more"),
        END_THINK,
        *b(" last\n<|invoke name=f>"),
        *b("args"),
        EOS,
    ]
    expected = None
    for first in range(len(token_ids) + 1):
        for second in range(first, len(token_ids) + 1, 7):
            stream = new_stream(REENTRANT)
            for chunk in (
                token_ids[:first],
                token_ids[first:second],
                token_ids[second:],
            ):
                stream.feed(chunk)
            expected = expected or stream.streamed
            assert stream.streamed == expected, (first, second)
    # Whitespace withheld before a detour rejoins the text when it resumes.
    assert expected == {
        "reasoning": "plan: 日本 🙂 \nmore",
        "content": "first <| then <|c  last",
    }


def test_finish_reports_a_contradicting_parse_and_sends_nothing_more(caplog):
    stream = new_stream()
    stream.feed(b("hello"))
    with caplog.at_level(logging.WARNING, logger="renderers.streaming"):
        tail = stream.finish(reasoning_content="why", content="goodbye")
    # The consistent channel still completes; the contradicted one stays put.
    assert tail == [TextDelta("reasoning", "why")]
    assert "content text is not a prefix" in caplog.text
    assert stream.streamed["content"] == "hello"


def test_remaining_deltas_without_a_stream_carry_the_whole_stripped_fields():
    assert remaining_deltas({}, reasoning_content=" r \n", content="\nc ") == [
        TextDelta("reasoning", "r"),
        TextDelta("content", "c"),
    ]
    assert remaining_deltas({}, reasoning_content=None, content="") == []


# ── Real grammars: the stream must agree with the renderer's own parser ──

QWEN3 = "Qwen/Qwen3-0.6B"
DEEPSEEK_V4 = "deepseek-ai/DeepSeek-V4-Flash-0731"
MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]


@lru_cache(maxsize=None)
def _tokenizer(model: str):
    return load_test_tokenizer(model)


def teardown_module():
    _tokenizer.cache_clear()


class Format:
    """A renderer plus the vocabulary to write completions for it."""

    def __init__(self, name: str):
        self.name = name
        if name == "qwen3":
            self.tokenizer = _tokenizer(QWEN3)
            self.renderer = Qwen3Renderer(self.tokenizer)
            self.markers = {
                name: self.tokenizer.convert_tokens_to_ids(token)
                for name, token in {
                    "THINK": "<think>",
                    "/THINK": "</think>",
                    "TOOL": "<tool_call>",
                    "/TOOL": "</tool_call>",
                    "STOP": "<|im_end|>",
                }.items()
            }
            self.tool_text = [
                "TOOL",
                '\n{"name": "weather", "arguments": {"city": "Paris"}}\n',
                "/TOOL",
            ]
        else:
            self.tokenizer = _tokenizer(DEEPSEEK_V4)
            self.renderer = DeepSeekV4Renderer(
                self.tokenizer,
                DeepSeekV4RendererConfig(
                    enable_thinking=name == "deepseek_v4_thinking"
                ),
            )
            self.markers = {
                name: self.tokenizer.convert_tokens_to_ids(token)
                for name, token in {
                    "THINK": "<think>",
                    "/THINK": "</think>",
                    "STOP": "<｜end▁of▁sentence｜>",
                }.items()
            }
            self.tool_text = [
                "\n\n<｜DSML｜tool_calls>\n",
                '<｜DSML｜invoke name="weather">\n',
                '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n',
                "</｜DSML｜invoke>\n",
                "</｜DSML｜tool_calls>",
            ]
        self.prompt_ids = self.renderer.render_ids(MESSAGES, add_generation_prompt=True)

    def ids(self, pieces: list[str]) -> list[int]:
        """Marker names become their token; other strings are encoded as text."""
        out: list[int] = []
        for piece in pieces:
            if piece in self.markers:
                out.append(self.markers[piece])
            else:
                out.extend(self.tokenizer.encode(piece, add_special_tokens=False))
        return out

    def prompts(self) -> dict[str, list[int]]:
        """The generation prompt, and one that leaves a reasoning region open."""
        opened = [*self.prompt_ids, *self.ids(["THINK", "\n"])]
        if self.name == "qwen3":
            return {"plain": self.prompt_ids, "reasoning-open": opened}
        if self.name == "deepseek_v4_thinking":
            return {"reasoning-open": self.prompt_ids}
        return {"plain": self.prompt_ids}


FORMATS = ["qwen3", "deepseek_v4", "deepseek_v4_thinking"]


@lru_cache(maxsize=None)
def _format(name: str) -> Format:
    return Format(name)


@pytest.fixture(params=FORMATS)
def fmt(request) -> Format:
    return _format(request.param)


def chunkings(token_ids: list[int], seed: int) -> list[list[list[int]]]:
    rng = random.Random(seed)
    plans = [[[t] for t in token_ids], [token_ids]]
    for _ in range(3):
        cuts = sorted(rng.sample(range(len(token_ids) + 1), min(4, len(token_ids) + 1)))
        plans.append(
            [token_ids[a:c] for a, c in zip([0, *cuts], [*cuts, len(token_ids)])]
        )
    return plans


def stream_and_check(fmt: Format, prompt_ids: list[int], token_ids: list[int], seed=0):
    """Stream ``token_ids`` under several chunkings and check the contract.

    Returns the text streamed before ``finish`` and the parsed response.
    """
    parsed = fmt.renderer.parse_response(token_ids, prompt_ids=prompt_ids)
    final = {
        "reasoning": (parsed.reasoning_content or "").strip(),
        "content": (parsed.content or "").strip(),
    }
    context = f"{fmt.name} completion={fmt.tokenizer.decode(token_ids)!r}"
    live = None
    for plan in chunkings(token_ids, seed):
        stream = fmt.renderer.response_stream(prompt_ids=prompt_ids)
        deltas = [delta for chunk in plan for delta in stream.feed(chunk)]
        assert all(delta.text for delta in deltas), context
        streamed = stream.streamed
        for channel in ("reasoning", "content"):
            assert joined(deltas, channel) == streamed[channel], context
            # Nothing streamed may ever need to be taken back.
            assert final[channel].startswith(streamed[channel]), (channel, context)
        tail = stream.finish(
            reasoning_content=parsed.reasoning_content, content=parsed.content
        )
        assert stream.streamed == final, context
        assert joined(deltas + tail, "reasoning") == final["reasoning"], context
        assert joined(deltas + tail, "content") == final["content"], context
        # Chunk boundaries are invisible in the text.
        live = live or streamed
        assert streamed == live, context
    return live, parsed


@pytest.mark.model
@pytest.mark.parametrize(
    "pieces",
    [
        ["Hello there."],
        ["  \n\nHello there.  \n", "STOP"],
        ["naïve café — 日本語 🙂 👩‍👩‍👧 done", "STOP"],
        ["a < b and a <| b, x </ y", "STOP"],
        ["line one\n\nline two\n", "    indented\n", "STOP"],
        ["   \n\n  ", "STOP"],
        ["STOP"],
        [],
    ],
)
def test_plain_completions_stream_completely(fmt, pieces):
    token_ids = fmt.ids(pieces)
    for name, prompt_ids in fmt.prompts().items():
        live, parsed = stream_and_check(fmt, prompt_ids, token_ids)
        # With no markers every character is live: nothing is left to finish.
        if name == "reasoning-open":
            assert live == {
                "reasoning": (parsed.reasoning_content or "").strip(),
                "content": "",
            }
        else:
            assert live == {"reasoning": "", "content": parsed.content.strip()}


@pytest.mark.model
def test_reasoning_then_answer_streams_both_channels_completely(fmt):
    if "reasoning-open" not in fmt.prompts():
        pytest.skip("format has no prompt-opened reasoning in this configuration")
    token_ids = fmt.ids(
        [
            "\nThe user asks about 天気.\nCheck it. \n",
            "/THINK",
            "\n\nIt is sunny ☀.",
            "STOP",
        ]
    )
    live, parsed = stream_and_check(fmt, fmt.prompts()["reasoning-open"], token_ids)
    assert parsed.reasoning_complete is not False
    assert live == {
        "reasoning": "The user asks about 天気.\nCheck it.",
        "content": "It is sunny ☀.",
    }


@pytest.mark.model
def test_reasoning_opened_by_the_completion_streams_as_reasoning(fmt):
    token_ids = fmt.ids(["THINK", "\nhmm\n", "/THINK", "\n\nfine", "STOP"])
    for prompt_ids in fmt.prompts().values():
        live, _ = stream_and_check(fmt, prompt_ids, token_ids)
        assert live == {"reasoning": "hmm", "content": "fine"}


@pytest.mark.model
def test_unclosed_reasoning_streams_as_reasoning_only(fmt):
    if "reasoning-open" not in fmt.prompts():
        pytest.skip("format has no prompt-opened reasoning in this configuration")
    token_ids = fmt.ids(["\nstill thinking about it"])
    live, parsed = stream_and_check(fmt, fmt.prompts()["reasoning-open"], token_ids)
    assert parsed.reasoning_complete is False
    assert live == {"reasoning": "still thinking about it", "content": ""}


@pytest.mark.model
def test_tool_calls_arrive_whole_after_the_streamed_text(fmt):
    token_ids = fmt.ids(["Let me check. \n", *fmt.tool_text, "STOP"])
    for name, prompt_ids in fmt.prompts().items():
        if name == "reasoning-open":
            token_ids = fmt.ids(
                [
                    "I should look it up.",
                    "/THINK",
                    "Let me check. \n",
                    *fmt.tool_text,
                    "STOP",
                ]
            )
        live, parsed = stream_and_check(fmt, prompt_ids, token_ids)
        assert [call.name for call in parsed.tool_calls] == ["weather"]
        assert parsed.tool_calls[0].arguments == {"city": "Paris"}
        # All of the text before the call is live, and none of the call is.
        assert live["content"] == "Let me check."
        for text in live.values():
            assert (
                "weather" not in text and "DSML" not in text and "tool_call" not in text
            )


@pytest.mark.model
def test_text_after_a_tool_call_is_left_to_the_final_parse(fmt):
    token_ids = fmt.ids(
        ["First.", *fmt.tool_text, "\nAfterthought.", *fmt.tool_text, "STOP"]
    )
    for prompt_ids in fmt.prompts().values():
        live, parsed = stream_and_check(fmt, prompt_ids, token_ids)
        assert len(parsed.tool_calls) == 2
        assert "Afterthought" not in "".join(live.values())


@pytest.mark.model
def test_tool_markup_inside_reasoning_never_reaches_a_delta(fmt):
    if "reasoning-open" not in fmt.prompts():
        pytest.skip("format has no prompt-opened reasoning in this configuration")
    prompt_ids = fmt.prompts()["reasoning-open"]
    for closing in (["STOP"], ["\nno, not yet", "/THINK", "Answer.", "STOP"]):
        token_ids = fmt.ids(["Maybe call it:", *fmt.tool_text, *closing])
        live, _ = stream_and_check(fmt, prompt_ids, token_ids)
        assert live["reasoning"] == "Maybe call it:"
        assert live["content"] == ""


@pytest.mark.model
def test_think_markers_after_content_follow_the_format_grammar(fmt):
    token_ids = fmt.ids(["Before", "THINK", "aside", "/THINK", "After", "STOP"])
    prompt_ids = fmt.prompts().get("plain")
    if prompt_ids is None:
        pytest.skip("format opens reasoning in this configuration")
    live, _ = stream_and_check(fmt, prompt_ids, token_ids)
    if fmt.name == "qwen3":
        assert live == {"reasoning": "", "content": "Before<think>aside</think>After"}
    else:
        assert live == {"reasoning": "aside", "content": "BeforeAfter"}


def _fuzz_vocabulary(fmt: Format) -> list[list[int]]:
    texts = [
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
        '{"name": "weather", "arguments": {"city": "Paris"}}',
        "<think>"[:3],
        "tool_call",
    ]
    if fmt.name != "qwen3":
        texts += [
            "<｜DSML｜tool_calls>",
            "</｜DSML｜tool_calls>",
            '<｜DSML｜invoke name="weather">',
            "</｜DSML｜invoke>",
            '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>',
            "<｜DSML｜",
            "｜DSML｜",
            "tool_calls>",
            "invoke",
        ]
    vocabulary = [fmt.ids([text]) for text in texts]
    # Markers are over-represented so most samples cross several of them.
    for name in fmt.markers:
        vocabulary += [[fmt.markers[name]]] * (1 if name == "STOP" else 3)
    # Lone bytes of a multi-byte character, as a truncated completion ends.
    emoji = fmt.ids(["🙂"])
    if len(emoji) > 1:
        vocabulary.append(emoji[:1])
    return vocabulary


@pytest.mark.model
@pytest.mark.parametrize("seed", range(400))
def test_random_completions_never_stream_text_the_parser_disowns(fmt, seed):
    rng = random.Random(f"{fmt.name}-{seed}")
    vocabulary = _fuzz_vocabulary(fmt)
    token_ids = [
        token_id
        for _ in range(rng.randint(1, 14))
        for token_id in rng.choice(vocabulary)
    ]
    for prompt_ids in fmt.prompts().values():
        stream_and_check(fmt, prompt_ids, token_ids, seed=seed)


@pytest.mark.model
def test_prompt_state_comes_from_the_prompt_not_the_configuration(fmt):
    token_ids = fmt.ids(["thought", "/THINK", "answer"])
    for name, prompt_ids in fmt.prompts().items():
        stream = fmt.renderer.response_stream(prompt_ids=prompt_ids)
        stream.feed(token_ids)
        if name == "reasoning-open":
            assert stream.streamed == {"reasoning": "thought", "content": "answer"}
        else:
            assert stream.streamed["reasoning"] == ""
