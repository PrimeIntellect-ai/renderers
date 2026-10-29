"""Token-aware boundaries and state transitions for tagged completion grammars."""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from renderers.reasoning import ReasoningBoundary


@dataclass(frozen=True)
class TaggedGrammar:
    transitions: Mapping[tuple[str, str], str]
    text_markers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaggedToolBlock:
    text: str
    token_span: tuple[int, int]
    closed: bool


@dataclass(frozen=True)
class TaggedResponse:
    boundary: ReasoningBoundary
    content: str
    tools: list[TaggedToolBlock]


@dataclass(frozen=True)
class _Lexeme:
    terminal: str
    text: str
    start: int
    end: int


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False) if ids else ""


def _text_lexemes(
    tokenizer, ids: list[int], offset: int, markers: Mapping[str, str]
) -> Iterator[_Lexeme]:
    text = _decode(tokenizer, ids)
    if not markers:
        yield _Lexeme("text", text, offset, offset + len(ids))
        return
    pattern = "|".join(
        re.escape(marker)
        for marker in sorted(markers, key=lambda marker: len(marker), reverse=True)
    )
    cursor = 0
    token_cursor = 0
    decoded_end = 0

    def token_boundary(char_offset: int, *, end: bool) -> int:
        nonlocal token_cursor, decoded_end
        while decoded_end < char_offset and token_cursor < len(ids):
            token_cursor += 1
            decoded_end = len(_decode(tokenizer, ids[:token_cursor]))
        # A token may contain both framing whitespace and part of a delimiter.
        # Cover the whole token without re-encoding or editing sampled IDs.
        return offset + token_cursor - int(not end and decoded_end > char_offset)

    for match in re.finditer(pattern, text):
        start = token_boundary(match.start(), end=False)
        end = token_boundary(match.end(), end=True)
        yield _Lexeme("text", text[cursor : match.start()], offset, start)
        yield _Lexeme(markers[match.group()], match.group(), start, end)
        cursor = match.end()
    yield _Lexeme("text", text[cursor:], offset, offset + len(ids))


def _lex(
    tokenizer,
    ids: list[int],
    atomic_markers: Mapping[int, str],
    text_markers: Mapping[str, str],
) -> Iterator[_Lexeme]:
    """Atomic terminals match IDs; textual terminals match only decoded runs.

    In particular, spelling an atomic marker with ordinary tokens leaves it as
    text. Splitting at atomic tokens also prevents a textual match across one.
    """
    start = 0
    for i, token_id in enumerate(ids):
        if token_id not in atomic_markers:
            continue
        yield from _text_lexemes(tokenizer, ids[start:i], start, text_markers)
        yield _Lexeme(
            atomic_markers[token_id], _decode(tokenizer, [token_id]), i, i + 1
        )
        start = i + 1
    yield from _text_lexemes(tokenizer, ids[start:], start, text_markers)


def scan_tagged_response(
    tokenizer,
    ids: list[int],
    *,
    thinking: bool,
    atomic_markers: Mapping[int, str],
    grammar: TaggedGrammar,
) -> TaggedResponse:
    """Classify channels and tool blocks in one pass over the same terminals.

    ``tool`` owns its payload until the grammar transitions out of it: reasoning
    delimiters and nested openers there are payload, not independent scans.
    Tool argument conversion is left to the model-specific parser.
    """
    state = "reasoning" if thinking else "content"
    opened = thinking
    reasoning: list[str] = []
    content: list[str] = []
    tools: list[TaggedToolBlock] = []
    tool_text: list[str] = []
    tool_start = 0
    for lexeme in _lex(tokenizer, ids, atomic_markers, grammar.text_markers):
        next_state = grammar.transitions.get((state, lexeme.terminal))
        if next_state is None:
            if state == "tool":
                tool_text.append(lexeme.text)
            elif state == "reasoning":
                reasoning.append(lexeme.text)
            elif state == "content":
                content.append(lexeme.text)
            continue
        if state == "tool" and next_state != "tool":
            closed = lexeme.terminal == "tool_end"
            if closed:
                tool_text.append(lexeme.text)
            tools.append(
                TaggedToolBlock(
                    "".join(tool_text),
                    (tool_start, lexeme.end if closed else lexeme.start),
                    closed,
                )
            )
        elif state != "tool" and next_state == "tool":
            tool_start = lexeme.start
            tool_text = [lexeme.text]
        state = next_state
        opened |= state == "reasoning"
    if state == "tool":
        tools.append(TaggedToolBlock("".join(tool_text), (tool_start, len(ids)), False))
    return TaggedResponse(
        ReasoningBoundary(state == "reasoning", "".join(reasoning) if opened else None),
        "".join(content),
        tools,
    )


_REASONING_TRANSITIONS = {
    ("content", "think_start"): "reasoning",
    ("reasoning", "think_start"): "reasoning",
    ("reasoning", "think_end"): "content",
    ("content", "think_end"): "content",
}

# vLLM 0.31: GLM resumes content after a call; DeepSeek stays between calls
# and drops trailing prose. Keep those format rules separate from lexing.
GLM_GRAMMAR = TaggedGrammar(
    {
        **_REASONING_TRANSITIONS,
        ("content", "tool_start"): "tool",
        ("reasoning", "tool_start"): "tool",
        ("tool", "tool_end"): "content",
    }
)

DEEPSEEK_V4_GRAMMAR = TaggedGrammar(
    {
        **_REASONING_TRANSITIONS,
        ("content", "section_start"): "preamble",
        ("reasoning", "section_start"): "preamble",
        ("between", "section_start"): "preamble",
        ("content", "tool_start"): "tool",
        ("preamble", "tool_start"): "tool",
        ("between", "tool_start"): "tool",
        ("tool", "tool_end"): "between",
        ("tool", "section_end"): "between",
    },
    {
        "<｜DSML｜tool_calls>": "section_start",
        "<｜DSML｜toolcalls>": "section_start",
        "<｜DSML｜tool>": "section_start",
        "</｜DSML｜tool_calls>": "section_end",
        "<｜DSML｜invoke": "tool_start",
        "</｜DSML｜invoke>": "tool_end",
    },
)
