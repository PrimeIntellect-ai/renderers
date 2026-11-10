"""Incremental reasoning/content text for a completion that is still arriving.

:class:`ResponseStream` consumes completion token IDs as the engine produces
them and returns the reasoning and content text that is already certain. It
is a display aid: the authoritative result is still
``Renderer.parse_response`` over the whole completion, and
:meth:`ResponseStream.finish` reconciles the two from its parsed fields.

Contract, per channel: the text deltas, including the ones ``finish``
returns, concatenate to the final parsed field with surrounding whitespace
stripped. Tool calls are never streamed. Once a tool-call opener arrives the
stream goes quiet, and whatever the final parse makes of the remainder is
delivered by ``finish``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from renderers.base import Tokenizer

_logger = logging.getLogger("renderers.streaming")

Channel = Literal["reasoning", "content"]
_CHANNELS: tuple[Channel, Channel] = ("reasoning", "content")
_REPLACEMENT = "\ufffd"


@dataclass(frozen=True)
class TextDelta:
    """Newly certain text on one channel."""

    channel: Channel
    text: str


@dataclass(frozen=True)
class StreamGrammar:
    """The markers a format's completion grammar switches channels on.

    ``tool_ids`` are atomic tool-call openers. ``tool_markers`` are textual
    openers, matched in decoded text the way the format's parser matches them.
    The first of either ends the stream.

    ``reentrant_reasoning`` selects between the two families of think-marker
    rules. ``False`` is the reasoning-first grammar: only an initial opener
    (after optional whitespace) or a prompt-opened region is reasoning, and
    think markers after it are literal content. ``True`` treats every opener
    as a channel switch and drops the markers from the text.
    """

    stop_ids: frozenset[int]
    think_open_ids: frozenset[int]
    think_close_ids: frozenset[int]
    tool_ids: frozenset[int] = frozenset()
    tool_markers: tuple[str, ...] = ()
    reentrant_reasoning: bool = False


def remaining_deltas(
    streamed: Mapping[Channel, str],
    *,
    reasoning_content: str | None,
    content: str | None,
) -> list[TextDelta]:
    """The deltas that complete ``streamed`` to the final parsed fields.

    A parse that contradicts already-streamed text is logged. The parse stays
    authoritative and that channel gets no further delta.
    """
    deltas: list[TextDelta] = []
    for channel, value in (("reasoning", reasoning_content), ("content", content)):
        final = (value or "").strip()
        sent = streamed.get(channel, "")
        if not final.startswith(sent):
            _logger.warning(
                "Streamed %s text is not a prefix of the parsed response "
                "(%d streamed chars, %d parsed chars)",
                channel,
                len(sent),
                len(final),
            )
        elif len(final) > len(sent):
            deltas.append(TextDelta(channel, final[len(sent) :]))
    return deltas


class _Detokenizer:
    """Decode one run of tokens incrementally, withholding incomplete characters."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._read = 0

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self._tokenizer.decode(self._ids, skip_special_tokens=False)
        if text.endswith(_REPLACEMENT):
            # Possibly half of a multi-byte character; wait for the rest.
            return ""
        # Decode with the already-read tokens as context, so tokens whose text
        # depends on their left neighbour come out as they do in a full decode.
        seen = self._tokenizer.decode(
            self._ids[: self._read], skip_special_tokens=False
        )
        delta = text[len(seen) :]
        self._ids = self._ids[self._read :]
        self._read = len(self._ids)
        return delta

    def flush(self) -> str:
        """End the run, returning the text of any withheld tokens."""
        text = self._tokenizer.decode(self._ids, skip_special_tokens=False)
        seen = self._tokenizer.decode(
            self._ids[: self._read], skip_special_tokens=False
        )
        self._ids = []
        self._read = 0
        return text[len(seen) :]


def _marker_prefix_len(text: str, markers: tuple[str, ...]) -> int:
    """Length of the longest suffix of ``text`` that could still become a marker."""
    longest = 0
    for marker in markers:
        for size in range(min(len(marker) - 1, len(text)), longest, -1):
            if text.endswith(marker[:size]):
                longest = size
                break
    return longest


class ResponseStream:
    """Split an arriving completion into reasoning and content text deltas.

    ``reasoning`` says whether the prompt left a reasoning region open.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        grammar: StreamGrammar,
        *,
        reasoning: bool = False,
    ) -> None:
        self._grammar = grammar
        self._detokenizer = _Detokenizer(tokenizer)
        # "start" is the reasoning-first grammar before its first visible
        # character: an opener there still starts reasoning.
        self._state: Literal["start", "reasoning", "content", "done"]
        if reasoning:
            self._state = "reasoning"
        elif grammar.reentrant_reasoning:
            self._state = "content"
        else:
            self._state = "start"
        self._pending: dict[Channel, str] = {"reasoning": "", "content": ""}
        self._streamed: dict[Channel, str] = {"reasoning": "", "content": ""}

    @property
    def streamed(self) -> dict[Channel, str]:
        """Text delivered so far, per channel."""
        return dict(self._streamed)

    def feed(self, token_ids: list[int]) -> list[TextDelta]:
        """Consume the next completion tokens; return the text they settle."""
        deltas: list[TextDelta] = []
        grammar = self._grammar
        for token_id in token_ids:
            if self._state == "done":
                break
            is_open = token_id in grammar.think_open_ids
            is_close = token_id in grammar.think_close_ids
            if is_open and self._state == "start":
                # A withheld partial character counts as text before the
                # opener, which then no longer starts reasoning.
                self._flush_run(deltas)
            if token_id in grammar.stop_ids or token_id in grammar.tool_ids:
                self._state = "done"
            elif is_open and self._opens_reasoning():
                self._switch("reasoning", deltas)
            elif is_open and self._state == "reasoning":
                # A repeated opener is dropped from the reasoning text.
                if grammar.reentrant_reasoning:
                    self._end_run(deltas)
            elif is_close and self._state == "reasoning":
                self._switch("content", deltas)
            elif is_close and grammar.reentrant_reasoning and self._state == "content":
                self._end_run(deltas)
            else:
                self._text(self._detokenizer.push(token_id), deltas)
        return deltas

    def finish(
        self, *, reasoning_content: str | None, content: str | None
    ) -> list[TextDelta]:
        """Return what the final parse holds beyond the streamed text.

        Covers text withheld at the end of the stream and everything after a
        tool-call opener.
        """
        self._state = "done"
        deltas = remaining_deltas(
            self._streamed, reasoning_content=reasoning_content, content=content
        )
        for delta in deltas:
            self._streamed[delta.channel] += delta.text
        return deltas

    def _opens_reasoning(self) -> bool:
        if self._grammar.reentrant_reasoning:
            return self._state == "content"
        return self._state == "start"

    def _switch(
        self, state: Literal["reasoning", "content"], deltas: list[TextDelta]
    ) -> None:
        self._end_run(deltas)
        if self._state != "done":
            self._state = state

    def _end_run(self, deltas: list[TextDelta]) -> None:
        """Close the decoded run: text markers never match across an atomic one."""
        self._flush_run(deltas)
        if self._state in _CHANNELS:
            self._settle(self._state, deltas, hold_marker_prefix=False)

    def _flush_run(self, deltas: list[TextDelta]) -> None:
        """Take the run's withheld bytes as they decode now, and start a new run."""
        self._text(self._detokenizer.flush(), deltas)

    def _text(self, text: str, deltas: list[TextDelta]) -> None:
        if not text:
            return
        channel: Channel = "reasoning" if self._state == "reasoning" else "content"
        self._pending[channel] += text
        self._settle(channel, deltas, hold_marker_prefix=True)
        if self._state == "start" and (
            self._streamed["content"] or self._pending["content"].strip()
        ):
            self._state = "content"

    def _settle(
        self, channel: Channel, deltas: list[TextDelta], *, hold_marker_prefix: bool
    ) -> None:
        """Emit the pending text that no later token can change."""
        text = self._pending[channel]
        if not self._streamed[channel]:
            text = text.lstrip()
        markers = self._grammar.tool_markers
        position = min(
            (found for marker in markers if (found := text.find(marker)) != -1),
            default=-1,
        )
        if position != -1:
            self._state = "done"
            text = text[:position]
            held = len(text)
        else:
            held = len(text)
            if hold_marker_prefix:
                held -= _marker_prefix_len(text, markers)
        # Trailing whitespace is stripped from the final field unless more
        # text follows, so it waits for that text.
        settled = text[:held].rstrip()
        self._pending[channel] = text[len(settled) :]
        self._emit(channel, settled, deltas)

    def _emit(self, channel: Channel, text: str, deltas: list[TextDelta]) -> None:
        if not text:
            return
        self._streamed[channel] += text
        if deltas and deltas[-1].channel == channel:
            deltas[-1] = TextDelta(channel, deltas[-1].text + text)
        else:
            deltas.append(TextDelta(channel, text))


__all__ = [
    "Channel",
    "ResponseStream",
    "StreamGrammar",
    "TextDelta",
    "remaining_deltas",
]
