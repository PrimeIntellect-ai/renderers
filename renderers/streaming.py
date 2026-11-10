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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from renderers.base import Tokenizer
from renderers.reasoning import _single_marker_id, prompt_ends_in_reasoning

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

    ``think_open_text`` / ``think_close_text`` are for reasoning-first formats
    whose think markers are not single tokens, so their parser finds them in
    decoded text.

    A completion that starts with one of ``silent_first_ids`` (a replayed
    assistant header, which an engine never samples) is not streamed at all.
    One of ``skipped_first_ids`` is a header the parser itself discards, so
    the stream discards it too and carries on.

    Content that begins with one of ``silent_content_prefixes`` may turn out
    to be a tool call as a whole, so it is left to the final parse.
    ``dropped_content_ids`` are framing tokens the parser removes from content.
    """

    stop_ids: frozenset[int]
    think_open_ids: frozenset[int] = frozenset()
    think_close_ids: frozenset[int] = frozenset()
    tool_ids: frozenset[int] = frozenset()
    tool_markers: tuple[str, ...] = ()
    reentrant_reasoning: bool = False
    think_open_text: str | None = None
    think_close_text: str | None = None
    silent_first_ids: frozenset[int] = frozenset()
    skipped_first_ids: frozenset[int] = frozenset()
    silent_content_prefixes: tuple[str, ...] = ()
    dropped_content_ids: frozenset[int] = frozenset()


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


class TextStream:
    """What a renderer's ``response_stream`` returns.

    ``feed`` takes the next completion tokens and returns the text they
    settle; ``finish`` takes the final parsed fields and returns the rest.
    Subclasses implement ``feed`` and ``_stop``.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._detokenizer = _Detokenizer(tokenizer)
        self._pending: dict[Channel, str] = {"reasoning": "", "content": ""}
        self._streamed: dict[Channel, str] = {"reasoning": "", "content": ""}

    @property
    def streamed(self) -> dict[Channel, str]:
        """Text delivered so far, per channel."""
        return dict(self._streamed)

    def feed(self, token_ids: list[int]) -> list[TextDelta]:
        """Consume the next completion tokens; return the text they settle."""
        raise NotImplementedError

    def finish(
        self, *, reasoning_content: str | None, content: str | None
    ) -> list[TextDelta]:
        """Return what the final parse holds beyond the streamed text.

        Covers text withheld at the end of the stream and everything after a
        tool-call opener.
        """
        self._stop()
        deltas = remaining_deltas(
            self._streamed, reasoning_content=reasoning_content, content=content
        )
        for delta in deltas:
            self._streamed[delta.channel] += delta.text
        return deltas

    def _stop(self) -> None:
        raise NotImplementedError

    def _emit(self, channel: Channel, text: str, deltas: list[TextDelta]) -> None:
        if not text:
            return
        self._streamed[channel] += text
        if deltas and deltas[-1].channel == channel:
            deltas[-1] = TextDelta(channel, deltas[-1].text + text)
        else:
            deltas.append(TextDelta(channel, text))


class ChannelStream(TextStream):
    """Base for formats with their own token grammar.

    A subclass decides, token by token, which channel a token's text belongs
    to; this class decodes it and applies the whitespace rules.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)
        self._done = False

    def feed(self, token_ids: list[int]) -> list[TextDelta]:
        deltas: list[TextDelta] = []
        for token_id in token_ids:
            if self._done:
                break
            self._token(token_id, deltas)
        return deltas

    def _token(self, token_id: int, deltas: list[TextDelta]) -> None:
        raise NotImplementedError

    def _stop(self) -> None:
        self._done = True

    def _push(self, channel: Channel, token_id: int, deltas: list[TextDelta]) -> None:
        """Decode ``token_id`` in the current run and add its text to ``channel``."""
        self._add(channel, self._detokenizer.push(token_id), deltas)

    def _end_run(self, channel: Channel | None, deltas: list[TextDelta]) -> None:
        """End the decoded run; its withheld bytes go to ``channel``, if any."""
        text = self._detokenizer.flush()
        if channel is not None:
            self._add(channel, text, deltas)

    def _add(self, channel: Channel, text: str, deltas: list[TextDelta]) -> None:
        text = self._pending[channel] + text
        if not self._streamed[channel]:
            text = text.lstrip()
        # Trailing whitespace waits for the text that would keep it.
        settled = text.rstrip()
        self._pending[channel] = text[len(settled) :]
        self._emit(channel, settled, deltas)


class ResponseStream(TextStream):
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
        super().__init__(tokenizer)
        self._grammar = grammar
        # "start" is the reasoning-first grammar before its first visible
        # character: an opener there still starts reasoning.
        self._state: Literal["start", "reasoning", "content", "done"]
        if reasoning:
            self._state = "reasoning"
        elif grammar.reentrant_reasoning:
            self._state = "content"
        else:
            self._state = "start"
        self._first = True

    def feed(self, token_ids: list[int]) -> list[TextDelta]:
        """Consume the next completion tokens; return the text they settle."""
        deltas: list[TextDelta] = []
        grammar = self._grammar
        for token_id in token_ids:
            if self._first:
                self._first = False
                if token_id in grammar.skipped_first_ids:
                    continue
                if token_id in grammar.silent_first_ids:
                    self._state = "done"
            if self._state == "done":
                break
            is_open = token_id in grammar.think_open_ids
            is_close = token_id in grammar.think_close_ids
            if is_open and self._state == "start":
                # A withheld partial character counts as text before the
                # opener, which then no longer starts reasoning.
                self._flush_run(deltas)
                self._settle(deltas, hold_marker_prefix=False)
            if token_id in grammar.stop_ids or token_id in grammar.tool_ids:
                self._state = "done"
            elif token_id in grammar.dropped_content_ids and self._state != "reasoning":
                # Framing counts as text before an opener, then disappears.
                self._flush_run(deltas)
                self._settle(deltas, hold_marker_prefix=False)
                if self._state == "start":
                    self._state = "content"
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

    def _stop(self) -> None:
        self._state = "done"

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
        self._settle(deltas, hold_marker_prefix=False)

    def _flush_run(self, deltas: list[TextDelta]) -> None:
        """Take the run's withheld bytes as they decode now, and start a new run."""
        self._text(self._detokenizer.flush(), deltas)

    def _text(self, text: str, deltas: list[TextDelta]) -> None:
        if not text or self._state == "done":
            return
        channel: Channel = "reasoning" if self._state == "reasoning" else "content"
        self._pending[channel] += text
        self._settle(deltas, hold_marker_prefix=True)

    def _text_markers(self) -> list[tuple[str, str]]:
        """The textual markers that mean something in the current state."""
        grammar = self._grammar
        markers = [(marker, "tool") for marker in grammar.tool_markers]
        if self._state == "reasoning" and grammar.think_close_text:
            markers.append((grammar.think_close_text, "close"))
            if grammar.think_open_text:
                markers.append((grammar.think_open_text, "drop"))
        return markers

    def _leave_start(self, text: str, *, hold_marker_prefix: bool) -> bool:
        """Decide what the first visible text makes of the "start" state.

        Returns False while the decision has to wait for more text.
        """
        opener = self._grammar.think_open_text
        if opener and text.startswith(opener):
            self._state = "reasoning"
            self._pending["content"] = ""
            self._pending["reasoning"] = text[len(opener) :]
        elif opener and hold_marker_prefix and opener.startswith(text):
            return False
        else:
            self._state = "content"
        return True

    def _settle(self, deltas: list[TextDelta], *, hold_marker_prefix: bool) -> None:
        """Emit the pending text that no later token can change."""
        while self._state != "done":
            channel: Channel = "reasoning" if self._state == "reasoning" else "content"
            text = self._pending[channel]
            if not self._streamed[channel]:
                text = text.lstrip()
                self._pending[channel] = text
            if self._state == "start":
                if not text or not self._leave_start(
                    text, hold_marker_prefix=hold_marker_prefix
                ):
                    return
                continue
            if channel == "content" and not self._streamed["content"] and text:
                prefixes = self._grammar.silent_content_prefixes
                if any(text.startswith(prefix) for prefix in prefixes):
                    self._state = "done"
                    return
                if hold_marker_prefix and any(p.startswith(text) for p in prefixes):
                    return
            markers = self._text_markers()
            position, marker, action = min(
                (
                    (found, marker, action)
                    for marker, action in markers
                    if (found := text.find(marker)) != -1
                ),
                default=(-1, "", ""),
            )
            if action == "drop":
                # A repeated opener is dropped from the reasoning text.
                self._pending[channel] = (
                    text[:position] + text[position + len(marker) :]
                )
                continue
            if action == "":
                held = len(text)
                if hold_marker_prefix:
                    held -= _marker_prefix_len(text, tuple(m for m, _ in markers))
                # Trailing whitespace is stripped from the final field unless
                # more text follows, so it waits for that text.
                settled = text[:held].rstrip()
                self._pending[channel] = text[len(settled) :]
                self._emit(channel, settled, deltas)
                return
            self._emit(channel, text[:position].rstrip(), deltas)
            self._pending[channel] = ""
            if action == "tool":
                self._state = "done"
            else:
                self._state = "content"
                self._pending["content"] += text[position + len(marker) :]


def reasoning_first_stream(
    tokenizer: Tokenizer,
    *,
    prompt_ids: list[int] | None,
    stop_ids: Iterable[int],
    tool_ids: Iterable[int | None] = (),
    tool_markers: tuple[str, ...] = (),
    think_open_id: int | None = None,
    think_close_id: int | None = None,
    open_marker: str = "<think>",
    close_marker: str = "</think>",
    assistant_prefix: str = "<|im_start|>assistant\n",
    skipped_first_ids: Iterable[int] = (),
    dropped_content_ids: Iterable[int] = (),
) -> ResponseStream:
    """A stream for the reasoning-first formats parsed with ``scan_reasoning``.

    Resolves think markers the way ``scan_reasoning`` does: by token ID when
    both are single tokens, otherwise in decoded text.
    """
    stop_ids = frozenset(stop_ids)
    if think_open_id is None:
        think_open_id = _single_marker_id(tokenizer, "<think>")
    if think_close_id is None:
        think_close_id = _single_marker_id(tokenizer, "</think>")
    atomic = think_open_id is not None and think_close_id is not None
    header = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    return ResponseStream(
        tokenizer,
        StreamGrammar(
            stop_ids=stop_ids,
            think_open_ids=frozenset({think_open_id} if atomic else ()),
            think_close_ids=frozenset({think_close_id} if atomic else ()),
            tool_ids=frozenset(i for i in tool_ids if i is not None),
            tool_markers=tool_markers,
            think_open_text=None if atomic else open_marker,
            think_close_text=None if atomic else close_marker,
            silent_first_ids=frozenset(header[:1]) - frozenset(skipped_first_ids),
            skipped_first_ids=frozenset(skipped_first_ids),
            dropped_content_ids=frozenset(dropped_content_ids),
        ),
        reasoning=prompt_ends_in_reasoning(
            tokenizer,
            prompt_ids,
            open_marker=open_marker,
            close_marker=close_marker,
            stop_ids=stop_ids,
            assistant_prefix=assistant_prefix,
        ),
    )


__all__ = [
    "Channel",
    "ChannelStream",
    "ResponseStream",
    "StreamGrammar",
    "TextDelta",
    "TextStream",
    "reasoning_first_stream",
    "remaining_deltas",
]
