"""Reasoning boundaries, independent of response parsing and renderer classes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningBoundary:
    """Reasoning state and text; None means no reasoning region was opened."""

    is_open: bool
    text: str | None


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False) if ids else ""


def _single_marker_id(tokenizer, marker: str) -> int | None:
    """Prefer grammar token IDs when the tokenizer exposes atomic markers."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    token = convert(marker)
    if isinstance(token, int) and token != getattr(tokenizer, "unk_token_id", None):
        return token
    return None


def _after_header(
    tokenizer, ids: list[int], header: str, *, last: bool = False
) -> list[int]:
    """Skip a renderer-declared assistant header, including its framing newlines."""
    prefix = list(tokenizer.encode(header, add_special_tokens=False))
    positions = range(len(ids) - len(prefix), -1, -1) if last else [0]
    for i in positions:
        if prefix and ids[i : i + len(prefix)] == prefix:
            ids = ids[i + len(prefix) :]
            while ids and _decode(tokenizer, ids[:1]).strip("\n") == "":
                ids = ids[1:]
            break
    return ids


def prompt_ends_in_reasoning(
    tokenizer,
    prompt_ids: list[int] | None,
    *,
    open_marker: str = "<think>",
    close_marker: str = "</think>",
    stop_ids: set[int] | frozenset[int] = frozenset(),
    assistant_prefix: str = "<|im_start|>assistant\n",
    initial_only: bool = True,
) -> bool:
    """Determine the initial channel from the current assistant prompt prefix."""
    if not prompt_ids:
        return False
    start = next(
        (
            i + 1
            for i in range(len(prompt_ids) - 1, -1, -1)
            if prompt_ids[i] in stop_ids
        ),
        0,
    )
    tail = _after_header(tokenizer, prompt_ids[start:], assistant_prefix, last=True)
    if not initial_only:
        text = _decode(tokenizer, tail)
        return text.rfind(open_marker) > text.rfind(close_marker)
    return scan_reasoning(
        tokenizer,
        tail,
        open_id=_single_marker_id(tokenizer, open_marker),
        close_id=_single_marker_id(tokenizer, close_marker),
        open_marker=open_marker,
        close_marker=close_marker,
        assistant_prefix=assistant_prefix,
    ).is_open


def scan_reasoning(
    tokenizer,
    ids: list[int],
    *,
    prefilled: bool = False,
    initial_only: bool = True,
    assistant_prefix: str = "<|im_start|>assistant\n",
    open_id: int | None = None,
    close_id: int | None = None,
    tool_start_id: int | None = None,
    prompt_ids: list[int] | None = None,
    stop_ids: set[int] | frozenset[int] = frozenset(),
    open_marker: str = "<think>",
    close_marker: str = "</think>",
) -> ReasoningBoundary:
    """Read the initial reasoning region without interpreting final content.

    Tagged formats recognize an opener only at the beginning of the assistant
    body, or continue reasoning opened in the prompt. A first close permanently
    switches to content. Atomic delimiters match token IDs, not lookalike text.
    Explicit channel formats can opt into channel transitions instead.
    """
    if prompt_ids is not None:
        prefilled = prompt_ends_in_reasoning(
            tokenizer,
            prompt_ids,
            stop_ids=stop_ids,
            open_marker=open_marker,
            close_marker=close_marker,
            assistant_prefix=assistant_prefix,
            initial_only=initial_only,
        )
    end = next((i for i, token in enumerate(ids) if token in stop_ids), len(ids))
    ids = (
        _after_header(tokenizer, ids[:end], assistant_prefix)
        if initial_only
        else ids[:end]
    )
    if open_id is None:
        open_id = _single_marker_id(tokenizer, "<think>")
    if close_id is None:
        close_id = _single_marker_id(tokenizer, "</think>")
    if initial_only:
        # Reasoning-first grammars have one initial reasoning region. Once it
        # closes, later markers are content, not a new reasoning channel.
        if open_id is not None and close_id is not None:
            explicit = bool(ids) and ids[0] == open_id
            if not prefilled and not explicit:
                return ReasoningBoundary(False, None)
            start = int(explicit)
            close = next((i for i in range(start, len(ids)) if ids[i] == close_id), -1)
            return ReasoningBoundary(
                close == -1,
                _decode(tokenizer, ids[start : close if close != -1 else len(ids)]),
            )
        text = _decode(tokenizer, ids)
        explicit = text.startswith(open_marker)
        if not prefilled and not explicit:
            return ReasoningBoundary(False, None)
        start = len(open_marker) if explicit else 0
        close = text.find(close_marker, start)
        return ReasoningBoundary(
            close == -1, text[start : close if close != -1 else len(text)]
        )

    events: list[tuple[int, int, bool | None]]
    if open_id is None or close_id is None:
        text = _decode(tokenizer, ids)
        import re

        events = [
            (m.start(), m.end(), m.group() == "<think>")
            for m in re.finditer(r"<think>|</think>", text)
        ]
        events.extend(
            (len(_decode(tokenizer, ids[:i])), 0, None)
            for i, token in enumerate(ids)
            if token == tool_start_id
        )
        end = len(text)

        def decode_region(a, b):
            return text[a:b]
    else:
        events = [
            (i, i + 1, t == open_id)
            for i, t in enumerate(ids)
            if t in {open_id, close_id}
        ]
        events.extend(
            (i, 0, None) for i, token in enumerate(ids) if token == tool_start_id
        )
        end = len(ids)

        def decode_region(a, b):
            return _decode(tokenizer, ids[a:b])

    active = prefilled
    start = 0
    closed_regions: list[tuple[int, int]] = []
    for pos, after, opening in sorted(events):
        if opening is None:
            if not active:
                break
            continue
        if opening:
            if not active:
                active = True
                start = after
            elif pos == 0:
                start = after
        elif active:
            closed_regions.append((start, pos))
            active = False
    if active:
        closed_regions.append((start, end))
    return ReasoningBoundary(
        is_open=active,
        text="".join(decode_region(a, b) for a, b in closed_regions),
    )
