from __future__ import annotations

import contextlib
import enum
import io
import logging
import queue
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    TypedDict,
    runtime_checkable,
)

if TYPE_CHECKING:
    from renderers.configs import AutoRendererConfig, RendererConfig

logger = logging.getLogger("renderers.base")


# ---------------------------------------------------------------------------
# Message types — strong typing for the conversation data model
# ---------------------------------------------------------------------------


class TextPart(TypedDict):
    """A chunk of text content in a message."""

    type: Literal["text"]
    text: str


class ThinkingPart(TypedDict):
    """Model's internal reasoning (chain-of-thought) as a content part."""

    type: Literal["thinking"]
    thinking: str


class ImagePart(TypedDict, total=False):
    """An image attached to a message.

    Accepts several source shapes so callers can pass whatever they have
    on hand — a pre-loaded PIL Image, a filesystem path, a URL, or the
    OpenAI ``image_url`` content part verbatim. The renderer resolves
    these to a PIL Image at render time.
    """

    type: Literal["image", "image_url"]
    image: Any
    url: str
    path: str
    image_url: dict[str, Any]


class VideoPart(TypedDict, total=False):
    """A video attached to a message.

    Mirrors :class:`ImagePart`; the renderer turns frames into the
    model's video placeholder sequence at render time.
    """

    type: Literal["video", "video_url"]
    video: Any
    url: str
    path: str
    video_url: dict[str, Any]


ContentPart = TextPart | ThinkingPart | ImagePart | VideoPart

# Content is either a plain string or a list of structured parts.
Content = str | list[ContentPart]


class ToolCallFunction(TypedDict):
    """Function body within a tool call."""

    name: str
    arguments: dict[str, Any] | str


class ToolCall(TypedDict, total=False):
    """Structured tool invocation following OpenAI function-calling format."""

    type: str  # "function"
    id: str
    function: ToolCallFunction


class ToolSpec(TypedDict):
    """Tool specification (OpenAI function-calling format)."""

    name: str
    description: str
    parameters: dict[str, Any]


class Message(TypedDict, total=False):
    """A single turn in a multi-turn conversation.

    Required keys: role, content.
    Optional keys mirror the OpenAI chat format for tool calling.
    """

    role: str
    content: Content
    tool_calls: list[ToolCall]
    tool_call_id: str
    name: str
    reasoning_content: str


def extract_message_tool_names(messages: list[Message]) -> list[str | None]:
    """Per-message tool function names parallel to ``message_roles``.

    Returns one entry per message: the function name for ``role="tool"``
    messages, ``None`` for every other message. Length matches the
    input list.

    For tool messages the name is taken from ``msg["name"]`` when set
    (caller-provided), otherwise recovered by joining
    ``msg["tool_call_id"]`` against any prior assistant's
    ``tool_calls[i].function.name`` in the same list. Tool messages
    whose issuing assistant lives outside the provided list (e.g. on
    a :meth:`Renderer.bridge_to_next_turn` call where ``new_messages``
    covers only the new turn) resolve to ``None``.

    Pure metadata: this never mutates the caller's messages and has
    no effect on the rendered token stream. It runs independently of
    the render path so the renderer can populate the field on
    :class:`RenderedTokens` without breaking HF byte parity for tool
    messages that carry no ``name``. Callers who *also* want the
    function name to appear in the rendered scaffold (e.g. GPT-OSS
    Harmony's ``functions.{name}`` prefix) must attach ``name`` to
    their tool messages before calling :meth:`Renderer.render`
    themselves — renderers don't synthesize ``name`` into the input,
    only into this metadata field.

    Trainers join this list with :attr:`RenderedTokens.message_indices`
    to recover per-token tool attribution — the canonical use case is
    SFT on tool response bodies while RL acts only on assistant tokens
    (tool body tokens get a constant positive advantage so the model
    learns to anticipate tool outputs without learning to emit
    ``<|tool_response>`` itself).

    Per-message rather than per-token because the data is naturally
    per-message — storing it per-token would duplicate the same
    string across every body token of the same tool message.
    """
    lookup: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, Mapping) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, Mapping):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function")
            tc_name = fn.get("name") if isinstance(fn, Mapping) else None
            if isinstance(tc_id, str) and isinstance(tc_name, str):
                lookup[tc_id] = tc_name
    out: list[str | None] = []
    for m in messages:
        if not isinstance(m, Mapping) or m.get("role") != "tool":
            out.append(None)
            continue
        name = m.get("name")
        if not (isinstance(name, str) and name):
            tc_id = m.get("tool_call_id")
            name = lookup.get(tc_id) if isinstance(tc_id, str) else None
        out.append(name if isinstance(name, str) and name else None)
    return out


# ---------------------------------------------------------------------------
# Renderer data types
# ---------------------------------------------------------------------------


@dataclass
class PlaceholderRange:
    """Where a single multimodal item's placeholder tokens sit in the stream.

    ``offset`` is the 0-based index into ``RenderedTokens.token_ids`` of the
    first placeholder token; ``length`` is the count of consecutive
    placeholder tokens. Wraps the vLLM-style ``mm_placeholders`` shape
    without depending on vLLM types.
    """

    offset: int
    length: int


@dataclass
class MultiModalData:
    """Multimodal sidecar produced alongside the token stream.

    Renderer output is framework-agnostic: ``mm_items[modality][i]`` is a
    plain ``dict`` mirroring the per-item output of a HuggingFace processor
    (e.g. ``{"pixel_values": Tensor, "image_grid_thw": Tensor}`` for
    Qwen3-VL images). Translation to engine-specific wire formats — vLLM's
    ``MultiModalKwargsItem``, SGLang's payload, etc. — happens in the
    inference glue layer (see ``renderers.client``).
    """

    mm_hashes: dict[str, list[str]] = field(default_factory=dict)
    mm_placeholders: dict[str, list[PlaceholderRange]] = field(default_factory=dict)
    mm_items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.mm_hashes or self.mm_placeholders or self.mm_items)


@dataclass
class RenderedTokens:
    """Result of rendering messages to tokens.

    Each token carries an index into the original message list so callers can
    build per-token loss masks without re-rendering. Tokens from structural
    scaffolding the renderer adds outside any single message (e.g. the
    trailing generation prompt) carry index ``-1``.

    ``sampled_mask`` is a separate per-token signal: ``True`` if the model
    would have produced this token at inference time (i.e. it appears in
    the sampled completion), ``False`` if it is template-injected
    scaffolding the model never emits (``<|im_start|>role\\n`` openers,
    inter-turn ``\\n`` separators, system / user / tool content from
    conversation history, etc.). This is distinct from
    ``message_indices``: a token can belong to an assistant message
    (``message_indices[k] >= 0``) and still be scaffolding the template
    adds around the model's actual completion. SFT loss masks should AND
    both: train on tokens whose role is trainable AND that the model
    would actually sample.

    Empty ``sampled_mask`` (``[]``) means the renderer doesn't provide
    this signal — consumers should fall back to attribution-only
    masking. ``DefaultRenderer`` leaves it empty because the Jinja
    template is opaque; hand-coded renderers populate it.

    ``is_content`` is a per-token signal generalizing the "scaffold vs
    body" distinction across all roles: ``True`` iff the token was
    produced from message-body bytes (caller-provided ``content`` /
    ``tool_calls`` / ``reasoning_content``, or the model's sampled
    emission for the assistant role), ``False`` iff it is template
    scaffolding the renderer added around message bodies — role-tag
    openers, closers when not model-sampled, inter-turn separators,
    tool-response wraps, the tools-header block, the generation prompt.
    Generalises ``sampled_mask``: where ``sampled_mask`` answers "would
    the model emit this?" (useful for assistant tokens; uniformly
    ``False`` elsewhere), ``is_content`` answers "is this from caller
    or model data?" (meaningful on every role). By construction
    ``is_content[k] == sampled_mask[k]`` over every token attributed to
    an assistant message; on other roles ``is_content`` carries new
    information that ``sampled_mask`` does not.

    The use case: SFT on tool response bodies while applying RL only to
    assistant tokens. The trainer wants the model to anticipate tool
    outputs but never to emit ``<|tool_response>`` itself (that would
    interrupt the rollout), so the SFT loss mask is
    ``message_role == "tool" AND is_content``.

    Empty ``is_content`` (``[]``) — like ``sampled_mask`` — means the
    renderer doesn't provide the signal. ``DefaultRenderer`` leaves it
    empty for the same reason.

    ``message_tool_names`` is the per-message tool function name list,
    parallel to ``message_roles`` (same length). For tool-role
    messages it carries the function name — either taken from
    ``msg["name"]`` (caller-provided) or recovered by joining
    ``msg["tool_call_id"]`` against a prior assistant's
    ``tool_calls[i].function.name`` in the rendered slice. Every
    other message is ``None``, as are tool messages whose issuing
    assistant lives outside the rendered slice (e.g. on a
    :meth:`Renderer.bridge_to_next_turn` call where ``new_messages``
    covers only the new turn).

    This is pure metadata, computed by :func:`extract_message_tool_names`
    independently of the render path: populating it never touches the
    rendered token stream, so HF chat-template byte parity is
    preserved for tool messages carrying no ``name``. Callers who
    *also* want the function name to appear in the rendered scaffold
    (e.g. GPT-OSS Harmony's ``functions.{name}`` prefix) must attach
    ``name`` to their tool messages before calling
    :meth:`Renderer.render` themselves.

    Trainers join this with ``message_indices`` to build per-tool
    selective loss masks (SFT on tool response bodies of a specific
    tool while RL acts on assistant tokens). Empty
    ``message_tool_names`` (``[]``) means the renderer doesn't
    provide the signal.

    ``multi_modal_data`` is populated by multimodal renderers (e.g.
    ``Qwen3VLRenderer``) when image / video content parts are present;
    text-only renderers leave it as ``None``.
    """

    token_ids: list[int] = field(default_factory=list)
    message_indices: list[int] = field(default_factory=list)
    sampled_mask: list[bool] = field(default_factory=list)
    is_content: list[bool] = field(default_factory=list)
    message_roles: list[str] = field(default_factory=list)
    message_tool_names: list[str | None] = field(default_factory=list)
    multi_modal_data: "MultiModalData | None" = None

    def tokens_per_message(
        self, n_messages: int | None = None, *, sampled_only: bool = False
    ) -> list[int]:
        """Count rendered tokens attributed to each caller-relative message.

        ``out[i]`` is the number of tokens with ``message_indices[k] == i``,
        i.e. tokens the renderer attributed to ``messages[i]``. This
        includes template scaffolding the renderer wraps around the
        message — the ``<|im_start|>role\\n`` opener, the closing
        ``<|im_end|>\\n``, etc. — because those are the renderer's own
        attribution decision and are preserved verbatim here. Tokens with
        ``message_indices[k] == -1`` (scaffolding outside any single
        message, e.g. the trailing generation prompt) are not counted.

        With ``sampled_only=True``, counts only tokens the model would
        have emitted at inference (``sampled_mask[k] is True``). For
        example, length-penalty signals in RL: the template wraps each
        assistant turn in scaffolding tokens (e.g. ``<|im_start|>assistant\\n``,
        ``<|im_end|>\\n``) that are constant-size and not chosen by the
        model, so they shouldn't enter the penalty. For roles the model
        never samples (``user``, ``tool``, ``system``), the
        ``sampled_only`` count is zero by construction. Renderers that
        don't populate ``sampled_mask`` (``DefaultRenderer`` — the Jinja
        template is opaque) return all zeros under ``sampled_only=True``.

        ``n_messages`` defaults to ``len(self.message_roles)``, which
        every Renderer populates with the caller-relative message list
        (caller's ``messages`` for ``render()``; ``new_messages`` for
        ``bridge_to_next_turn()``). Pass it explicitly only to truncate
        — indices outside ``[0, n_messages)`` are ignored, so passing a
        smaller value won't raise; it just drops the tail. Values larger
        than ``len(self.message_roles)`` are clamped, so the returned
        list never claims more messages than the renderer attributed.

        Works on results from both :meth:`Renderer.render` and
        :meth:`Renderer.bridge_to_next_turn`. For a bridge result the
        indices are relative to the new messages the bridge added, not
        the full conversation history; the prior portion is uniformly
        ``-1`` (and ``sampled_mask`` uniformly ``False``), so it
        contributes nothing to either count.
        """
        if n_messages is None:
            n_messages = len(self.message_roles)
        else:
            n_messages = min(n_messages, len(self.message_roles))
        out = [0] * n_messages
        if sampled_only:
            if len(self.sampled_mask) != len(self.token_ids):
                return out
            for idx, sampled in zip(self.message_indices, self.sampled_mask):
                if sampled and 0 <= idx < n_messages:
                    out[idx] += 1
        else:
            for idx in self.message_indices:
                if 0 <= idx < n_messages:
                    out[idx] += 1
        return out

    def message_token_spans(self) -> list[tuple[int, int] | None]:
        """Per-message ``(start, end)`` slices into :attr:`token_ids`.

        ``out[i]`` is the half-open span ``[start, end)`` such that
        ``token_ids[start:end]`` are the tokens attributed to
        ``messages[i]`` (or ``new_messages[i]`` for a bridge result).
        Messages that contributed no tokens get ``None``. Renderer
        scaffolding outside any message (``message_indices[k] == -1``)
        is not represented.

        Hand-coded renderers emit each message's tokens contiguously,
        so the span is well-defined. The implementation tolerates
        non-contiguous attribution by returning the outer span
        ``(first_k, last_k + 1)``; if you suspect interleaving, slice
        ``message_indices`` yourself to verify.

        Returns ``len(self.message_roles)`` entries when ``message_roles``
        is populated. Otherwise infers the count from
        ``max(message_indices) + 1`` — useful for manually-constructed
        ``RenderedTokens`` in tests but only correct when the last
        message contributed at least one token.

        Cheap to call: single pass over ``message_indices``. Re-call
        rather than caching the result if you mutate the dataclass.
        """
        if self.message_roles:
            n_messages = len(self.message_roles)
        else:
            max_idx = -1
            for idx in self.message_indices:
                if idx > max_idx:
                    max_idx = idx
            n_messages = max_idx + 1

        firsts: list[int] = [-1] * n_messages
        lasts: list[int] = [-1] * n_messages
        for k, idx in enumerate(self.message_indices):
            if 0 <= idx < n_messages:
                if firsts[idx] == -1:
                    firsts[idx] = k
                lasts[idx] = k

        out: list[tuple[int, int] | None] = []
        for i in range(n_messages):
            if firsts[i] == -1:
                out.append(None)
            else:
                out.append((firsts[i], lasts[i] + 1))
        return out

    def role_token_spans(self) -> dict[str, list[tuple[int, int]]]:
        """:meth:`message_token_spans` regrouped by ``message_roles``.

        Maps each role appearing in :attr:`message_roles` to a list of
        ``(start, end)`` spans — one per occurrence of that role, in
        message order. Messages with no contributed tokens are skipped.
        Returns an empty dict if :attr:`message_roles` is empty.

        Intended for per-role statistics that operate on per-token
        signals — e.g. ``logprobs[start:end]`` for each assistant span
        to compute per-turn perplexity, or
        ``attention[start:end]`` for tool-response attention analysis.
        """
        spans = self.message_token_spans()
        out: dict[str, list[tuple[int, int]]] = {}
        for role, span in zip(self.message_roles, spans):
            if span is None:
                out.setdefault(role, [])
                continue
            out.setdefault(role, []).append(span)
        return out

    def tokens_by_role(self, *, sampled_only: bool = False) -> dict[str, int]:
        """Sum :meth:`tokens_per_message` grouped by ``message_roles``.

        Convenience for length-penalty bookkeeping in RL trainers:
        ``rendered.tokens_by_role(sampled_only=True)["assistant"]`` is
        the count of tokens the model actually emitted across all
        assistant turns — template scaffolding excluded.
        ``rendered.tokens_by_role()["tool"]`` is the raw count of
        tool-response tokens (``sampled_only`` is zero for ``tool`` by
        construction since the model never samples those).

        Roles present in :attr:`message_roles` always appear in the
        returned dict, even with post-filter count ``0``, so callers
        can index directly without ``KeyError`` on conversations that
        happen to lack a role. Returns an empty dict if
        :attr:`message_roles` is empty.
        """
        counts = self.tokens_per_message(sampled_only=sampled_only)
        out: dict[str, int] = {}
        for role, n in zip(self.message_roles, counts):
            out[role] = out.get(role, 0) + n
        return out

    def content_token_spans_by_role(self) -> dict[str, list[tuple[int, int]]]:
        """Per-role spans of contiguous body-only tokens (``is_content=True``).

        Maps each role appearing in :attr:`message_roles` to a list of
        half-open ``[start, end)`` slices into :attr:`token_ids` over
        which every token satisfies ``is_content=True`` AND belongs to
        a message of that role. Spans never cross message boundaries:
        a tool message contributes its own runs; an immediately
        adjacent assistant message contributes separate runs even when
        the bodies abut on the token axis.

        Returns an empty dict when :attr:`is_content` or
        :attr:`message_roles` is empty (renderer didn't populate the
        signal — e.g. ``DefaultRenderer``).

        Intended for selective loss masking: SFT on tool response
        bodies while RL acts only on assistant turns is the canonical
        case::

            spans = rendered.content_token_spans_by_role()
            tool_sft_mask = [False] * len(rendered.token_ids)
            for s, e in spans.get("tool", []):
                for k in range(s, e):
                    tool_sft_mask[k] = True

        See also :meth:`content_mask_for_roles` for the same
        computation returned as a per-token bool list.
        """
        out: dict[str, list[tuple[int, int]]] = {}
        if not self.is_content or not self.message_roles:
            return out
        n = len(self.token_ids)
        if len(self.is_content) != n or len(self.message_indices) != n:
            return out

        msg_spans = self.message_token_spans()
        for role, span in zip(self.message_roles, msg_spans):
            bucket = out.setdefault(role, [])
            if span is None:
                continue
            start, end = span
            run_start: int | None = None
            for k in range(start, end):
                if self.is_content[k]:
                    if run_start is None:
                        run_start = k
                else:
                    if run_start is not None:
                        bucket.append((run_start, k))
                        run_start = None
            if run_start is not None:
                bucket.append((run_start, end))
        return out

    def content_mask_for_roles(self, roles: "set[str] | frozenset[str]") -> list[bool]:
        """Per-token bool list: ``True`` iff the token is body of a
        message whose role is in ``roles``.

        Length matches :attr:`token_ids`. Returns an all-``False``
        list of that length when :attr:`is_content` or
        :attr:`message_roles` is empty — consumers can AND this with
        their own attribution masks without length checks.

        ``role_to_mask`` style helpers in :func:`build_training_sample`
        cover the trainable-role question; this one covers the
        complementary "body-only" question. The two compose: SFT mask
        on tool body is
        ``rendered.content_mask_for_roles({"tool"})``; RL mask on
        assistant tokens stays
        ``[s and (mi >= 0 and rendered.message_roles[mi] == "assistant")
        for s, mi in zip(rendered.sampled_mask, rendered.message_indices)]``.
        """
        n = len(self.token_ids)
        mask = [False] * n
        if not self.is_content or not self.message_roles:
            return mask
        if len(self.is_content) != n or len(self.message_indices) != n:
            return mask

        for k, msg_idx in enumerate(self.message_indices):
            if msg_idx < 0:
                continue
            if msg_idx >= len(self.message_roles):
                continue
            if self.message_roles[msg_idx] in roles and self.is_content[k]:
                mask[k] = True
        return mask


class ToolCallParseStatus(str, enum.Enum):
    """Per-attempt outcome of parsing a single ``<tool_call>`` block.

    The renderer parser's job is JSON-syntax → ``dict`` (the parser-level
    contract). Schema validation — required fields, argument types, tool
    name lookup — is the *tool*'s job and is intentionally not done here.
    See ``ParsedToolCall.status`` for what each value means.

    Diverges from vLLM/SGLang on purpose. Both engines collapse parse
    failures into either a single ``tools_called: bool`` (vLLM) or silent
    drops (SGLang), with no way to express "the model emitted three
    parallel tool calls and the second was malformed." Renderers expose
    that information because verifier / RL-loss code needs it for
    schema-adherence rubrics and selective token masking — use cases the
    inference engines don't serve.
    """

    OK = "ok"
    INVALID_JSON = "invalid_json"  # body wasn't valid JSON
    UNCLOSED_BLOCK = "unclosed_block"  # opening delim hit EOS / stop
    MISSING_NAME = "missing_name"  # parsed structurally, but no function name
    MALFORMED_STRUCTURE = "malformed_structure"  # format-specific shape error


@dataclass
class ParsedToolCall:
    """A single ``<tool_call>`` block as the renderer parsed it.

    One record per *attempt* — successful and malformed calls both land
    here, distinguished by ``status``. Ordering is preserved across the
    response, so ``[OK, INVALID_JSON, OK]`` is a faithful record of "the
    model emitted three parallel calls; the second was broken."

    ``token_span`` is a half-open ``[start, end)`` slice into the
    completion's stripped token id stream (i.e. ``token_ids`` after
    ``_strip_stop_tokens``); some text-based parsers can't cheaply
    recover token offsets and leave it ``None``. Useful for trainer-side
    selective loss masking: zero the mask over the spans of non-OK
    entries to avoid reinforcing malformed structures.

    ``raw`` is the decoded text of the block as the model emitted it
    (before any JSON normalization). Always populated — for failed
    attempts it's the only way to see what actually went wrong.
    """

    raw: str
    name: str | None = None
    arguments: dict[str, Any] | str | None = None
    token_span: tuple[int, int] | None = None
    status: ToolCallParseStatus = ToolCallParseStatus.OK
    id: str | None = None  # native tool-call id when the format carries one (Kimi K2)


@dataclass
class ParsedResponse:
    """Result of parsing completion tokens back into a structured message.

    ``tool_calls`` is a list of every parse attempt — successful and
    malformed alike. Filter with ``[tc for tc in r.tool_calls if
    tc.status == ToolCallParseStatus.OK]`` to get only the calls that
    came out clean. Empty list = the model didn't emit any tool calls
    (different from "tried and failed entirely", which produces a list
    with non-OK entries).
    """

    content: str
    reasoning_content: str | None = None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


@dataclass
class RenderedConversation:
    """Exact token state for a rendered conversation."""

    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    parsed_completion: ParsedResponse | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.completion_ids

    def with_completion(
        self,
        completion_ids: list[int],
        *,
        completion_logprobs: list[float] | None = None,
        parsed_completion: ParsedResponse | None = None,
    ) -> "RenderedConversation":
        return RenderedConversation(
            prompt_ids=list(self.prompt_ids),
            completion_ids=list(completion_ids),
            completion_logprobs=list(completion_logprobs or []),
            messages=list(self.messages),
            parsed_completion=parsed_completion,
        )


@runtime_checkable
class Renderer(Protocol):
    """Owns message ↔ token conversion for a specific model family."""

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        """Render messages to token IDs with per-token message attribution.

        Behaviour around historical ``reasoning_content`` is owned by the
        renderer instance — the ``preserve_all_thinking`` and
        ``preserve_thinking_between_tool_calls`` flags are constructor
        kwargs, not call-site kwargs. To render with a different
        configuration, build a different renderer (or different pool).
        Defaults preserve byte-identity with each model's chat template;
        flipping a flag at construction restores ``reasoning_content``
        the template would otherwise drop. See
        ``should_preserve_past_thinking`` for the per-message
        classification.
        """
        ...

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render messages to token IDs (without attribution metadata)."""
        ...

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        """Parse completion tokens back into a structured message.

        ``tools`` is the same list passed to ``render`` for this turn.
        XML-style formats (Qwen3.5, GLM, MiniMax, Laguna) render argument
        values verbatim inside ``<arg_value>`` tags with no quoting, so
        a value like ``true`` is ambiguous between bool and the string
        ``"true"``. When ``tools`` is supplied, the parser consults each
        parameter's declared JSON-schema type to preserve string args
        verbatim. Without ``tools``, parsers fall back to the historical
        ``json.loads``-with-text-fallback behavior.
        """
        ...

    def get_stop_token_ids(self) -> list[int]:
        """Return token IDs that signal generation should stop."""
        ...

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> "RenderedTokens | None":
        """Extend ``prev_prompt_ids + prev_completion_ids`` with the tokens
        the next turn adds, without re-rendering the sampled tokens.

        Contract: if the return value's ``token_ids`` sequence ``B`` is
        not None, then
        ``B[: len(prev_prompt) + len(prev_completion)] == prev_prompt + prev_completion``
        and ``B`` ends at the position where the next assistant turn
        begins generating (i.e. equivalent to rendering the full message
        list so far with ``add_generation_prompt=True`` — except prev
        sampled tokens are kept verbatim rather than re-rendered).

        Attribution on the returned ``RenderedTokens``:

        - ``message_indices`` is ``-1`` over the entire prior portion
          (length ``len(previous_ids)`` after :func:`trim_to_turn_close`)
          because the bridge gets the prior as raw token lists with no
          attribution. Over the bridge-added portion, indices are
          relative to ``new_messages``: a token rendered as part of
          ``new_messages[i]`` carries ``i``, and inter-turn separators /
          the trailing generation prompt carry ``-1``. So
          ``bridge.tokens_per_message(len(new_messages))`` gives the
          per-new-message token count for length-penalty bookkeeping.
        - ``sampled_mask`` is uniformly ``False`` across the entire
          returned sequence. The bridge output is consumed as the next
          turn's prompt; nothing it emits was model-sampled, and the
          bridge has no way to recover which prior tokens were. If the
          caller needs that distinction for the prior portion, they
          have it directly: every token in ``prev_completion_ids`` was
          sampled; every token in ``prev_prompt_ids`` was not.
        - ``is_content`` mirrors ``sampled_mask``'s scheme for the
          prior portion (uniformly ``False`` — body-vs-wrap
          attribution can't be recovered from raw token ids), and on
          the bridge-added portion the renderer populates it the same
          way as in :meth:`render`: ``True`` over the body bytes of
          each new message, ``False`` over the surrounding scaffold.
          Consumers walk the trajectory and read each step's own
          ``is_content`` for full-conversation body masks; the bridge
          output covers only the *new* tokens this turn adds.

        Text-only renderers return :class:`RenderedTokens` with
        ``multi_modal_data=None``. Multimodal renderers (see
        :class:`MultimodalRenderer`) populate ``multi_modal_data`` so
        the caller can recover placeholder offsets + per-item processed
        tensors for the new full prompt; they also accept a
        ``previous_multi_modal_data`` kwarg via the
        :class:`MultimodalRenderer` Protocol override.

        Return ``None`` whenever the renderer can't prove that contract
        holds — the caller falls back to a full re-render. In particular,
        bridges refuse assistant messages in ``new_messages`` (those would
        re-tokenize model-sampled content). Hand-coded renderers know their
        canonical close and synthesise it on truncated priors;
        DefaultRenderer always returns ``None`` because the template's
        close is unknown.
        """
        ...


@runtime_checkable
class MultimodalRenderer(Renderer, Protocol):
    """A :class:`Renderer` that supports multimodal inputs (images, video).

    Concrete classes (``Qwen3VLRenderer``, ``Qwen35Renderer``,
    ``Qwen36Renderer``, ``KimiK25Renderer``) implement this Protocol
    structurally — no explicit inheritance required. Callers that need
    to drive vLLM's ``multi_modal_data`` features field or carry images
    forward across turns should dispatch on ``isinstance(r,
    MultimodalRenderer)`` and use the extended ``bridge_to_next_turn``
    signature below.
    """

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        """Map from special-token IDs to per-token modality markers.

        Convention: ``1`` = image placeholder (e.g. ``<|image_pad|>``),
        ``2`` = video placeholder (e.g. ``<|video_pad|>``). The
        orchestrator stamps these onto each rendered token to drive
        the trainer's vision-encoder slicing logic.
        """
        ...

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        previous_multi_modal_data: "MultiModalData | None" = None,
    ) -> "RenderedTokens | None":
        """Same contract as :meth:`Renderer.bridge_to_next_turn`, plus:

        - accepts ``previous_multi_modal_data`` so prior-turn images
          carry forward into the new prompt's ``mm_placeholders``;
          without this, vLLM sees placeholder counts that don't match
          the combined token sequence and silently falls back to
          hash-cache lookup (or errors)
        - returns :class:`RenderedTokens` (not ``list[int]``) so the
          caller can recover the placeholder offsets + per-item
          processed tensors for the new full prompt
        """
        ...


# Per-type cache for ``is_multimodal``. The ``runtime_checkable`` Protocol
# isinstance check walks every protocol member via ``hasattr`` on each
# call; per-type caching collapses that to a single dict lookup on the
# hot path (e.g. per-bridge dispatch). Pools expose ``is_multimodal``
# directly as a snapshot attribute (different pools share a class but
# wrap different renderer types), so we don't need to special-case them.
_IS_MULTIMODAL_BY_TYPE: dict[type, bool] = {}


def is_multimodal(r: object) -> bool:
    """True iff ``r`` satisfies the :class:`MultimodalRenderer` protocol.

    Equivalent to ``isinstance(r, MultimodalRenderer)`` but cached. Use
    this on hot paths (per-rollout, per-bridge dispatch) instead of
    re-running the runtime_checkable Protocol walk on every call.
    """
    direct = getattr(r, "is_multimodal", None)
    if isinstance(direct, bool):
        return direct
    cls = type(r)
    cached = _IS_MULTIMODAL_BY_TYPE.get(cls)
    if cached is None:
        cached = isinstance(r, MultimodalRenderer)
        _IS_MULTIMODAL_BY_TYPE[cls] = cached
    return cached


class RendererPool:
    """Pool of Renderer instances that itself satisfies the Renderer protocol.

    Callers treat a pool like a single renderer — ``pool.render_ids(...)``,
    ``pool.bridge_to_next_turn(...)``, ``isinstance(pool, MultimodalRenderer)``
    all work via structural delegation. The pool internally serializes
    access to its inner renderers (each wraps its own tokenizer copy).

    Concurrency model:
    - ``size == 1``: a single inner renderer guarded by a ``threading.Lock``.
      Avoids the queue's per-call overhead on the common default config.
    - ``size > 1``: a ``queue.Queue`` of independent renderers, checked out
      one at a time. HuggingFace fast tokenizers release the GIL during
      Rust encoding, so threads achieve real parallelism.

    Construction parallelism for ``size > 1``: ``AutoTokenizer.from_pretrained``
    takes hundreds of ms per call (JSON parse + Rust tokenizer build + HF
    cache lookup), so populating a 32-slot pool serially costs ~10-15s on
    startup and shows up directly as a step-0 stall. We fan the factory out
    across a short-lived thread pool; the GIL-bound Python portion stops
    scaling past ~8 workers, so we clamp there.
    """

    def __init__(self, factory: Callable[[], Renderer], size: int):
        from concurrent.futures import ThreadPoolExecutor

        self._factory = factory
        self._size = size

        if size == 1:
            renderer = factory()
            self._sole: Renderer | None = renderer
            self._lock: threading.Lock | None = threading.Lock()
            self._pool: queue.Queue[Renderer] | None = None
            sample: Renderer = renderer
        else:
            self._sole = None
            self._lock = None
            self._pool = queue.Queue(maxsize=size)
            workers = min(size, 8)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for renderer in executor.map(lambda _: factory(), range(size)):
                    self._pool.put(renderer)
            # Peek without removing — safe at construction time before any
            # checkout has been served.
            sample = self._pool.queue[0]

        # Snapshot the protocol-shaped attributes from a sample renderer.
        # They are constant per renderer class, so resolving them once at
        # construction (a) eliminates per-call ``getattr``/``isinstance``
        # overhead and (b) lets a future out-of-process pool variant skip
        # holding a live tokenizer in the parent process.
        self._renderer_cls: type[Renderer] = type(sample)
        self.supports_tools: bool = getattr(sample, "supports_tools", True)
        self.is_multimodal: bool = is_multimodal(sample)
        # ``mm_token_type_id_map`` is set ONLY on pools wrapping a
        # ``MultimodalRenderer``. We deliberately don't expose this as a
        # class-level property: ``runtime_checkable`` Protocol's
        # isinstance check uses ``inspect.getattr_static``, which finds
        # property descriptors on the class regardless of whether their
        # fget raises. Conditional instance attributes (present in
        # ``self.__dict__`` only when applicable) are the only way to
        # make ``isinstance(pool, MultimodalRenderer)`` reflect the
        # inner renderer's actual protocol conformance.
        if isinstance(sample, MultimodalRenderer):
            self.mm_token_type_id_map: dict[int, int] = sample.mm_token_type_id_map

    @contextmanager
    def checkout(self):
        if self._sole is not None:
            assert self._lock is not None
            with self._lock:
                yield self._sole
            return
        assert self._pool is not None
        renderer = self._pool.get()
        try:
            yield renderer
        finally:
            self._pool.put(renderer)

    @property
    def size(self) -> int:
        return self._size

    @property
    def renderer_cls(self) -> type[Renderer]:
        """Class of the renderers in this pool (uniform across all slots)."""
        return self._renderer_cls

    # ── Renderer protocol delegation ────────────────────────────────────
    # Pool structurally satisfies ``Renderer`` (and ``MultimodalRenderer``
    # when its slots wrap multimodal renderers). Callers can call methods
    # directly and dispatch with ``isinstance(pool, MultimodalRenderer)``
    # without reaching into ``checkout()``.

    def render(self, *args: Any, **kwargs: Any) -> "RenderedTokens":
        with self.checkout() as r:
            return r.render(*args, **kwargs)

    def render_ids(self, *args: Any, **kwargs: Any) -> list[int]:
        with self.checkout() as r:
            return r.render_ids(*args, **kwargs)

    def parse_response(self, *args: Any, **kwargs: Any) -> "ParsedResponse":
        with self.checkout() as r:
            return r.parse_response(*args, **kwargs)

    def get_stop_token_ids(self) -> list[int]:
        with self.checkout() as r:
            return r.get_stop_token_ids()

    def bridge_to_next_turn(self, *args: Any, **kwargs: Any) -> "RenderedTokens | None":
        with self.checkout() as r:
            return r.bridge_to_next_turn(*args, **kwargs)

    # ``mm_token_type_id_map`` (the MultimodalRenderer protocol attribute)
    # is set in ``__init__`` only for pools wrapping multimodal renderers;
    # see the comment there for why this isn't a class-level property.


RENDERER_REGISTRY: dict[str, type] = {}

# Exact canonical HF model names → renderer. We do NOT use prefix
# matching because models with the same architecture may ship different
# chat templates (base vs instruct, tuned vs pretrained) — matching on
# prefix silently routes them to a renderer that doesn't produce
# template-parity output. Fine-tunes and renamed checkpoints MUST pass
# ``renderer=<name>`` explicitly; the auto path falls back to
# ``DefaultRenderer`` (which uses ``apply_chat_template`` verbatim) and
# logs a loud INFO line with the chosen fallback.
MODEL_RENDERER_MAP: dict[str, str] = {
    # Qwen3 — base and Instruct variants share the same chat template.
    "Qwen/Qwen3-0.6B": "qwen3",
    "Qwen/Qwen3-1.7B": "qwen3",
    "Qwen/Qwen3-4B": "qwen3",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3",
    "Qwen/Qwen3-4B-Thinking-2507": "qwen3",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-14B": "qwen3",
    "Qwen/Qwen3-32B": "qwen3",
    "Qwen/Qwen3-30B-A3B": "qwen3",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen3",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "qwen3",
    "Qwen/Qwen3-235B-A22B": "qwen3",
    # Qwen3.5. All seven sizes share the same renderer. The 4B / 9B /
    # 35B-A3B / 122B-A10B / 397B-A17B chat template defaults
    # ``enable_thinking=true`` (open ``<think>\n`` at the gen prompt);
    # the smaller 0.8B / 2B variants flip the polarity (default
    # ``enable_thinking=false``, empty ``<think>\n\n</think>\n\n``).
    # ``Qwen35Renderer`` hard-codes this polarity per model
    # (``_ENABLE_THINKING_DEFAULTS``), so all seven sizes are
    # token-for-token parity-tested against their own
    # ``apply_chat_template`` — including with
    # ``add_generation_prompt=True``.
    "Qwen/Qwen3.5-0.8B": "qwen3.5",
    "Qwen/Qwen3.5-2B": "qwen3.5",
    "Qwen/Qwen3.5-4B": "qwen3.5",
    "Qwen/Qwen3.5-9B": "qwen3.5",
    "Qwen/Qwen3.5-35B-A3B": "qwen3.5",
    "Qwen/Qwen3.5-122B-A10B": "qwen3.5",
    "Qwen/Qwen3.5-397B-A17B": "qwen3.5",
    # Qwen3.6.
    "Qwen/Qwen3.6-35B-A3B": "qwen3.6",
    # Qwen3-VL.
    "Qwen/Qwen3-VL-4B-Instruct": "qwen3-vl",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3-vl",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen3-vl",
    # GLM-5 family (GLM-4.7 reuses the GLM-5 template).
    "zai-org/GLM-5": "glm-5",
    "zai-org/GLM-5-FP8": "glm-5",
    "zai-org/GLM-4.7-Flash": "glm-5",
    "zai-org/GLM-5.1": "glm-5.1",
    # GLM-4.5.
    "THUDM/GLM-4.5-Air": "glm-4.5",
    "zai-org/GLM-4.5-Air": "glm-4.5",
    # MiniMax.
    "MiniMaxAI/MiniMax-M2": "minimax-m2",
    "MiniMaxAI/MiniMax-M2.5": "minimax-m2",
    # DeepSeek V3.
    "deepseek-ai/DeepSeek-V3": "deepseek-v3",
    "deepseek-ai/DeepSeek-V3-Base": "deepseek-v3",
    # Kimi K2 (K2.5 and K2.6 share the K2.5 template, distinct from K2).
    "moonshotai/Kimi-K2-Instruct": "kimi-k2",
    "moonshotai/Kimi-K2.5": "kimi-k2.5",
    "moonshotai/Kimi-K2.6": "kimi-k2.5",
    # Nemotron 3.
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron-3",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nemotron-3",
    # Poolside Laguna.
    "poolside/Laguna-XS.2": "laguna-xs.2",
    # GPT-OSS.
    "openai/gpt-oss-20b": "gpt-oss",
    "openai/gpt-oss-120b": "gpt-oss",
}


# Per-model declaration of supported non-text modalities. Drives the
# multimodal parity test matrix in ``tests/test_multimodal.py`` — each
# ``(model, modality)`` pair gets a parity test against
# ``processor.apply_chat_template`` + ``processor(...)``. Add a model
# here when its renderer supports a new modality; the test matrix
# picks it up automatically.
#
# Modality values: ``"image"``, ``"video"``, ``"audio"``. Text is implicit
# (every model supports it), so it doesn't appear in the set.
MULTIMODAL_MODELS: dict[str, set[str]] = {
    "Qwen/Qwen3-VL-4B-Instruct": {"image"},
    "Qwen/Qwen3-VL-8B-Instruct": {"image"},
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {"image"},
    # Qwen3.5 is itself a VLM family (HF tag ``image-text-to-text``,
    # processor class ``Qwen3VLProcessor``) — same vision tokens and
    # image-processor as Qwen3-VL, with a different tool-call format.
    "Qwen/Qwen3.5-0.8B": {"image"},
    "Qwen/Qwen3.5-2B": {"image"},
    "Qwen/Qwen3.5-4B": {"image"},
    "Qwen/Qwen3.5-9B": {"image"},
    "Qwen/Qwen3.5-35B-A3B": {"image"},
    "Qwen/Qwen3.5-122B-A10B": {"image"},
    "Qwen/Qwen3.5-397B-A17B": {"image"},
    # Qwen3.6 extends Qwen3.5's chat template; same VL bits, only
    # tool-call argument serialization differs.
    "Qwen/Qwen3.6-35B-A3B": {"image"},
    # Kimi K2.5 / K2.6 are unified VLMs (HF tag ``image-text-to-text``)
    # with custom processor (``KimiK25Processor`` + ``KimiK25VisionProcessor``).
    # Vision wrap is different from Qwen-VL:
    # ``<|media_begin|>image<|media_content|><|media_pad|><|media_end|>`` —
    # only ONE ``<|media_pad|>`` per image in ``input_ids``; per-patch
    # expansion happens internally in the model from ``pixel_values`` /
    # ``grid_thws``.
    "moonshotai/Kimi-K2.5": {"image"},
    "moonshotai/Kimi-K2.6": {"image"},
}


def _model_has_vision_config(model_name: str) -> bool:
    """Return True if the HF config for ``model_name`` declares vision inputs.

    Used by ``create_renderer`` to fail loudly on VLMs that miss the
    ``MODEL_RENDERER_MAP`` exact-match lookup. DefaultRenderer silently
    drops images (it only knows ``apply_chat_template`` + text tokens),
    so a VLM falling back to it would produce token streams that don't
    match what the trainer reconstructs — a class of bug the renderer
    abstraction exists to prevent.

    Returns False on any AutoConfig failure (offline, gated, missing) so
    a flaky HF probe never blocks a legitimate text-only fine-tune.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    except Exception:
        return False
    # Most VLM configs nest a vision tower as ``vision_config`` (Qwen-VL,
    # Llava, Gemma3, Idefics, MiniCPM-V, ...). A few use ``vision_tower``
    # or expose a top-level ``image_token_id``; check those too.
    if getattr(cfg, "vision_config", None) is not None:
        return True
    if getattr(cfg, "vision_tower", None) is not None:
        return True
    if getattr(cfg, "image_token_id", None) is not None:
        return True
    return False


# Models whose tokenizer requires ``trust_remote_code=True`` AND a pinned
# revision. Empirical audit (2026-05-07) confirms only the Moonshot
# Kimi-K2 family ships an ``auto_map.AutoTokenizer`` entry that runs
# repo-supplied Python on every ``AutoTokenizer.from_pretrained`` call —
# every other model in ``MODEL_RENDERER_MAP`` loads cleanly without it.
#
# Pinning the revision keeps the trust narrow: even with
# ``trust_remote_code=True``, transformers downloads / executes the
# tokenizer Python from this exact commit only. A future malicious push
# to the Moonshot HF repo doesn't auto-propagate to anyone using
# ``create_renderer_pool``. Bump these SHAs deliberately, with review.
TRUSTED_REVISIONS: dict[str, str] = {
    "moonshotai/Kimi-K2-Instruct": "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
    "moonshotai/Kimi-K2.5": "4d01dfe0332d63057c186e0b262165819efb6611",
    "moonshotai/Kimi-K2.6": "2755962d07cb42aa2d988a35bcb65cd4a9c2de82",
}


# Models for which ``fastokens`` is known to diverge from vanilla
# ``transformers.AutoTokenizer`` and therefore must NOT be patched.
# Empirical audit ran each entry of ``MODEL_RENDERER_MAP`` through both
# backends. The entries below fail to load under fastokens (DeepSeek-V3
# family — Metaspace pretokenizer not yet implemented).
FASTOKENS_INCOMPATIBLE: frozenset[str] = frozenset(
    {
        # fastokens: ``ValueError: pre-tokenizer error: unsupported
        # pre-tokenizer type: Metaspace`` — DeepSeek's tokenizer uses
        # SentencePiece-style Metaspace pretokenization which fastokens
        # doesn't yet implement.
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V3-Base",
    }
)


_FASTOKENS_PATCH_LOCK = threading.Lock()
_FASTOKENS_ANNOUNCED = False


def _patched_load(model_name_or_path: str, **kwargs):
    """Run ``AutoTokenizer.from_pretrained`` with fastokens patched in
    process-locally — patch around the load, unpatch right after.

    fastokens captures the loaded backend on a per-tokenizer basis, so
    after we unpatch the returned tokenizer object continues to use
    fastokens for ``encode``/``decode`` while subsequent
    ``AutoTokenizer.from_pretrained`` calls (outside our control) go
    back to vanilla. This keeps the global side effect minimal.

    fastokens itself prints ``[fastokens] patch_transformers: ...`` to
    stdout on every patch/unpatch call. Building a pool of size N would
    therefore emit ~N lines (more under thread contention, where some
    threads see ``already patched``). We swallow those prints under a
    lock — ``contextlib.redirect_stdout`` swaps ``sys.stdout``
    process-wide, so the lock keeps unrelated stdout writes from other
    threads from disappearing into our buffer. The patch/unpatch calls
    are cheap; only the brief patch+unpatch is serialized, the actual
    ``from_pretrained`` still runs concurrently across pool slots. A
    single ``logger.info`` is emitted on the first patch so the fast
    path is still discoverable in logs.
    """
    import fastokens

    global _FASTOKENS_ANNOUNCED

    with _FASTOKENS_PATCH_LOCK:
        with contextlib.redirect_stdout(io.StringIO()):
            fastokens.patch_transformers()
        if not _FASTOKENS_ANNOUNCED:
            logger.info(
                "fastokens enabled — tokenizers load through the Rust BPE fast path (~10x encode speedup)."
            )
            _FASTOKENS_ANNOUNCED = True
    try:
        return _load_tokenizer_via_auto(model_name_or_path, **kwargs)
    finally:
        with _FASTOKENS_PATCH_LOCK:
            with contextlib.redirect_stdout(io.StringIO()):
                fastokens.unpatch_transformers()


def _load_fast_tokenizer_directly(
    model_name_or_path: str, revision: str | None
) -> Any | None:
    """Load a self-contained fast tokenizer without building the model config.

    ``AutoTokenizer.from_pretrained`` eagerly constructs the *model* config to
    resolve the tokenizer class — even for a plain ``PreTrainedTokenizerFast``.
    That construction can raise on modeling-only concerns the tokenizer never
    needs (e.g. RoPE parameter validation for configs that carry nested
    ``rope_parameters``). When the repo ships a complete ``tokenizer.json`` and
    declares no custom tokenizer, the tokenizer is fully self-describing, so we
    load it directly and skip the config detour.

    Returns ``None`` when there's nothing safe to load this way — a custom
    ``auto_map`` tokenizer (which must run through ``AutoTokenizer`` with
    ``trust_remote_code``) or no fast tokenizer at all — so the caller can
    surface its original error instead.
    """
    from transformers import PreTrainedTokenizerFast
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    try:
        if "auto_map" in get_tokenizer_config(model_name_or_path, revision=revision):
            return None
        return PreTrainedTokenizerFast.from_pretrained(
            model_name_or_path, revision=revision
        )
    except Exception:
        return None


def _load_tokenizer_via_auto(model_name_or_path: str, **kwargs) -> Any:
    """``AutoTokenizer.from_pretrained`` with a config-free fallback.

    renderers needs the tokenizer, not the model. If ``AutoTokenizer`` fails
    while building the model config it loads to resolve the tokenizer class,
    retry by loading the repo's self-contained ``tokenizer.json`` directly. The
    original error is re-raised if the repo has no such tokenizer.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    except Exception as exc:
        tok = _load_fast_tokenizer_directly(
            model_name_or_path, revision=kwargs.get("revision")
        )
        if tok is None:
            raise
        logger.debug(
            "AutoTokenizer.from_pretrained(%r) failed building the model config "
            "(%s: %s); loaded the tokenizer directly from tokenizer.json.",
            model_name_or_path,
            type(exc).__name__,
            str(exc)[:160],
        )
        return tok


def load_tokenizer(
    model_name_or_path: str,
    *,
    use_fastokens: bool = True,
):
    """Load a tokenizer with the renderers-package security + perf policy.

    **Security** — default ``trust_remote_code=False``. Models listed in
    ``TRUSTED_REVISIONS`` (Moonshot Kimi-K2 family) load with
    ``trust_remote_code=True`` AND a pinned ``revision=<sha>`` so
    transformers only executes the reviewed commit's tokenizer Python.

    **Performance** — ``use_fastokens=True`` (default) routes the load
    through ``fastokens.patch_transformers()`` so the resulting tokenizer
    encodes ~10x faster than vanilla ``tokenizers``. The patch is
    bracketed: it's applied before ``from_pretrained`` and removed
    immediately after, so global ``AutoTokenizer.from_pretrained`` calls
    elsewhere in the user's process are not affected.

    Models in ``FASTOKENS_INCOMPATIBLE`` (DeepSeek-V3 family) skip the
    patch — fastokens currently fails to load them. Pass
    ``use_fastokens=False`` to force the vanilla backend for any other
    model.

    Unknown / fine-tuned model paths fall through to
    ``trust_remote_code=False`` and the patched-load fast path. If
    fastokens raises during the patched load (e.g. an unknown
    pre-tokenizer type), we automatically retry with the vanilla
    backend and emit an INFO log.

    ``AutoTokenizer.from_pretrained`` eagerly builds the model config to
    resolve the tokenizer class. If that construction raises on a
    modeling-only concern the tokenizer doesn't need (e.g. RoPE
    validation for configs with nested ``rope_parameters``), we fall
    back to loading the repo's self-contained ``tokenizer.json``
    directly — see ``_load_tokenizer_via_auto``.
    """
    kwargs: dict[str, Any] = {}
    revision = TRUSTED_REVISIONS.get(model_name_or_path)
    if revision is not None:
        kwargs = {"trust_remote_code": True, "revision": revision}
    else:
        kwargs = {"trust_remote_code": False}

    if not use_fastokens or model_name_or_path in FASTOKENS_INCOMPATIBLE:
        return _load_tokenizer_via_auto(model_name_or_path, **kwargs)

    try:
        return _patched_load(model_name_or_path, **kwargs)
    except Exception as exc:
        logger.info(
            "fastokens could not load %r (%s: %s); falling back to vanilla "
            "AutoTokenizer. Add this model to FASTOKENS_INCOMPATIBLE in "
            "renderers.base to suppress the retry.",
            model_name_or_path,
            type(exc).__name__,
            str(exc)[:160],
        )
        return _load_tokenizer_via_auto(model_name_or_path, **kwargs)


def _populate_registry():
    if RENDERER_REGISTRY:
        return
    from renderers.deepseek_v3 import DeepSeekV3Renderer
    from renderers.default import DefaultRenderer
    from renderers.glm5 import GLM5Renderer, GLM51Renderer
    from renderers.glm45 import GLM45Renderer
    from renderers.gpt_oss import GptOssRenderer
    from renderers.kimi_k2 import KimiK2Renderer
    from renderers.kimi_k25 import KimiK25Renderer
    from renderers.laguna_xs2 import LagunaXS2Renderer
    from renderers.minimax_m2 import MiniMaxM2Renderer
    from renderers.nemotron3 import Nemotron3Renderer
    from renderers.qwen3 import Qwen3Renderer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from renderers.qwen35 import Qwen35Renderer
    from renderers.qwen36 import Qwen36Renderer

    RENDERER_REGISTRY.update(
        {
            "default": DefaultRenderer,
            "qwen3": Qwen3Renderer,
            "qwen3-vl": Qwen3VLRenderer,
            "qwen3.5": Qwen35Renderer,
            "qwen3.6": Qwen36Renderer,
            "glm-5": GLM5Renderer,
            "glm-5.1": GLM51Renderer,
            "glm-4.5": GLM45Renderer,
            "minimax-m2": MiniMaxM2Renderer,
            "deepseek-v3": DeepSeekV3Renderer,
            "kimi-k2": KimiK2Renderer,
            "kimi-k2.5": KimiK25Renderer,
            "laguna-xs.2": LagunaXS2Renderer,
            "nemotron-3": Nemotron3Renderer,
            "gpt-oss": GptOssRenderer,
        }
    )


def create_renderer_pool(
    tokenizer_name_or_path: str,
    config: RendererConfig | None = None,
    *,
    size: int = 16,
) -> RendererPool:
    """Create a RendererPool with *size* independent tokenizer copies.

    Each slot loads its own tokenizer so threads never share mutable
    state. HuggingFace fast tokenizers release the GIL during Rust
    encoding, so threads achieve real parallelism.

    ``config`` is the typed renderer config (one of the variants of
    :data:`renderers.RendererConfig`). Defaults to
    :class:`AutoRendererConfig`, which resolves to a concrete renderer
    via ``MODEL_RENDERER_MAP`` at construction time using the loaded
    tokenizer's name. Every slot in the pool shares the same config; to
    run a different config, build a different pool.

    Tokenizers load via ``load_tokenizer`` — see its docstring for the
    ``trust_remote_code`` policy (default off; Moonshot Kimi-K2 family
    opts in with a pinned ``revision``).
    """

    def factory() -> Renderer:
        tokenizer = load_tokenizer(tokenizer_name_or_path)
        return create_renderer(tokenizer, config)

    return RendererPool(factory, size=size)


def create_renderer(
    tokenizer,
    config: RendererConfig | None = None,
) -> Renderer:
    """Create a Renderer from a typed config.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        config: Typed renderer config — one of the variants of
            :data:`renderers.RendererConfig`. ``None`` defaults to
            :class:`AutoRendererConfig`, which resolves to a concrete
            renderer using ``tokenizer.name_or_path`` against
            ``MODEL_RENDERER_MAP``. To enable structured-output parsing
            on the default renderer, pass :class:`DefaultRendererConfig`
            with ``tool_parser`` / ``reasoning_parser`` set. To override
            template-control kwargs (e.g. ``enable_thinking``), pass
            the specific :class:`Qwen3RendererConfig`,
            :class:`GLM5RendererConfig` etc. and set those fields.

    Selecting the auto-renderer for a model without a registered
    renderer falls back to :class:`DefaultRenderer` for text-only models
    and raises for VLMs (where ``apply_chat_template`` would silently
    drop images).
    """
    from renderers.configs import AutoRendererConfig

    _populate_registry()

    if config is None:
        config = AutoRendererConfig()

    if not isinstance(config, AutoRendererConfig):
        cls = RENDERER_REGISTRY.get(config.name)
        if cls is None:
            raise ValueError(
                f"Unknown renderer {config.name!r}. Available: {', '.join(sorted(RENDERER_REGISTRY))}"
            )
        return cls(tokenizer, config)

    return _resolve_auto(tokenizer, config)


def _resolve_auto(tokenizer, auto: AutoRendererConfig) -> Renderer:
    """Map ``AutoRendererConfig`` → concrete typed config via the
    tokenizer's ``name_or_path``, then instantiate the matching renderer.

    Fine-tunes and renamed checkpoints miss on purpose — their chat
    template may differ from the original even when the architecture
    matches, so silently mapping them would produce template-parity
    bugs. Set ``config=<typed renderer config>`` explicitly for those.
    """
    from renderers.configs import DefaultRendererConfig, _config_class_for

    model_name = getattr(tokenizer, "name_or_path", "")
    renderer_name = MODEL_RENDERER_MAP.get(model_name)

    preserve_carry = {
        "preserve_all_thinking": auto.preserve_all_thinking,
        "preserve_thinking_between_tool_calls": auto.preserve_thinking_between_tool_calls,
    }

    if renderer_name is not None:
        cfg_cls = _config_class_for(renderer_name)
        return RENDERER_REGISTRY[renderer_name](tokenizer, cfg_cls(**preserve_carry))

    # No match. For VLMs this must be fatal: DefaultRenderer only knows
    # ``apply_chat_template`` + text tokens, so it would silently drop
    # images and produce a token stream the trainer can't reconstruct.
    # Catch this at the renderer-selection seam — well before any
    # rollout — so the failure mode is "config error at startup," not
    # "mysterious KL divergence after 100 steps."
    if model_name in MULTIMODAL_MODELS or _model_has_vision_config(model_name):
        supported_vlms = sorted(MULTIMODAL_MODELS)
        raise ValueError(
            f"No multimodal renderer registered for {model_name!r}, and "
            f"DefaultRenderer would silently drop images. Register a "
            f"renderer in MODEL_RENDERER_MAP (currently supported VLMs: "
            f"{supported_vlms}), or pass an explicit typed renderer "
            f"config if you know what you're doing."
        )

    # Text-only fall back to default (apply_chat_template). For fine-tunes
    # with customized chat templates this is the *correct* choice, so we
    # don't warn. Note the pick at INFO and advertise the parser knobs.
    if auto.preserve_all_thinking or auto.preserve_thinking_between_tool_calls:
        raise NotImplementedError(
            "Auto-resolved DefaultRenderer can't selectively re-emit "
            "dropped reasoning_content. Pass an explicit typed renderer "
            "config (model-specific) if you need preserve_*_thinking."
        )
    logger.info(
        "No model-specific renderer matched %r. Using DefaultRenderer "
        "(apply_chat_template). Pass DefaultRendererConfig(tool_parser=..., "
        "reasoning_parser=...) to enable structured output parsing.",
        model_name or "<unnamed tokenizer>",
    )
    return RENDERER_REGISTRY["default"](tokenizer, DefaultRendererConfig())


# ---------------------------------------------------------------------------
# Standalone helpers that work with any Renderer implementation
# ---------------------------------------------------------------------------


def build_training_sample(
    renderer: Renderer,
    messages: list[Message],
    *,
    role_to_mask: Callable[[Message], bool] | None = None,
    tools: list[ToolSpec] | None = None,
    content_sft_roles: "set[str] | frozenset[str] | None" = None,
) -> tuple[list[int], list[bool]]:
    """Build (token_ids, loss_mask) for supervised training.

    Single render() call + message_indices → per-token mask.
    Replaces build_incremental_token_mask (O(N) renders → O(1)).

    When ``role_to_mask`` is omitted, ``loss_mask`` is the renderer's
    ``sampled_mask`` directly: every token the model would have
    produced at inference is trainable, regardless of which message
    it's attributed to. This is the recommended default for renderer
    callers — the renderer owns the per-token "is this model output"
    signal, so role-level filtering becomes a downstream constraint
    rather than a precondition. (Some role markers — e.g. GLM
    ``<|user|>`` / ``<|observation|>`` after a tool-calling assistant
    turn — *are* sampled by the model at inference and live inside the
    next message's span; ``sampled_mask`` captures that, but a
    naive role filter would mask them out.)

    When ``role_to_mask`` is provided, ``loss_mask`` is the AND of the
    role-based attribution and the sampled signal: only tokens the
    model would have produced at inference AND attributed to a
    trainable role pass through. Useful when the caller needs to
    restrict training to a specific role (e.g. assistant-only) even on
    a renderer whose ``sampled_mask`` already covers other roles.

    Renderers that don't populate ``sampled_mask`` (empty list) fall
    back to attribution-only masking — every token attributed to a
    trainable role is trained on, including template-injected
    ``<|im_start|>role\\n`` openers. In this fallback mode
    ``role_to_mask`` is required; calling without it raises
    ``ValueError``.

    ``content_sft_roles`` opts in additional roles for "body-only"
    supervision: for every message whose role is in this set, tokens
    with ``is_content=True`` are marked trainable even though the
    ``sampled_mask`` gate excludes them (the model never samples
    tool / user / system tokens). Template scaffolding around those
    messages — ``<|im_start|>role\\n`` openers, ``<|im_end|>``
    closers, ``<|tool_response>`` wraps, inter-turn ``\\n`` — stays
    masked out, so the model learns to anticipate the body text
    without producing the surrounding special tokens (which would
    interrupt a real rollout). The canonical use case is RL on
    assistant tokens (``role_to_mask=lambda m: m["role"] ==
    "assistant"``) plus SFT on tool response bodies
    (``content_sft_roles={"tool"}``).

    Requires the renderer to populate ``is_content`` for the body-only
    path to fire. Renderers that leave it empty (``DefaultRenderer``,
    or hand-coded renderers that haven't been wired up yet) ignore
    ``content_sft_roles`` silently — falling back to the original
    ``role_to_mask`` + ``sampled_mask`` behaviour.
    """
    rendered = renderer.render(messages, tools=tools)
    has_sampled_info = len(rendered.sampled_mask) == len(rendered.token_ids)
    has_content_info = len(rendered.is_content) == len(rendered.token_ids)
    body_roles: "frozenset[str]"
    if content_sft_roles and has_content_info:
        body_roles = frozenset(content_sft_roles)
    else:
        body_roles = frozenset()

    if role_to_mask is None and not has_sampled_info:
        raise ValueError(
            "role_to_mask is required when the renderer does not populate "
            "sampled_mask. Pass an explicit role filter (e.g. "
            "lambda m: m['role'] == 'assistant') for this renderer."
        )

    loss_mask: list[bool] = []
    for k, msg_idx in enumerate(rendered.message_indices):
        if msg_idx < 0:
            loss_mask.append(False)
            continue
        msg = messages[msg_idx]
        # Body-only path for opt-in roles. Fires only on tokens whose
        # is_content bit is set; never adds the scaffolding around the
        # message, so the model isn't supervised on emitting the role
        # tags / wraps that would derail a rollout.
        if body_roles and msg.get("role") in body_roles:
            loss_mask.append(rendered.is_content[k])
            continue
        if has_sampled_info and not rendered.sampled_mask[k]:
            loss_mask.append(False)
        elif role_to_mask is None:
            # sampled_mask alone gates the loss when no role filter is
            # supplied. ``sampled_mask[k]`` is True here (handled by the
            # branch above), so this token is trainable.
            loss_mask.append(True)
        else:
            loss_mask.append(role_to_mask(msg))
    return rendered.token_ids, loss_mask


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    max_len = min(len(a), len(b))
    for idx in range(max_len):
        if a[idx] != b[idx]:
            return idx
    return max_len


def trim_to_turn_close(
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    close_token_ids: set[int],
    *,
    synthesize_close: int | None = None,
) -> list[int] | None:
    """Return the longest prefix of ``prev_prompt + prev_completion`` that
    ends at a turn-close token, or ``None`` if none exists and
    ``synthesize_close`` is not provided.

    Scans only within ``prev_completion_ids`` — a close token in
    ``prev_prompt_ids`` is structural template scaffolding, not a turn
    boundary the current step's completion produced.

    When ``prev_completion_ids`` has no close token, the prior turn was
    truncated at max_tokens. The caller opts in to synthesising the
    canonical close by passing ``synthesize_close`` (its token id).
    Otherwise the caller falls back to a fresh re-render.

    Hand-coded renderers pass this helper a set they know describes their
    turn boundaries. DefaultRenderer can't know its template's close, so
    it doesn't call this — it returns ``None`` from ``bridge_to_next_turn``
    unconditionally.
    """
    previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
    for idx in range(len(previous_ids) - 1, len(previous_prompt_ids) - 1, -1):
        if previous_ids[idx] in close_token_ids:
            return previous_ids[: idx + 1]
    if synthesize_close is None:
        return None
    previous_ids.append(synthesize_close)
    return previous_ids


# Per-model offset-aware tokenizer cache. ``attribute_text_segments``
# uses the fast HuggingFace tokenizer's ``offset_mapping`` to attribute
# each token to its source text segment under one BPE pass. Fastokens
# (the Rust BPE we patch in by default for ~10x faster encode) does not
# track character offsets — the patched tokenizer's
# ``return_offsets_mapping=True`` raises ``NotImplementedError``. So we
# keep a parallel vanilla tokenizer per model purely for offset queries.
# Memory cost is one extra tokenizer per *unique* model name across all
# pools / renderers (the cache is process-global), independent of pool
# size.
_offset_tokenizers: dict[str, Any] = {}
_offset_tokenizers_lock = threading.Lock()


def _get_offset_tokenizer(tokenizer):
    """Return a tokenizer that supports ``return_offsets_mapping=True``.

    If ``tokenizer`` itself supports offsets, returns it unchanged.
    Otherwise loads a vanilla (non-fastokens) tokenizer from
    ``tokenizer.name_or_path`` and caches it. Raises if the tokenizer
    has no usable ``name_or_path`` — hand-coded renderers always pass
    a tokenizer loaded via ``load_tokenizer`` which does set it.
    """
    # Cheap probe: does this tokenizer already provide offsets?
    try:
        tokenizer("a", add_special_tokens=False, return_offsets_mapping=True)
        return tokenizer
    except (NotImplementedError, ValueError, TypeError):
        pass

    name_or_path = getattr(tokenizer, "name_or_path", "")
    if not name_or_path:
        raise RuntimeError(
            "Cannot construct an offset-aware tokenizer: the supplied "
            "tokenizer has no ``name_or_path`` to fall back on. Pass a "
            "tokenizer loaded via ``renderers.base.load_tokenizer``."
        )

    with _offset_tokenizers_lock:
        cached = _offset_tokenizers.get(name_or_path)
        if cached is not None:
            return cached
        from transformers import AutoTokenizer

        kwargs: dict[str, Any] = {}
        revision = TRUSTED_REVISIONS.get(name_or_path)
        if revision is not None:
            kwargs = {"trust_remote_code": True, "revision": revision}
        else:
            kwargs = {"trust_remote_code": False}
        # Explicitly vanilla — we want HF's Rust tokenizer with offset
        # tracking, not the fastokens shim. ``load_tokenizer`` would
        # patch fastokens in by default; calling
        # ``AutoTokenizer.from_pretrained`` directly here keeps the
        # fastokens patch out of this code path entirely.
        offset_tok = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        if not getattr(offset_tok, "is_fast", False):
            raise RuntimeError(
                f"Vanilla tokenizer for {name_or_path!r} is not a fast "
                "tokenizer; offset_mapping is unavailable. Hand-coded "
                "renderers require a fast tokenizer for body/scaffold "
                "attribution."
            )
        _offset_tokenizers[name_or_path] = offset_tok
        return offset_tok


def attribute_text_segments(
    tokenizer,
    segments: "list[tuple[str, bool]]",
) -> "list[tuple[int, bool]]":
    """Tokenize concatenated segments as a single BPE pass and return
    ``(token_id, is_content)`` pairs.

    ``segments`` is a list of ``(text, is_content)`` chunks the renderer
    wants to emit contiguously — for example ``[("user\\n", False),
    (content, True)]`` for a user message. Concatenation is done before
    encoding to preserve BPE merges across the wrap/body boundary; the
    resulting tokens are then attributed back to their source segment
    via the fast tokenizer's ``offset_mapping``.

    A token is attributed to the segment containing its first source
    character (``offset_mapping[k][0]``). Tokens whose first character
    falls exactly on a segment boundary are attributed to the segment
    that *starts* at that offset (the "later" segment). Zero-length
    tokens (rare; usually pre-tokenizer artefacts) are attributed to
    the most recently entered segment.

    Requires a HuggingFace fast tokenizer with offset tracking. The
    ``fastokens`` patch ``load_tokenizer`` applies by default does
    **not** track offsets — when that's the case we transparently load
    a vanilla offset-capable tokenizer for the same model and cache it
    (see :func:`_get_offset_tokenizer`). Hand-coded renderers are only
    registered for model families that ship a fast tokenizer, so a
    silent slow-tokenizer fallback isn't supported — BPE drift at the
    wrap/body boundary would defeat the whole point.

    Empty input or empty joined text returns an empty list.
    """
    if not segments:
        return []
    full_text = "".join(text for text, _ in segments)
    if not full_text:
        return []

    offset_tokenizer = _get_offset_tokenizer(tokenizer)
    encoding = offset_tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoding["input_ids"])
    offsets = list(encoding["offset_mapping"])

    # Build segment char-span lookup. Track the half-open span
    # [seg_start, seg_end) of each segment and its is_content bit.
    spans: list[tuple[int, int, bool]] = []
    pos = 0
    for text, is_content in segments:
        spans.append((pos, pos + len(text), is_content))
        pos += len(text)
    total_len = pos

    out: list[tuple[int, bool]] = []
    last_is_content = spans[-1][2] if spans else False
    for tok_id, (start, _end) in zip(token_ids, offsets):
        if start >= total_len:
            # Token's character offset is past every segment (shouldn't
            # normally happen for add_special_tokens=False, but defensive
            # against tokenizer-specific edge cases).
            out.append((tok_id, last_is_content))
            continue
        # Find the segment that contains `start`. Segments are
        # contiguous and ordered, so a linear scan is fine — the inner
        # loop runs at most len(segments) times per token and segments
        # is typically 2-3 in practice.
        is_content = last_is_content
        for seg_start, seg_end, seg_is_content in spans:
            if seg_start <= start < seg_end:
                is_content = seg_is_content
                break
        else:
            # start == total_len handled above; the remaining case is
            # an empty segment in the middle. Empty segments emit no
            # characters, so no token can land in them; fall through to
            # the last non-empty segment's bit.
            pass
        out.append((tok_id, is_content))
    return out


def reject_assistant_in_extension(new_messages: list[Message]) -> bool:
    """Return True if any message in ``new_messages`` is an assistant turn.

    Bridges refuse to re-tokenize assistant content because it would
    replace model-sampled tokens with canonical template text — violating
    the contract that sampled tokens land in training exactly as emitted.
    """
    return any(m.get("role") == "assistant" for m in new_messages)


def should_preserve_past_thinking(
    messages: list[Message],
    msg_idx: int,
    *,
    preserve_all_thinking: bool,
    preserve_thinking_between_tool_calls: bool,
) -> bool:
    """Should ``messages[msg_idx]``'s ``reasoning_content`` be emitted as
    thinking even when the chat template would drop it?

    Returns ``True`` only as an override above the template default. Each
    renderer ORs this into its own "render thinking?" condition; a result
    of ``False`` means "follow the template" (drop or keep as the template
    decides), not "force-drop".

    Override rules:

    - ``preserve_all_thinking`` — every past-asst's thinking is kept.
    - ``preserve_thinking_between_tool_calls`` — keeps thinking only
      inside the *current* tool cycle: the contiguous A-T-...-A block
      after the most recent ``user`` message, and only if that block
      contains at least one ``tool`` response. As soon as a new
      ``user`` turn arrives, the previous block becomes "older" and
      its thinking is dropped (template default), matching how most
      chat templates already handle multi-turn contexts. Use
      ``preserve_all_thinking`` if you need thinking on older blocks
      to survive the user-turn boundary too.
    """
    if preserve_all_thinking:
        return True
    if not preserve_thinking_between_tool_calls:
        return False
    # Most recent user message (or -1 if none).
    last_user = -1
    for j in range(len(messages) - 1, -1, -1):
        if messages[j].get("role") == "user":
            last_user = j
            break
    if msg_idx <= last_user:
        return False
    # The current segment must contain a tool response for it to count
    # as an in-flight tool cycle.
    return any(
        messages[j].get("role") == "tool" for j in range(last_user + 1, len(messages))
    )


def build_trajectory_step(
    renderer: Renderer,
    prompt_messages: list[Message],
    completion_messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
) -> dict[str, Any]:
    """Build prompt_ids / completion_ids / masks for a trajectory step.

    Uses common_prefix_len to find the split point because generation prompts
    may diverge from the full sequence at token boundaries (e.g., ``\\n`` vs
    ``\\n\\n`` when thinking content is empty in Qwen3.5).

    For multimodal renderers, attaches ``multi_modal_data`` keyed on the
    full message sequence (assistant text doesn't carry placeholders, so
    the full-render's mm sidecar covers every image up to and including
    the completion).
    """
    has_completion = len(completion_messages) > 0
    prompt_ids = renderer.render_ids(
        prompt_messages, tools=tools, add_generation_prompt=has_completion
    )
    full_rendered = renderer.render(prompt_messages + completion_messages, tools=tools)
    full_ids = full_rendered.token_ids

    split_idx = _common_prefix_len(prompt_ids, full_ids)
    completion_ids = full_ids[split_idx:]

    out: dict[str, Any] = {
        "prompt_ids": full_ids[:split_idx],
        "prompt_mask": [False] * split_idx,
        "completion_ids": completion_ids,
        "completion_mask": [True] * len(completion_ids),
        "completion_logprobs": [0.0] * len(completion_ids),
        "routed_experts": None,
    }
    if (
        full_rendered.multi_modal_data is not None
        and not full_rendered.multi_modal_data.is_empty()
    ):
        out["multi_modal_data"] = full_rendered.multi_modal_data
    return out
