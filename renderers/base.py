from __future__ import annotations

import enum
import logging
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TypedDict, runtime_checkable

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
    build per-token loss masks without re-rendering.  Tokens from structural
    scaffolding (generation prompt, im_start/im_end wrapping) carry index -1.

    ``multi_modal_data`` is populated by multimodal renderers (e.g.
    ``Qwen3VLRenderer``) when image / video content parts are present;
    text-only renderers leave it as ``None``.
    """

    token_ids: list[int] = field(default_factory=list)
    message_indices: list[int] = field(default_factory=list)
    multi_modal_data: "MultiModalData | None" = None


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
    "Qwen/Qwen3-235B-A22B": "qwen3",
    # Qwen3.5. All seven sizes share the same renderer. The 4B / 9B /
    # 35B-A3B / 122B-A10B / 397B-A17B chat template defaults
    # ``enable_thinking=true`` (open ``<think>\n`` at the gen prompt);
    # the smaller 0.8B / 2B variants flip the polarity (default
    # ``enable_thinking=false``, empty ``<think>\n\n</think>\n\n``).
    # ``Qwen35Renderer`` auto-detects polarity from the tokenizer's
    # chat_template at construction, so all seven sizes are
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


def load_tokenizer(model_name_or_path: str):
    """Load a tokenizer with the renderers-package security policy.

    Default: ``trust_remote_code=False`` — the safe choice for every
    model in ``MODEL_RENDERER_MAP`` *except* the Kimi-K2 family.

    Models listed in ``TRUSTED_REVISIONS`` load with
    ``trust_remote_code=True`` AND ``revision=<pinned sha>`` — required
    because their tokenizer config has an ``auto_map.AutoTokenizer``
    entry pointing at a repo-supplied Python class
    (``tokenization_kimi.TikTokenTokenizer``). Pinning the revision
    means transformers executes only the reviewed commit's code, not
    whatever ``HEAD`` points at when the call fires.

    Unknown / fine-tuned model paths fall through to
    ``trust_remote_code=False``. Callers who legitimately need to load
    a custom-code tokenizer outside this allow-list should call
    ``AutoTokenizer.from_pretrained`` themselves and pass the result to
    ``create_renderer`` (which doesn't load tokenizers — only
    ``create_renderer_pool`` does).
    """
    from transformers import AutoTokenizer

    revision = TRUSTED_REVISIONS.get(model_name_or_path)
    if revision is not None:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            revision=revision,
        )
    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)


def _populate_registry():
    if RENDERER_REGISTRY:
        return
    from renderers.default import DefaultRenderer
    from renderers.deepseek_v3 import DeepSeekV3Renderer
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
    renderer: str = "auto",
    size: int = 16,
    *,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    preserve_all_thinking: bool = False,
    preserve_thinking_between_tool_calls: bool = False,
) -> RendererPool:
    """Create a RendererPool with *size* independent tokenizer copies.

    Each slot loads its own tokenizer so threads never share mutable state.
    HuggingFace fast tokenizers release the GIL during Rust encoding, so
    threads achieve real parallelism.

    ``tool_parser`` and ``reasoning_parser`` are forwarded to
    ``create_renderer`` when the pool falls back to ``DefaultRenderer``.

    ``preserve_all_thinking`` and ``preserve_thinking_between_tool_calls``
    are forwarded to each pooled renderer's constructor — every slot in
    the pool shares one configuration. To run with a different
    configuration, build a different pool.

    Tokenizers load via ``load_tokenizer`` — see its docstring for the
    ``trust_remote_code`` policy (default off; Moonshot Kimi-K2 family
    opts in with a pinned ``revision``).
    """

    def factory() -> Renderer:
        tokenizer = load_tokenizer(tokenizer_name_or_path)
        return create_renderer(
            tokenizer,
            renderer=renderer,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            preserve_all_thinking=preserve_all_thinking,
            preserve_thinking_between_tool_calls=preserve_thinking_between_tool_calls,
        )

    return RendererPool(factory, size=size)


def create_renderer(
    tokenizer,
    renderer: str = "auto",
    *,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    preserve_all_thinking: bool = False,
    preserve_thinking_between_tool_calls: bool = False,
) -> Renderer:
    """Create a Renderer by name, or auto-detect from the tokenizer's model name.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        renderer: Renderer name ('qwen3', 'qwen3-vl', 'qwen3.5', 'qwen3.6',
                  'glm-5', 'glm-5.1', 'glm-4.5', 'minimax-m2', 'deepseek-v3',
                  'kimi-k2', 'kimi-k2.5', 'laguna-xs.2', 'nemotron-3',
                  'gpt-oss', 'default') or 'auto' to detect from model name.
        tool_parser: Name of a tool parser registered in ``renderers.parsers``.
                  Only consumed by DefaultRenderer. Model-specific renderers
                  have their own parsing wired in.
        reasoning_parser: Name of a reasoning parser registered in
                  ``renderers.parsers``. Only consumed by DefaultRenderer.
        preserve_all_thinking: Forwarded to the renderer's constructor.
                  When ``True``, the instance restores ``reasoning_content``
                  the chat template would otherwise drop on historical
                  assistants — useful when a downstream pass (e.g.
                  compaction prompts the model with a fresh ``user`` turn
                  asking for a summary) would lose the trajectory's
                  reasoning. See ``Renderer.render`` and
                  ``should_preserve_past_thinking``.
        preserve_thinking_between_tool_calls: Forwarded to the renderer's
                  constructor. ``True`` keeps reasoning on in-flight
                  tool-cycle assistants when the template would drop them.
                  See ``Renderer.render`` for semantics.
    """
    _populate_registry()

    default_kwargs: dict = {}
    if tool_parser is not None:
        default_kwargs["tool_parser"] = tool_parser
    if reasoning_parser is not None:
        default_kwargs["reasoning_parser"] = reasoning_parser

    preserve_kwargs: dict = {
        "preserve_all_thinking": preserve_all_thinking,
        "preserve_thinking_between_tool_calls": preserve_thinking_between_tool_calls,
    }

    if renderer != "auto":
        cls = RENDERER_REGISTRY.get(renderer)
        if cls is None:
            raise ValueError(
                f"Unknown renderer {renderer!r}. Available: {', '.join(sorted(RENDERER_REGISTRY))}"
            )
        if renderer == "default":
            return cls(tokenizer, **default_kwargs, **preserve_kwargs)
        if default_kwargs:
            logger.info(
                "tool_parser / reasoning_parser are only consumed by "
                "DefaultRenderer; ignoring for renderer=%r which has "
                "built-in behavior.",
                renderer,
            )
        return cls(tokenizer, **preserve_kwargs)

    # Auto-detect from model name via exact match on the canonical HF id.
    # Fine-tunes and renamed checkpoints miss on purpose — their chat
    # template may differ from the original even when the architecture
    # matches, so silently mapping them would produce template-parity
    # bugs. Set ``renderer=<name>`` explicitly for those.
    model_name = getattr(tokenizer, "name_or_path", "")
    renderer_name = MODEL_RENDERER_MAP.get(model_name)
    if renderer_name is not None:
        return RENDERER_REGISTRY[renderer_name](tokenizer, **preserve_kwargs)

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
            f"{supported_vlms}), or pass ``renderer='<name>'`` explicitly "
            f"if you know what you're doing."
        )

    # Text-only fall back to default (apply_chat_template). For fine-tunes
    # with customized chat templates this is the *correct* choice, so we don't
    # warn. Note the pick at INFO and advertise the parser knobs.
    logger.info(
        "No model-specific renderer matched %r. Using DefaultRenderer "
        "(apply_chat_template). Pass tool_parser=<name> or "
        "reasoning_parser=<name> to enable structured output parsing.",
        model_name or "<unnamed tokenizer>",
    )
    return RENDERER_REGISTRY["default"](tokenizer, **default_kwargs, **preserve_kwargs)


# ---------------------------------------------------------------------------
# Standalone helpers that work with any Renderer implementation
# ---------------------------------------------------------------------------


def build_training_sample(
    renderer: Renderer,
    messages: list[Message],
    *,
    role_to_mask: Callable[[Message], bool],
    tools: list[ToolSpec] | None = None,
) -> tuple[list[int], list[bool]]:
    """Build (token_ids, loss_mask) for supervised training.

    Single render() call + message_indices → per-token mask.
    Replaces build_incremental_token_mask (O(N) renders → O(1)).
    """
    rendered = renderer.render(messages, tools=tools)
    loss_mask: list[bool] = []
    for msg_idx in rendered.message_indices:
        if msg_idx < 0:
            loss_mask.append(False)
        else:
            loss_mask.append(role_to_mask(messages[msg_idx]))
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
