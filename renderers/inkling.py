"""Inkling renderer — ``thinkingmachines/Inkling`` and ``Inkling-Small``.

Inkling's chat template is **token-delimited**: every structural piece is a
single atomic special token, so there is no BOS and text between markers BPEs
independently. The message stream is a flat sequence of role-tagged,
``<|end_message|>``-terminated blocks:

    <|message_{role}|> [ <|content_{kind}|> {body} ] <|end_message|>

with these pieces (all single tokens):

- Roles: ``<|message_user|>`` / ``<|message_model|>`` (assistant) /
  ``<|message_system|>`` / ``<|message_tool|>``.
- Content kinds: ``<|content_text|>``, ``<|content_thinking|>`` (assistant
  reasoning), ``<|content_image|>``, ``<|content_audio_input|>`` (… ``<|audio_end|>``),
  ``<|content_invoke_tool_json|>`` (tool call), ``<|content_xml|>`` (tools decl).
- ``<|content_model_end_sampling|>`` closes an assistant turn (also the eos).

Ordering: an optional tools declaration comes first, then a one-off
``Thinking effort level: {N}`` system message emitted immediately before the
first non-system message (or at the very end if the whole conversation is
system-only). See :class:`renderers.configs.InklingRendererConfig` for the
effort knob.

Multimodal parity matches the native ``InklingProcessor`` (transformers
≥ 5.14): the template emits a single placeholder per media item, and the
processor **expands** it — image → ``num_patches`` copies of ``<|unused_200054|>``,
audio → one ``<|unused_200053|>`` per mel frame. This renderer expands the same
way (counts taken straight from the processor) and ships ``pixel_values`` /
``audio_input_ids`` in :class:`~renderers.base.MultiModalData`. The processor is
a native transformers class, so it loads with **no** ``trust_remote_code``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from renderers.base import (
    Content,
    Message,
    MultiModalData,
    ParsedResponse,
    PlaceholderRange,
    RenderedTokens,
    ToolSpec,
    Tokenizer,
    _content_mask_or_empty,
    _require_transformers,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import INKLING_EFFORT_MAP, InklingRendererConfig
from renderers.parsing import parse_inkling
from renderers.qwen3_vl import _image_hash, _load_pil_image

# Content-part ``type`` values the template maps to each modality. Untyped
# parts (``type`` absent) are treated as text — matching the Jinja template's
# ``part.type is not defined or part.type in ("text", "input_text")`` gate.
_TEXT_TYPES = frozenset({"text", "input_text"})
_IMAGE_TYPES = frozenset({"image", "input_image", "image_url"})
_AUDIO_TYPES = frozenset({"audio", "input_audio", "audio_url"})
_VIDEO_TYPES = frozenset({"video", "input_video", "video_url"})


def _format_effort_number(num: float) -> str:
    """Render the effort number exactly as the chat template does.

    ``0.0`` prints as ``"0"``; every other value uses Python ``str`` of the
    float (``0.9`` → ``"0.9"``), which matches Jinja's float stringification
    for the values in :data:`INKLING_EFFORT_MAP` and arbitrary user floats.
    """
    return "0" if num == 0.0 else str(num)


def _classify_part(part: Any) -> tuple[str, Any]:
    """Classify a content-list part → ``(kind, data)``.

    ``kind`` is one of ``"text"``/``"image"``/``"audio"``/``"video"``.
    Mirrors the template's part dispatch: a bare string or an untyped/text part
    is text; unknown part types raise just as the template does. Silently
    dropping a part would produce a token stream that does not represent the
    caller's conversation.
    """
    if isinstance(part, str):
        return "text", part
    if not isinstance(part, Mapping):
        raise ValueError(f"Unexpected Inkling content part: {part!r}")
    part = dict(part)
    ptype = part.get("type")
    if ptype is None or ptype in _TEXT_TYPES:
        text = part.get("text")
        return "text", text if isinstance(text, str) else ""
    if ptype in _IMAGE_TYPES:
        return "image", part
    if ptype in _AUDIO_TYPES:
        return "audio", part
    if ptype in _VIDEO_TYPES:
        return "video", part
    raise ValueError(f"Unsupported Inkling content part type: {ptype!r}")


def _load_inkling_image(part: Any):
    """Resolve Inkling's ``input_image`` alias before shared image loading."""
    if isinstance(part, Mapping) and part.get("type") == "input_image":
        normalized = dict(part)
        raw = normalized.get("input_image")
        if isinstance(raw, Mapping):
            for key in ("image", "image_url", "url", "path"):
                if key in raw:
                    normalized[key] = raw[key]
                    break
            else:
                normalized["image"] = raw
        else:
            normalized["image"] = raw
        part = normalized
    return _load_pil_image(part)


def _load_audio(part: Any) -> tuple[np.ndarray, int]:
    """Resolve an audio content part to ``(waveform, sampling_rate)``.

    Accepts a raw mono waveform (``np.ndarray`` / list of floats) or a
    HuggingFace-``datasets``-style ``{"array", "sampling_rate"}`` dict, under
    ``part["audio"]`` / ``["input_audio"]`` / ``["audio_url"]`` or the part
    itself. Decoding bytes / paths / URLs is out of scope (the processor's
    ``apply_chat_template`` does that upstream) — those raise so the caller
    passes a decoded waveform.
    """
    data: Any = part
    if isinstance(part, dict):
        for key in ("audio", "input_audio", "audio_url"):
            if key in part:
                data = part[key]
                break
    sampling_rate = 16000
    if isinstance(data, dict):
        sampling_rate = int(data.get("sampling_rate", sampling_rate))
        data = data.get("array")
    if isinstance(data, str) or data is None:
        raise NotImplementedError(
            "InklingRenderer needs a decoded audio waveform (np.ndarray or a "
            "{'array', 'sampling_rate'} dict); byte/path/URL decoding is not "
            "supported here."
        )
    return np.asarray(data, dtype=np.float32), sampling_rate


def _audio_hash(wav: np.ndarray, sampling_rate: int) -> str:
    # Sampling rate is part of the identity: identical samples at different
    # rates yield different mel frames / sidecar tensors, so they must not
    # share an ``mm_hashes`` entry (media IDs must be unique).
    h = hashlib.sha256(str(sampling_rate).encode())
    h.update(np.ascontiguousarray(wav).tobytes())
    return h.hexdigest()


class InklingRenderer:
    """Deterministic renderer for both Thinking Machines Inkling checkpoints.

    Text / tools / reasoning render to byte-parity with
    ``tokenizer.apply_chat_template``; image / audio render to byte-parity with
    the native ``InklingProcessor`` (placeholder expansion included). Satisfies
    the :class:`~renderers.base.MultimodalRenderer` protocol.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: InklingRendererConfig | None = None,
        *,
        processor: Any = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self.config = config or InklingRendererConfig()
        # Inkling always renders historical reasoning (the template has no
        # history-dropping knob), so the effective bridge policy is "all".
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )

        eff = self.config.reasoning_effort
        self._effort_num: float = (
            INKLING_EFFORT_MAP[eff.strip()] if isinstance(eff, str) else float(eff)
        )

        # Role markers.
        self._message_user = self._token_id("<|message_user|>")
        self._message_model = self._token_id("<|message_model|>")
        self._message_system = self._token_id("<|message_system|>")
        self._message_tool = self._token_id("<|message_tool|>")
        # Content-kind markers.
        self._content_text = self._token_id("<|content_text|>")
        self._content_thinking = self._token_id("<|content_thinking|>")
        self._content_image = self._token_id("<|content_image|>")
        self._content_audio_input = self._token_id("<|content_audio_input|>")
        self._content_xml = self._token_id("<|content_xml|>")
        self._content_invoke_tool_json = self._token_id("<|content_invoke_tool_json|>")
        self._content_invoke_tool_text = self._token_id("<|content_invoke_tool_text|>")
        self._content_model_end_sampling = self._token_id(
            "<|content_model_end_sampling|>"
        )
        self._end_message = self._token_id("<|end_message|>")
        self._audio_end = self._token_id("<|audio_end|>")
        # Media soft-token placeholders (expanded per patch / mel frame).
        self._image_pad = self._token_id("<|unused_200054|>")
        self._audio_pad = self._token_id("<|unused_200053|>")
        # eos: the model closes its turn with <|content_model_end_sampling|>.
        self._endoftext = self._try_token_id("<|endoftext|>")

        # Per-instance FIFO media caches (keyed by content hash).
        self._image_cache: dict[str, tuple[Any, int]] = {}
        self._audio_cache: dict[str, dict[str, Any]] = {}

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        """Token-id → modality marker. Only the image placeholder carries a
        vision marker (1); audio has no entry in the orchestrator's
        image/video type map (the model consumes ``audio_input_ids`` directly)."""
        return {self._image_pad: 1}

    # ------------------------------------------------------------------
    # Token / processor helpers
    # ------------------------------------------------------------------

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _try_token_id(self, token: str) -> int | None:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        if not isinstance(tid, int) or tid == self._tokenizer.unk_token_id:
            return None
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        name = getattr(self._tokenizer, "name_or_path", None)
        if not name:
            raise RuntimeError(
                "InklingRenderer needs a processor to render image/audio "
                "content. Pass processor=AutoProcessor.from_pretrained(name), "
                "or load the tokenizer with a known name_or_path so the "
                "processor can be auto-loaded."
            )
        # InklingProcessor is native in transformers >=5.14, so no
        # trust_remote_code is required. Keep text-only rendering installable
        # with older downstream pins and fail only when multimodal processing
        # is actually requested.
        transformers = _require_transformers("Auto-loading an Inkling processor")
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(name)
        except (ImportError, KeyError, ValueError) as exc:
            raise RuntimeError(
                "Inkling image/audio rendering requires Transformers >=5.14 "
                "with native InklingProcessor support. Upgrade Transformers "
                "and retry."
            ) from exc
        return self._processor

    def _process_image(self, pil, image_hash: str) -> tuple[Any, int]:
        """Return ``(processor_out, num_patches)`` for one image (cached).

        ``processor_out`` carries ``pixel_values`` (shape
        ``(num_patches, temporal, 40, 40, 3)``); ``num_patches`` is the count
        of ``<|unused_200054|>`` placeholders the processor expands this image
        into (``ceil(H/40) * (W//40 + 1)``)."""
        cached = self._image_cache.get(image_hash)
        if cached is not None:
            return cached
        out = self._get_processor().image_processor.preprocess(
            [pil], return_tensors="pt"
        )
        num_patches = int(out["num_patches"][0])
        if len(self._image_cache) >= self.config.image_cache_max:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[image_hash] = (out, num_patches)
        return out, num_patches

    def _process_audio(self, wav: np.ndarray, sampling_rate: int) -> dict[str, Any]:
        """Return the processor's ``{audio_input_ids, audio_input_ids_mask}``
        for one clip (cached). The number of ``<|unused_200053|>`` placeholders
        is ``audio_input_ids_mask[0].sum()`` (one per valid mel frame)."""
        key = _audio_hash(wav, sampling_rate)
        cached = self._audio_cache.get(key)
        if cached is not None:
            return cached
        processed = self._get_processor()(
            audio=[wav], sampling_rate=sampling_rate, return_tensors="pt"
        )
        if len(self._audio_cache) >= self.config.audio_cache_max:
            self._audio_cache.pop(next(iter(self._audio_cache)))
        self._audio_cache[key] = processed
        return processed

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []
        mm_hashes: dict[str, list[str]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        mm_items: dict[str, list[dict[str, Any]]] = {}

        def emit_special(
            token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(
            text: str, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_image(
            part: Any,
            msg_idx: int,
            role_open,
            *,
            is_sampled: bool,
            marker_is_content: bool,
        ) -> None:
            pil = _load_inkling_image(part)
            h = _image_hash(pil)
            out, num_patches = self._process_image(pil, h)
            role_open()
            emit_special(
                self._content_image,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            offset = len(tokens)
            for _ in range(num_patches):
                emit_special(
                    self._image_pad, msg_idx, is_sampled=is_sampled, is_content=True
                )
            emit_special(
                self._end_message,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            mm_hashes.setdefault("image", []).append(h)
            mm_placeholders.setdefault("image", []).append(
                PlaceholderRange(offset=offset, length=num_patches)
            )
            mm_items.setdefault("image", []).append(
                {"pixel_values": out["pixel_values"]}
            )

        def emit_audio(
            part: Any,
            msg_idx: int,
            role_open,
            *,
            is_sampled: bool,
            marker_is_content: bool,
        ) -> None:
            wav, sr = _load_audio(part)
            processed = self._process_audio(wav, sr)
            n_frames = int(processed["audio_input_ids_mask"][0].sum())
            role_open()
            emit_special(
                self._content_audio_input,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            offset = len(tokens)
            for _ in range(n_frames):
                emit_special(
                    self._audio_pad, msg_idx, is_sampled=is_sampled, is_content=True
                )
            emit_special(
                self._audio_end,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            emit_special(
                self._end_message,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            mm_hashes.setdefault("audio", []).append(_audio_hash(wav, sr))
            mm_placeholders.setdefault("audio", []).append(
                PlaceholderRange(offset=offset, length=n_frames)
            )
            mm_items.setdefault("audio", []).append(
                {
                    "audio_input_ids": processed["audio_input_ids"],
                    "audio_input_ids_mask": processed["audio_input_ids_mask"],
                }
            )

        tool_names = extract_message_tool_names(messages)

        # ── Tools declaration (first, before the effort line) ──
        if tools:
            self._emit_tools(tools, emit_special, emit_text)

        # ── Message loop; effort emitted once before the first non-system ──
        effort_emitted = False
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if not effort_emitted and role != "system":
                self._emit_effort(emit_special, emit_text)
                effort_emitted = True

            if role == "assistant":
                self._render_assistant(
                    msg, i, emit_special, emit_text, emit_image, emit_audio
                )
            elif role == "tool":
                self._render_tool(msg, i, tool_names[i], emit_special, emit_text)
            elif role in ("system", "user"):
                role_id = (
                    self._message_system if role == "system" else self._message_user
                )

                def role_open(_role_id=role_id, _i=i):
                    emit_special(_role_id, _i, is_sampled=False, is_content=False)

                self._emit_content(
                    msg.get("content"),
                    i,
                    role_open=role_open,
                    is_sampled=False,
                    marker_is_content=False,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_image=emit_image,
                    emit_audio=emit_audio,
                )
            else:
                raise ValueError(f"Unknown message role: {role!r}")

        if not effort_emitted:
            self._emit_effort(emit_special, emit_text)

        # ── Generation prompt ──
        if add_generation_prompt:
            emit_special(self._message_model, -1, is_sampled=False, is_content=False)

        mm_data: MultiModalData | None = None
        if mm_hashes or mm_placeholders or mm_items:
            mm_data = MultiModalData(
                mm_hashes=mm_hashes,
                mm_placeholders=mm_placeholders,
                mm_items=mm_items,
            )

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=_content_mask_or_empty(self._tokenizer, content_mask),
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=tool_names,
            multi_modal_data=mm_data,
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — args are native JSON, no schema coercion
    ) -> ParsedResponse:
        return parse_inkling(
            self._tokenizer,
            token_ids,
            stop_ids=set(self.get_stop_token_ids()),
            message_model_id=self._message_model,
            content_text_id=self._content_text,
            content_thinking_id=self._content_thinking,
            invoke_json_id=self._content_invoke_tool_json,
            invoke_text_id=self._content_invoke_tool_text,
            end_message_id=self._end_message,
        )

    def get_stop_token_ids(self) -> list[int]:
        stop = [self._content_model_end_sampling]
        if self._endoftext is not None:
            stop.append(self._endoftext)
        return stop

    # ------------------------------------------------------------------
    # Per-message emitters (shared by render + bridge)
    # ------------------------------------------------------------------

    def _emit_effort(self, emit_special, emit_text) -> None:
        emit_special(self._message_system, -1, is_sampled=False, is_content=False)
        emit_special(self._content_text, -1, is_sampled=False, is_content=False)
        emit_text(
            "Thinking effort level: " + _format_effort_number(self._effort_num),
            -1,
            is_sampled=False,
            is_content=False,
        )
        emit_special(self._end_message, -1, is_sampled=False, is_content=False)

    def _emit_tools(self, tools, emit_special, emit_text) -> None:
        specs: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else tool
            if not isinstance(fn, dict):
                fn = {}
            specs.append(
                {
                    "description": fn.get("description") or "",
                    "name": fn.get("name"),
                    "parameters": fn.get("parameters") or {},
                    "type": (tool.get("type") if isinstance(tool, dict) else None)
                    or "function",
                }
            )
        tools_json = json.dumps(
            specs, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        emit_special(self._message_system, -1, is_sampled=False, is_content=False)
        emit_text("tool_declare", -1, is_sampled=False, is_content=False)
        emit_special(self._content_xml, -1, is_sampled=False, is_content=False)
        emit_text(tools_json, -1, is_sampled=False, is_content=False)
        emit_special(self._end_message, -1, is_sampled=False, is_content=False)

    def _emit_content(
        self,
        content: Content | None,
        msg_idx: int,
        *,
        role_open,
        is_sampled: bool,
        marker_is_content: bool,
        emit_special,
        emit_text,
        emit_image,
        emit_audio,
    ) -> None:
        """Emit a message's content as one or more role-tagged blocks.

        String content is one ``<|content_text|>`` block (emitted even when
        empty). List content is one block per part, each re-opening the role
        via ``role_open`` — matching the template's per-part role re-emission.
        ``marker_is_content`` sets the ``is_content`` bit on the structural
        markers (True for assistant, where every sampled token counts as body;
        False for user/system, where markers are scaffold)."""
        if isinstance(content, str):
            role_open()
            emit_special(
                self._content_text,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            if content:
                emit_text(content, msg_idx, is_sampled=is_sampled, is_content=True)
            emit_special(
                self._end_message,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            return
        if not content:
            return
        for part in content:
            kind, data = _classify_part(part)
            if kind == "text":
                role_open()
                emit_special(
                    self._content_text,
                    msg_idx,
                    is_sampled=is_sampled,
                    is_content=marker_is_content,
                )
                if data:
                    emit_text(data, msg_idx, is_sampled=is_sampled, is_content=True)
                emit_special(
                    self._end_message,
                    msg_idx,
                    is_sampled=is_sampled,
                    is_content=marker_is_content,
                )
            elif kind == "image":
                emit_image(
                    data,
                    msg_idx,
                    role_open,
                    is_sampled=is_sampled,
                    marker_is_content=marker_is_content,
                )
            elif kind == "audio":
                emit_audio(
                    data,
                    msg_idx,
                    role_open,
                    is_sampled=is_sampled,
                    marker_is_content=marker_is_content,
                )
            elif kind == "video":
                raise NotImplementedError(
                    "Video parts are not supported by InklingRenderer."
                )

    def _render_tool(self, msg, msg_idx, inline_name, emit_special, emit_text) -> None:
        """Tool response: ``<|message_tool|>{name?}<|content_text|>{content}<|end_message|>``.

        The name (resolved from ``msg['name']`` or the matching prior
        assistant tool-call id) is structural routing info, so it is scaffold.
        Only string content is rendered — the template drops list content in
        tool messages."""
        emit_special(self._message_tool, msg_idx, is_sampled=False, is_content=False)
        if inline_name:
            emit_text(inline_name, msg_idx, is_sampled=False, is_content=False)
        emit_special(self._content_text, msg_idx, is_sampled=False, is_content=False)
        content = msg.get("content")
        if isinstance(content, str) and content:
            emit_text(content, msg_idx, is_sampled=False, is_content=True)
        emit_special(self._end_message, msg_idx, is_sampled=False, is_content=False)

    def _render_assistant(
        self, msg, msg_idx, emit_special, emit_text, emit_image, emit_audio
    ) -> None:
        """Assistant turn: optional reasoning block, content block(s), tool-call
        blocks, then the ``<|content_model_end_sampling|>`` close.

        The very first ``<|message_model|>`` is the generation-prompt-equivalent
        (scaffold, never sampled); every token after it through the close is the
        model's sampled emission, so ``is_content == sampled_mask`` holds across
        the whole turn."""
        reasoning = msg.get("reasoning_content")
        reasoning = reasoning if isinstance(reasoning, str) and reasoning else ""
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []

        first = True

        def model_tag():
            nonlocal first
            if first:
                emit_special(
                    self._message_model, msg_idx, is_sampled=False, is_content=False
                )
                first = False
            else:
                emit_special(
                    self._message_model, msg_idx, is_sampled=True, is_content=True
                )

        if reasoning:
            model_tag()
            emit_special(
                self._content_thinking, msg_idx, is_sampled=True, is_content=True
            )
            emit_text(reasoning, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._end_message, msg_idx, is_sampled=True, is_content=True)

        self._emit_content(
            content,
            msg_idx,
            role_open=model_tag,
            is_sampled=True,
            marker_is_content=True,
            emit_special=emit_special,
            emit_text=emit_text,
            emit_image=emit_image,
            emit_audio=emit_audio,
        )

        for tc in tool_calls:
            name, args = self._tool_call_name_args(tc)
            model_tag()
            if name:
                emit_text(name, msg_idx, is_sampled=True, is_content=True)
            emit_special(
                self._content_invoke_tool_json,
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
            emit_text(
                self._invoke_json(name, args),
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
            emit_special(self._end_message, msg_idx, is_sampled=True, is_content=True)

        emit_special(
            self._content_model_end_sampling, msg_idx, is_sampled=True, is_content=True
        )

    @staticmethod
    def _tool_call_name_args(tc: Any) -> tuple[str, Any]:
        func = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(func, dict):
            func = tc if isinstance(tc, dict) else {}
        name = func.get("name") or ""
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError) as exc:
                raise TypeError(
                    "Inkling tool_calls[].function.arguments must be a JSON object."
                ) from exc
        if not isinstance(args, dict):
            raise TypeError(
                "Inkling tool_calls[].function.arguments must be a JSON object."
            )
        return name, args

    @staticmethod
    def _invoke_json(name: str, args: Any) -> str:
        """``{"name":<name>,"args":<args>}`` with the template's compact,
        key-sorted encoding."""
        return (
            '{"name":'
            + json.dumps(
                name, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            + ',"args":'
            + json.dumps(
                args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            + "}"
        )

    # ------------------------------------------------------------------
    # Bridge
    # ------------------------------------------------------------------

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        previous_multi_modal_data: MultiModalData | None = None,
    ) -> "RenderedTokens | None":
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention, new_messages
        ):
            return None

        # A tool message whose name would be recovered from a prior-turn
        # assistant tool-call id can't be resolved from ``new_messages`` alone
        # (assistants are excluded), so a bridge would diverge from a full
        # re-render — refuse and let the caller re-render.
        for msg in new_messages:
            if (
                msg.get("role") == "tool"
                and not msg.get("name")
                and msg.get("tool_call_id")
            ):
                return None

        # Prior assistant turn closes with <|content_model_end_sampling|>;
        # synthesise it on a truncated completion.
        close_ids: set[int] = {self._content_model_end_sampling}
        if self._endoftext is not None:
            close_ids.add(self._endoftext)
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            close_ids,
            synthesize_close=self._content_model_end_sampling,
        )
        if previous_ids is None:
            return None

        tokens: list[int] = list(previous_ids)
        indices: list[int] = [-1] * len(previous_ids)
        sampled: list[bool] = [False] * len(previous_ids)
        content_mask: list[bool] = [False] * len(previous_ids)
        new_hashes: dict[str, list[str]] = {}
        new_placeholders: dict[str, list[PlaceholderRange]] = {}
        new_items: dict[str, list[dict[str, Any]]] = {}

        def emit_special(
            token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(
            text: str, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_image(
            part, msg_idx, role_open, *, is_sampled, marker_is_content
        ) -> None:
            pil = _load_inkling_image(part)
            h = _image_hash(pil)
            out, num_patches = self._process_image(pil, h)
            role_open()
            emit_special(
                self._content_image,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            offset = len(tokens)
            for _ in range(num_patches):
                emit_special(
                    self._image_pad, msg_idx, is_sampled=is_sampled, is_content=True
                )
            emit_special(
                self._end_message,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            new_hashes.setdefault("image", []).append(h)
            new_placeholders.setdefault("image", []).append(
                PlaceholderRange(offset=offset, length=num_patches)
            )
            new_items.setdefault("image", []).append(
                {"pixel_values": out["pixel_values"]}
            )

        def emit_audio(
            part, msg_idx, role_open, *, is_sampled, marker_is_content
        ) -> None:
            wav, sr = _load_audio(part)
            processed = self._process_audio(wav, sr)
            n_frames = int(processed["audio_input_ids_mask"][0].sum())
            role_open()
            emit_special(
                self._content_audio_input,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            offset = len(tokens)
            for _ in range(n_frames):
                emit_special(
                    self._audio_pad, msg_idx, is_sampled=is_sampled, is_content=True
                )
            emit_special(
                self._audio_end,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            emit_special(
                self._end_message,
                msg_idx,
                is_sampled=is_sampled,
                is_content=marker_is_content,
            )
            new_hashes.setdefault("audio", []).append(_audio_hash(wav, sr))
            new_placeholders.setdefault("audio", []).append(
                PlaceholderRange(offset=offset, length=n_frames)
            )
            new_items.setdefault("audio", []).append(
                {
                    "audio_input_ids": processed["audio_input_ids"],
                    "audio_input_ids_mask": processed["audio_input_ids_mask"],
                }
            )

        tool_names = extract_message_tool_names(new_messages)
        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            if role == "tool":
                self._render_tool(msg, i, tool_names[i], emit_special, emit_text)
            elif role in ("system", "user"):
                role_id = (
                    self._message_system if role == "system" else self._message_user
                )

                def role_open(_role_id=role_id, _i=i):
                    emit_special(_role_id, _i, is_sampled=False, is_content=False)

                self._emit_content(
                    msg.get("content"),
                    i,
                    role_open=role_open,
                    is_sampled=False,
                    marker_is_content=False,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_image=emit_image,
                    emit_audio=emit_audio,
                )
            else:
                return None

        emit_special(self._message_model, -1, is_sampled=False, is_content=False)

        # Merge carried-forward mm_data (prior turns) with this turn's items.
        # Copy the per-modality lists (not just the outer dict) so appending
        # this turn's items never mutates the caller's previous_multi_modal_data.
        merged_hashes = (
            {k: list(v) for k, v in previous_multi_modal_data.mm_hashes.items()}
            if previous_multi_modal_data
            else {}
        )
        merged_placeholders = (
            {k: list(v) for k, v in previous_multi_modal_data.mm_placeholders.items()}
            if previous_multi_modal_data
            else {}
        )
        merged_items = (
            {k: list(v) for k, v in previous_multi_modal_data.mm_items.items()}
            if previous_multi_modal_data
            else {}
        )
        for modality, vals in new_hashes.items():
            merged_hashes.setdefault(modality, []).extend(vals)
        for modality, vals in new_placeholders.items():
            merged_placeholders.setdefault(modality, []).extend(vals)
        for modality, vals in new_items.items():
            merged_items.setdefault(modality, []).extend(vals)

        mm_data: MultiModalData | None = None
        if merged_hashes or merged_placeholders or merged_items:
            mm_data = MultiModalData(
                mm_hashes=merged_hashes,
                mm_placeholders=merged_placeholders,
                mm_items=merged_items,
            )

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=_content_mask_or_empty(self._tokenizer, content_mask),
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=tool_names,
            multi_modal_data=mm_data,
        )
