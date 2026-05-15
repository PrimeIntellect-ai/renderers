"""Qwen3-VL renderer with multimodal (image + video) support.

Produces a token stream that matches ``Qwen3VLProcessor.apply_chat_template``
byte-for-byte for text-only inputs and emits the same
``<|vision_start|>`` + N×``<|image_pad|>`` + ``<|vision_end|>`` expansion
for image inputs as the HF processor (``N = image_grid_thw.prod() //
merge_size**2``).

Image data is shipped to the inference engine via
``RenderedTokens.multi_modal_data``: ``mm_placeholders`` records the
``(offset, length)`` span of each image's placeholder tokens in the
prompt, ``mm_items`` carries the per-image processor output
(``pixel_values``, ``image_grid_thw``), and ``mm_hashes`` carries a
stable identifier for cache lookup. The wire-format conversion to
vLLM's ``/inference/v1/generate`` ``features`` field lives in
``renderers.client``.

BPE boundary discipline: text runs that the chat template emits
contiguously (e.g. ``"user\\n" + content_text``) must be encoded as a
single tokenizer call — otherwise BPE merges differ from the template's
output. The internal ``_Emitter`` buffers text and flushes on special
tokens (``<|im_start|>``, ``<|im_end|>``, ``<tool_response>``,
``<|vision_start|>``…), which act as atomic boundaries the template
also can't merge across.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from typing import Any
from urllib.parse import urlparse

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    MultiModalData,
    ParsedResponse,
    PlaceholderRange,
    RenderedTokens,
    ToolSpec,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.parsing import parse_qwen3

_TOOLS_HEADER = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)

_TOOLS_FOOTER = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def _is_image_part(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    t = item.get("type")
    if t in ("image", "image_url"):
        return True
    # Permissive fallback: chat templates check ``'image' in content`` to
    # accept loosely-shaped image parts, so mirror that.
    return "image" in item or "image_url" in item


def _is_video_part(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    t = item.get("type")
    if t in ("video", "video_url"):
        return True
    return "video" in item or "video_url" in item


def _load_pil_image(item: dict[str, Any]):
    """Resolve an ImagePart to a PIL Image.

    Accepts pre-loaded PIL Images, raw bytes, filesystem paths,
    ``file://``/``http(s)://`` URLs, and ``data:image/...;base64,...`` URIs.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for multimodal rendering. Install with "
            "`pip install Pillow` (or `pip install renderers[multimodal]`)."
        ) from exc

    raw: Any
    if "image" in item:
        raw = item["image"]
    elif "image_url" in item:
        # OpenAI canonical shape is ``image_url: {"url": "..."}`` — but
        # some VLM processors (Kimi K2.5 / K2.6) hand a raw PIL / str
        # directly under ``image_url``. Accept both.
        iu = item.get("image_url")
        raw = iu.get("url") if isinstance(iu, dict) else iu
    else:
        raw = item.get("url") or item.get("path")

    if isinstance(raw, Image.Image):
        return raw.convert("RGB") if raw.mode != "RGB" else raw

    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if not isinstance(raw, str):
        raise TypeError(
            f"Unsupported image source {type(raw).__name__!r}; expected PIL "
            "Image, bytes, path, http(s):// URL, file:// URL, or data: URI."
        )

    if raw.startswith("data:"):
        # data:image/png;base64,XXXX
        _, _, payload = raw.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        import urllib.request

        with urllib.request.urlopen(raw) as resp:  # noqa: S310 — user-supplied URL
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    if parsed.scheme == "file" or parsed.scheme == "":
        path = parsed.path if parsed.scheme == "file" else raw
        return Image.open(path).convert("RGB")

    raise ValueError(f"Unsupported image URL scheme: {parsed.scheme!r} in {raw!r}")


def _image_hash(pil_image) -> str:
    """Stable per-image identifier for cache lookup.

    Uses the resolved RGB bytes so two ``ImagePart``\\s pointing at the
    same logical image (path, in-memory, data URI) hash identically.
    """
    h = hashlib.sha256()
    h.update(pil_image.tobytes())
    h.update(f"{pil_image.size}".encode())
    return h.hexdigest()[:32]


class _Emitter:
    """Token-stream builder with BPE-safe text buffering.

    Special tokens are atomic boundaries — the BPE encoder can't merge
    across them, and neither can a chat template's Jinja output. So we
    buffer plain text and flush to ``tokenizer.encode`` only when we
    hit a special token (or end of message). Two text fragments emitted
    back-to-back end up in the same flush, exactly matching how the
    chat template concatenates string outputs before the final encode.

    Per-token ``sampled_mask`` is tracked alongside ``token_ids`` /
    ``message_indices``: every ``text(..., is_sampled=...)`` and
    ``special(..., is_sampled=...)`` call stamps its emitted tokens with
    the supplied flag. A text fragment with a different ``is_sampled``
    value than the current buffer triggers a flush first — split points
    are always at the ``is_sampled`` boundary, which the caller is
    expected to place at a ``\\n`` boundary so BPE merges don't drift.
    """

    def __init__(self, encode_fn, msg_idx: int = -1):
        self._encode = encode_fn
        self.token_ids: list[int] = []
        self.message_indices: list[int] = []
        self.sampled: list[bool] = []
        self._buf: str = ""
        self._buf_idx: int = msg_idx
        self._buf_sampled: bool = False
        self.msg_idx = msg_idx

    def set_msg_idx(self, msg_idx: int) -> None:
        # When the active message changes, flush so the new message's
        # text doesn't get glued to the previous one's BPE context.
        # In practice messages are always separated by an <|im_end|>
        # special token, which already flushes — but be defensive.
        if self._buf:
            self._flush()
        self.msg_idx = msg_idx
        self._buf_idx = msg_idx

    def text(self, text: str, *, is_sampled: bool) -> None:
        if not text:
            return
        # Adjacent text under different msg_idx or is_sampled is rare in
        # this template — but flush at those boundaries so attribution
        # and the sampled signal stay accurate.
        if self._buf and (
            self._buf_idx != self.msg_idx or self._buf_sampled != is_sampled
        ):
            self._flush()
        if not self._buf:
            self._buf_idx = self.msg_idx
            self._buf_sampled = is_sampled
        self._buf += text

    def special(self, token_id: int, *, is_sampled: bool) -> None:
        if self._buf:
            self._flush()
        self.token_ids.append(token_id)
        self.message_indices.append(self.msg_idx)
        self.sampled.append(is_sampled)

    def cursor(self) -> int:
        """Current token offset after flushing — used to anchor placeholder ranges."""
        if self._buf:
            self._flush()
        return len(self.token_ids)

    def finalize(self) -> None:
        if self._buf:
            self._flush()

    def _flush(self) -> None:
        ids = self._encode(self._buf)
        self.token_ids.extend(ids)
        self.message_indices.extend([self._buf_idx] * len(ids))
        self.sampled.extend([self._buf_sampled] * len(ids))
        self._buf = ""


class Qwen3VLRenderer:
    """Deterministic message-to-token renderer for Qwen3-VL models.

    Constructor args:
        tokenizer: HF tokenizer for the model.
        processor: Optional ``Qwen3VLProcessor``. Required when rendering
            messages that contain image / video parts. If not supplied,
            the renderer lazy-loads it via ``AutoProcessor.from_pretrained``
            keyed off ``tokenizer.name_or_path`` the first time a
            multimodal part is seen.
        preserve_all_thinking / preserve_thinking_between_tool_calls:
            No-ops on Qwen3-VL — the chat template already drops past
            ``<think>`` blocks unconditionally. Stored for Protocol parity.
        image_cache_max: Max entries in the per-instance image-processor
            cache (FIFO eviction). Default 256 covers typical RL pools
            (``rollouts_per_example`` × in-flight examples). Bump for runs
            with large image sets where the working set exceeds the cap.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        processor: Any = None,
        preserve_all_thinking: bool = False,
        preserve_thinking_between_tool_calls: bool = False,
        image_cache_max: int = 256,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self._preserve_all_thinking = preserve_all_thinking
        self._preserve_thinking_between_tool_calls = (
            preserve_thinking_between_tool_calls
        )

        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")
        self._vision_start = self._token_id("<|vision_start|>")
        self._vision_end = self._token_id("<|vision_end|>")
        self._image_pad = self._token_id("<|image_pad|>")
        self._video_pad = self._token_id("<|video_pad|>")

        # Per-instance image-processor cache. The HF image processor is the
        # most expensive step on the renderer hot path (~tens of ms per
        # image for typical grid_thw). The same image gets re-seen across
        # ``rollouts_per_example`` rollouts of one example and (for
        # multi-turn) across turn boundaries when the bridge re-renders
        # rather than extends. Cache keyed by content hash — values are
        # tuples of ``(processor_out, num_image_tokens)`` — bounded to
        # avoid unbounded growth on long-lived pools.
        self._image_cache: dict[str, tuple[Any, int]] = {}
        self._image_cache_max = image_cache_max

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        """Token-id → modality marker used to build ``mm_token_type_ids``.

        Qwen3-VL uses a single placeholder token per image/video patch
        (``<|image_pad|>``, ``<|video_pad|>``); the trainer's forward
        expects a per-token int tensor where 1 = image patch, 2 = video
        patch, 0 = anything else. The orchestrator walks the final token
        sequence and applies this map (constant per renderer instance,
        cached at construction) — no separate processor load needed.
        """
        return {self._image_pad: 1, self._video_pad: 2}

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        from transformers import AutoProcessor

        name = getattr(self._tokenizer, "name_or_path", None)
        if not name:
            raise RuntimeError(
                "Qwen3VLRenderer needs a processor to render image / video parts. "
                "Pass `processor=AutoProcessor.from_pretrained(...)` to the "
                "constructor, or load the tokenizer with a known name_or_path "
                "so the processor can be auto-loaded."
            )
        self._processor = AutoProcessor.from_pretrained(name)
        return self._processor

    @staticmethod
    def _render_text_content(content: Any) -> str:
        """Flatten a content list to a single text string, dropping media parts.

        Used for paths where we only care about text (e.g. system role,
        assistant non-tool-call content).
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if _is_image_part(item) or _is_video_part(item):
                        continue
                    if "text" in item:
                        parts.append(item["text"])
                    elif item.get("type") == "thinking" and "thinking" in item:
                        parts.append(item["thinking"])
                else:
                    raise ValueError(f"Unexpected content item: {item}")
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    def _process_image(self, part: dict[str, Any]):
        """Resolve, process, and characterize a single image part.

        Returns ``(pil, processor_out, num_image_tokens, image_hash)``.
        Hashes the loaded PIL first and consults ``self._image_cache``;
        on hit the HF image-processor call is skipped entirely.
        """
        pil = _load_pil_image(part)
        h = _image_hash(pil)
        cached = self._image_cache.get(h)
        if cached is not None:
            out, num_image_tokens = cached
            return pil, out, num_image_tokens, h
        proc = self._get_processor()
        out = proc.image_processor(images=[pil], return_tensors="np")
        grid_thw = out["image_grid_thw"][0]
        merge_size = proc.image_processor.merge_size
        num_image_tokens = int(grid_thw.prod()) // (merge_size * merge_size)
        if len(self._image_cache) >= self._image_cache_max:
            # FIFO eviction — Python dicts preserve insertion order, so
            # ``next(iter(...))`` is the oldest key.
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[h] = (out, num_image_tokens)
        return pil, out, num_image_tokens, h

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        em = _Emitter(self._encode)
        mm_hashes: dict[str, list[str]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        mm_items: dict[str, list[dict[str, Any]]] = {}

        def emit_image(part: dict[str, Any]) -> None:
            # Image placeholders are prompt-side scaffolding the user
            # message attaches — the model never samples ``<|vision_start|>``
            # / ``<|image_pad|>`` / ``<|vision_end|>``.
            _, out, n, h = self._process_image(part)
            em.special(self._vision_start, is_sampled=False)
            offset = em.cursor()
            for _ in range(n):
                em.special(self._image_pad, is_sampled=False)
            em.special(self._vision_end, is_sampled=False)
            mm_hashes.setdefault("image", []).append(h)
            mm_placeholders.setdefault("image", []).append(
                PlaceholderRange(offset=offset, length=n)
            )
            mm_items.setdefault("image", []).append(
                {
                    "pixel_values": out["pixel_values"],
                    "image_grid_thw": out["image_grid_thw"],
                }
            )

        def render_media_content(content: Any) -> None:
            """Emit a user/tool content list with media handled inline.

            User / tool content is conversation context the model never
            samples — every text fragment goes in as ``is_sampled=False``.
            """
            if isinstance(content, str):
                em.text(content, is_sampled=False)
                return
            if not isinstance(content, list):
                em.text(self._render_text_content(content), is_sampled=False)
                return
            for item in content:
                if isinstance(item, str):
                    em.text(item, is_sampled=False)
                elif isinstance(item, dict):
                    if _is_image_part(item):
                        emit_image(item)
                    elif _is_video_part(item):
                        raise NotImplementedError(
                            "Video parts are not yet supported by Qwen3VLRenderer."
                        )
                    elif "text" in item:
                        em.text(item["text"], is_sampled=False)

        # ── 1. System + tools ───────────────────────────────────────
        first_is_system = messages[0].get("role") == "system"

        if tools:
            sys_idx = 0 if first_is_system else -1
            em.set_msg_idx(sys_idx)
            em.special(self._im_start, is_sampled=False)
            buf = "system\n"
            if first_is_system:
                buf += self._render_text_content(messages[0].get("content")) + "\n\n"
            buf += _TOOLS_HEADER
            for tool in tools:
                buf += "\n" + json.dumps(tool, ensure_ascii=False)
            buf += _TOOLS_FOOTER
            em.text(buf, is_sampled=False)
            em.special(self._im_end, is_sampled=False)
            em.text("\n", is_sampled=False)
        elif first_is_system:
            em.set_msg_idx(0)
            em.special(self._im_start, is_sampled=False)
            em.text(
                "system\n" + self._render_text_content(messages[0].get("content")),
                is_sampled=False,
            )
            em.special(self._im_end, is_sampled=False)
            em.text("\n", is_sampled=False)

        # ── 2. Iterate messages ─────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]

            if role == "system":
                continue

            em.set_msg_idx(i)

            if role == "user":
                em.special(self._im_start, is_sampled=False)
                em.text("user\n", is_sampled=False)
                render_media_content(msg.get("content"))
                em.special(self._im_end, is_sampled=False)
                em.text("\n", is_sampled=False)

            elif role == "assistant":
                self._render_assistant(msg, em)

            elif role == "tool":
                self._render_tool(messages, i, em, render_media_content)

            else:
                raise ValueError(f"Unexpected message role: {role}")

        # ── 3. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            em.set_msg_idx(-1)
            em.special(self._im_start, is_sampled=False)
            em.text("assistant\n", is_sampled=False)

        em.finalize()

        mm_data: MultiModalData | None = None
        if mm_hashes or mm_placeholders or mm_items:
            mm_data = MultiModalData(
                mm_hashes=mm_hashes,
                mm_placeholders=mm_placeholders,
                mm_items=mm_items,
            )

        return RenderedTokens(
            token_ids=em.token_ids,
            message_indices=em.message_indices,
            sampled_mask=em.sampled,
            message_roles=[m.get("role") or "" for m in messages],
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
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — hermes wire format quotes strings, schema not needed
    ) -> ParsedResponse:
        return parse_qwen3(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end, self._endoftext]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        previous_multi_modal_data: MultiModalData | None = None,
    ) -> "RenderedTokens | None":
        """Extend the prior turn with ``new_messages``.

        Returns ``RenderedTokens`` so the caller recovers placeholder
        offsets and processed tensors for images that appear in
        ``new_messages``. ``previous_multi_modal_data`` carries forward
        images from earlier turns — their placeholder offsets are
        unchanged in the new prompt (they sit at lower positions than
        the synthesized close token), so we just concatenate.
        """
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._im_end, self._endoftext},
            synthesize_close=self._im_end,
        )
        if previous_ids is None:
            return None

        # Bridge populates ``message_indices`` (relative to ``new_messages``)
        # and ``sampled_mask`` (uniformly ``False`` — every token the
        # bridge emits is template scaffolding for the next prompt, not
        # something the model sampled). Downstream consumers can run
        # :meth:`RenderedTokens.tokens_per_message` on the bridge output
        # to get per-new-message token counts without re-rendering.
        em = _Emitter(self._encode)
        # Seed the emitter with the prior turn's tokens so cursor() reports
        # absolute offsets in the combined sequence. Per-token attribution
        # for the prior portion is unknown to the bridge (it only has
        # prev_prompt_ids + prev_completion_ids as raw lists), so seed
        # both side channels with the "no info" sentinel.
        em.token_ids = list(previous_ids)
        em.message_indices = [-1] * len(previous_ids)
        em.sampled = [False] * len(previous_ids)

        new_hashes: dict[str, list[str]] = {}
        new_placeholders: dict[str, list[PlaceholderRange]] = {}
        new_items: dict[str, list[dict[str, Any]]] = {}

        def emit_image(part: dict[str, Any]) -> None:
            _, out, n, h = self._process_image(part)
            em.special(self._vision_start, is_sampled=False)
            offset = em.cursor()
            for _ in range(n):
                em.special(self._image_pad, is_sampled=False)
            em.special(self._vision_end, is_sampled=False)
            new_hashes.setdefault("image", []).append(h)
            new_placeholders.setdefault("image", []).append(
                PlaceholderRange(offset=offset, length=n)
            )
            new_items.setdefault("image", []).append(
                {
                    "pixel_values": out["pixel_values"],
                    "image_grid_thw": out["image_grid_thw"],
                }
            )

        def render_media_content(content: Any) -> None:
            if isinstance(content, str):
                em.text(content, is_sampled=False)
                return
            if not isinstance(content, list):
                em.text(self._render_text_content(content), is_sampled=False)
                return
            for item in content:
                if isinstance(item, str):
                    em.text(item, is_sampled=False)
                elif isinstance(item, dict):
                    if _is_image_part(item):
                        emit_image(item)
                    elif _is_video_part(item):
                        raise NotImplementedError(
                            "Video parts are not yet supported by Qwen3VLRenderer."
                        )
                    elif "text" in item:
                        em.text(item["text"], is_sampled=False)

        em.set_msg_idx(-1)
        em.text("\n", is_sampled=False)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            em.set_msg_idx(i)
            if role == "user":
                em.special(self._im_start, is_sampled=False)
                em.text("user\n", is_sampled=False)
                render_media_content(msg.get("content"))
                em.special(self._im_end, is_sampled=False)
                em.text("\n", is_sampled=False)
            elif role == "system":
                em.special(self._im_start, is_sampled=False)
                em.text("system\n", is_sampled=False)
                render_media_content(msg.get("content"))
                em.special(self._im_end, is_sampled=False)
                em.text("\n", is_sampled=False)
            elif role == "tool":
                self._render_tool(new_messages, i, em, render_media_content)
            else:
                return None

        em.set_msg_idx(-1)
        em.special(self._im_start, is_sampled=False)
        em.text("assistant\n", is_sampled=False)
        em.finalize()

        # Merge prev mm_data with the new turn's items.
        merged_hashes = (
            dict(previous_multi_modal_data.mm_hashes)
            if previous_multi_modal_data
            else {}
        )
        merged_placeholders = (
            dict(previous_multi_modal_data.mm_placeholders)
            if previous_multi_modal_data
            else {}
        )
        merged_items = (
            dict(previous_multi_modal_data.mm_items)
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
            token_ids=em.token_ids,
            message_indices=em.message_indices,
            sampled_mask=em.sampled,
            message_roles=[m.get("role") or "" for m in new_messages],
            multi_modal_data=mm_data,
        )

    def _render_assistant(self, msg: Message, em: _Emitter) -> None:
        content = self._render_text_content(msg.get("content"))
        original_content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []

        # ``<|im_start|>assistant\n`` is template-injected scaffolding —
        # at inference the chat template emits these as the generation
        # prompt and the model never samples them. Splitting the text
        # at the ``\n`` after the role tag is safe: Qwen3 BPE treats
        # ``\n`` as a token boundary.
        em.special(self._im_start, is_sampled=False)
        em.text("assistant\n", is_sampled=False)

        # Body (content + tool calls) is the model-sampled portion.
        if not tool_calls:
            em.text(content, is_sampled=True)
        else:
            for tc_idx, tc in enumerate(tool_calls):
                if tc_idx == 0:
                    separator = "\n" if original_content else ""
                    em.text(content + separator, is_sampled=True)
                else:
                    em.text("\n", is_sampled=True)

                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                )

                em.special(self._tool_call, is_sampled=True)
                em.text(
                    '\n{"name": "' + name + '", "arguments": ' + args_str + "}\n",
                    is_sampled=True,
                )
                em.special(self._tool_call_end, is_sampled=True)

        # ``<|im_end|>`` is the model's stop signal — it samples this to
        # end its turn. The trailing ``\n`` is template-appended between
        # turns and never sampled.
        em.special(self._im_end, is_sampled=True)
        em.text("\n", is_sampled=False)

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        em: _Emitter,
        render_media_content,
    ) -> None:
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False. (render_media_content
        # also stamps is_sampled=False on its emissions.)
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )

        if not prev_is_tool:
            em.special(self._im_start, is_sampled=False)
            em.text("user", is_sampled=False)

        em.text("\n", is_sampled=False)
        em.special(self._tool_response, is_sampled=False)
        em.text("\n", is_sampled=False)
        render_media_content(messages[msg_idx].get("content"))
        em.text("\n", is_sampled=False)
        em.special(self._tool_response_end, is_sampled=False)

        if not next_is_tool:
            em.special(self._im_end, is_sampled=False)
            em.text("\n", is_sampled=False)
