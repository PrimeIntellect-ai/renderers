"""Qwen3.5 Renderer — hard-coded Python that mirrors the Qwen3.5 Jinja chat template.

Produces token-for-token identical output to tokenizer.apply_chat_template() while
also tracking which message produced each token (for per-token loss masks).

Multimodal: the Qwen3.5 family is itself a VLM (HF tag ``image-text-to-text``;
processor class ``Qwen3VLProcessor``). When a user/tool message carries an
``ImagePart``, the renderer emits the same ``<|vision_start|>``+N×``<|image_pad|>``
+``<|vision_end|>`` expansion as the HF chat template (``N =
image_grid_thw.prod() // merge_size**2``) and ships processed pixel_values via
``RenderedTokens.multi_modal_data``. Text-only inputs take the original fast
path and remain byte-identical to ``apply_chat_template``.
"""

from __future__ import annotations

import json
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    MultiModalData,
    ParsedResponse,
    PlaceholderRange,
    RenderedTokens,
    ToolSpec,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    should_preserve_past_thinking,
    trim_to_turn_close,
)
from renderers.configs import Qwen35RendererConfig
from renderers.parsing import parse_qwen35
from renderers.qwen3_vl import (
    _image_hash,
    _is_image_part,
    _is_video_part,
    _load_pil_image,
)

# ---------------------------------------------------------------------------
# Tool system prompt constants (must match the Jinja template exactly)
# ---------------------------------------------------------------------------

_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"

_TOOLS_FOOTER = "\n</tools>"

_TOOLS_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1"
    "\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter"
    "\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:"
    "\n- Function calls MUST follow the specified format:"
    " an inner <function=...></function> block must be nested within"
    " <tool_call></tool_call> XML tags"
    "\n- Required parameters MUST be specified"
    "\n- You may provide optional reasoning for your function call"
    " in natural language BEFORE the function call, but NOT after"
    "\n- If there is no function call available, answer the question like normal"
    " with your current knowledge and do not tell the user about function calls"
    "\n</IMPORTANT>"
)


# Per-model ``enable_thinking`` default, applied when the renderer config
# leaves it ``None``. The Qwen3.5 family ships two chat-template variants
# that differ only in the polarity of the gated thinking branch:
#
#   * Big sizes (4B / 9B / 35B-A3B / 122B-A10B / 397B-A17B) default
#     ``enable_thinking=true`` — an open ``<think>\n`` at the gen prompt.
#   * Small sizes (0.8B / 2B) flip it — default ``false``, emitting the
#     empty ``<think>\n\n</think>\n\n`` block.
#
# These are hard-coded (keyed by ``tokenizer.name_or_path``) rather than
# probed from the live ``chat_template``: probing meant calling
# ``apply_chat_template`` at construction, which pulls ``transformers`` onto
# the hot path and breaks bring-your-own-tokenizer use. The values are the
# ground truth pinned by ``tests/test_qwen35_size_coverage.py`` — both the
# polarity assertions and byte-parity against each size's own
# ``apply_chat_template``.
_ENABLE_THINKING_DEFAULTS: dict[str, bool] = {
    "Qwen/Qwen3.5-0.8B": False,
    "Qwen/Qwen3.5-2B": False,
    "Qwen/Qwen3.5-4B": True,
    "Qwen/Qwen3.5-9B": True,
    "Qwen/Qwen3.5-35B-A3B": True,
    "Qwen/Qwen3.5-122B-A10B": True,
    "Qwen/Qwen3.5-397B-A17B": True,
    # Qwen3.6 extends the Qwen3.5 template; same big-size polarity.
    "Qwen/Qwen3.6-35B-A3B": True,
}


def _default_enable_thinking(tokenizer) -> bool:
    """Hard-coded ``enable_thinking`` default for ``tokenizer``'s model.

    Falls back to ``True`` (the big-model default, and the majority of the
    family) for unknown / fine-tuned checkpoints whose ``name_or_path`` isn't
    in ``_ENABLE_THINKING_DEFAULTS``; pass an explicit ``enable_thinking=`` to
    a small-size fine-tune that needs ``False``.
    """
    return _ENABLE_THINKING_DEFAULTS.get(getattr(tokenizer, "name_or_path", ""), True)


class Qwen35Renderer:
    """Deterministic message → token renderer for Qwen3.5 models."""

    _config_cls: type = Qwen35RendererConfig

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Qwen35RendererConfig | None = None,
        *,
        processor: Any = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        cfg = config or type(self)._config_cls()
        # ``enable_thinking=None`` defers to the model's known default (see
        # ``_ENABLE_THINKING_DEFAULTS``). Materialise here so downstream reads
        # see a concrete bool; rebind the config with the resolved value so
        # introspection sees the same.
        if cfg.enable_thinking is None:
            cfg = cfg.model_copy(
                update={"enable_thinking": _default_enable_thinking(tokenizer)}
            )
        self.config = cfg

        # Look up special token IDs from the tokenizer (not hardcoded)
        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._think = self._token_id("<think>")
        self._think_end = self._token_id("</think>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")
        self._vision_start = self._token_id("<|vision_start|>")
        self._vision_end = self._token_id("<|vision_end|>")
        self._image_pad = self._token_id("<|image_pad|>")
        self._video_pad = self._token_id("<|video_pad|>")

        # Per-instance image-processor cache; see Qwen3VLRenderer for the
        # rationale (FIFO-bounded; same image seen across rollouts /
        # bridge re-renders).
        self._image_cache: dict[str, tuple[Any, int]] = {}

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        """Token-id → modality marker (1 = image, 2 = video) used by the
        trainer to build ``mm_token_type_ids``. Same convention as
        ``Qwen3VLRenderer``.
        """
        return {self._image_pad: 1, self._video_pad: 2}

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        from transformers import AutoProcessor

        name = getattr(self._tokenizer, "name_or_path", None)
        if not name:
            raise RuntimeError(
                "Qwen35Renderer needs a processor to render image / video parts. "
                "Pass `processor=AutoProcessor.from_pretrained(...)` to the "
                "constructor, or load the tokenizer with a known name_or_path "
                "so the processor can be auto-loaded."
            )
        self._processor = AutoProcessor.from_pretrained(name)
        return self._processor

    def _process_image(self, part: dict[str, Any]):
        """Resolve, process, and characterize a single image part.

        Returns ``(pil, processor_out, num_image_tokens, image_hash)``.
        Mirrors ``Qwen3VLRenderer._process_image``: hashes the loaded PIL,
        consults ``self._image_cache``, runs the HF image processor on
        miss, FIFO-evicts on overflow.
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
        if len(self._image_cache) >= self.config.image_cache_max:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[h] = (out, num_image_tokens)
        return pil, out, num_image_tokens, h

    @staticmethod
    def _content_has_media(content: Any) -> bool:
        """True when ``content`` is a structured list containing image / video parts."""
        if not isinstance(content, list):
            return False
        return any(
            isinstance(item, dict) and (_is_image_part(item) or _is_video_part(item))
            for item in content
        )

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Content rendering (mirrors the render_content Jinja macro)
    # ------------------------------------------------------------------

    @staticmethod
    def _render_content(content: Any) -> str:
        """Render message content to a text string (before tokenization).

        Handles string, list of text parts, and None. Image / video parts
        are silently skipped — callers that need the per-message text
        view (e.g. ``_last_query_index``, system / assistant / tool
        rendering) just want the text. The user branch detects media
        separately via ``_content_has_media`` and emits an
        image-interleaved stream that doesn't go through this helper.
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
                    else:
                        raise ValueError(f"Unexpected content item: {item}")
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    # ------------------------------------------------------------------
    # last_query_index computation
    # ------------------------------------------------------------------

    @staticmethod
    def _last_query_index(messages: list[Message]) -> int:
        """Find the index of the last 'real' user query (not a tool_response wrapper).

        Returns ``len(messages)`` — an out-of-range sentinel — when no such
        query exists. Callers compare ``msg_idx > last_query_index`` to
        decide whether an assistant turn sits after the last user query
        (and so keeps its thinking block). The sentinel makes that check
        uniformly ``False``, which is the only reasonable default for
        assistant-only inputs (e.g. the bridge's dummy-assistant render).
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "user":
                continue
            content = Qwen35Renderer._render_content(msg.get("content")).strip()
            if not (
                content.startswith("<tool_response>")
                and content.endswith("</tool_response>")
            ):
                return i
        return len(messages)

    # ------------------------------------------------------------------
    # Core render method
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
        # 1-indexed counters for ``add_vision_id`` (mirrors the Jinja's
        # ``image_count`` / ``video_count`` namespaces). Increment only
        # in the main message loop — the template renders the system
        # message with ``do_vision_count=False`` and would raise on
        # vision in system content anyway, so the renderer's
        # ``emit_image`` is only reached from user / tool emission paths.
        vision_counts = {"image": 0, "video": 0}

        def emit_ids(
            ids: list[int], msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

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
            emit_ids(
                self._encode(text),
                msg_idx,
                is_sampled=is_sampled,
                is_content=is_content,
            )

        def emit_text_segments(
            segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool
        ) -> None:
            """Tokenize concatenated segments as one BPE pass; per-token
            ``is_content`` follows each token's source segment.

            Lets call sites express "this wrap + this body, joined the
            same way as the chat template, but attributed separately"
            without splitting the encode call (which could shift BPE
            merges at the boundary)."""
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        def emit_image(part: dict[str, Any], msg_idx: int) -> None:
            # Image placeholders only appear in user / tool messages; the
            # model never samples them. Pin is_sampled=False here so
            # callers don't need to thread the flag through. The
            # ``<|image_pad|>`` placeholders represent caller-provided
            # image data, so they ARE body content (is_content=True);
            # the surrounding ``<|vision_start|>`` / ``<|vision_end|>``
            # specials are template scaffold.
            _, out, n, h = self._process_image(part)
            vision_counts["image"] += 1
            if self.config.add_vision_id:
                emit_text(
                    f"Picture {vision_counts['image']}: ",
                    msg_idx,
                    is_sampled=False,
                    is_content=False,
                )
            emit_special(
                self._vision_start, msg_idx, is_sampled=False, is_content=False
            )
            offset = len(tokens)
            for _ in range(n):
                emit_special(
                    self._image_pad, msg_idx, is_sampled=False, is_content=True
                )
            emit_special(self._vision_end, msg_idx, is_sampled=False, is_content=False)
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

        def emit_user_with_media(content_list: list[Any], msg_idx: int) -> None:
            """Emit a user message whose content list contains image parts.

            Buffers text segments and flushes them as single ``encode()``
            calls on special-token boundaries (``<|vision_start|>``,
            ``<|im_end|>``), matching how Jinja's ``render_content`` macro
            concatenates strings before tokenization. This preserves BPE
            byte-parity against ``apply_chat_template``.

            Within each flush the ``"user\\n"`` wrap is scaffold and the
            text parts are body. ``emit_text_segments`` carries that
            attribution through the BPE pass.
            """
            # Every token in a user message is conversation history that
            # the model never samples at inference.
            emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
            # First flush includes the ``"user\n"`` wrap as a scaffold
            # segment; subsequent flushes are pure body (after a media
            # break).
            buf_segments: list[tuple[str, bool]] = [("user\n", False)]

            def flush_buf() -> None:
                if buf_segments:
                    emit_text_segments(buf_segments, msg_idx, is_sampled=False)
                    buf_segments.clear()

            for item in content_list:
                if isinstance(item, str):
                    if item:
                        buf_segments.append((item, True))
                elif isinstance(item, dict):
                    if _is_image_part(item):
                        flush_buf()
                        emit_image(item, msg_idx)
                    elif _is_video_part(item):
                        raise NotImplementedError(
                            "Video parts are not yet supported by Qwen35Renderer."
                        )
                    elif "text" in item:
                        if item["text"]:
                            buf_segments.append((item["text"], True))
                    else:
                        raise ValueError(f"Unexpected content item: {item}")
                else:
                    raise ValueError(f"Unexpected content item: {item}")
            flush_buf()
            emit_special(self._im_end, msg_idx, is_sampled=False, is_content=False)
            emit_text("\n", msg_idx, is_sampled=False, is_content=False)

        # ── 1. System message + optional tools ──────────────────────
        first_is_system = messages[0].get("role") == "system"

        if tools:
            # System message index for attribution
            sys_idx = 0 if first_is_system else -1

            emit_special(self._im_start, sys_idx, is_sampled=False, is_content=False)
            # Body = system content (if any). Everything else in this
            # block — role tag, tools header / footer / instructions, the
            # JSON tool specs — is scaffold. The tools dict is
            # recoverable from the ``tools`` argument; don't re-attribute
            # its embedded JSON as message body.
            segments: list[tuple[str, bool]] = [
                ("system\n", False),
                (_TOOLS_HEADER, False),
            ]
            for tool in tools:
                segments.append(("\n" + json.dumps(tool, ensure_ascii=False), False))
            segments.append((_TOOLS_FOOTER, False))
            segments.append((_TOOLS_INSTRUCTIONS, False))
            if first_is_system:
                sys_content = self._render_content(messages[0].get("content")).strip()
                if sys_content:
                    segments.append(("\n\n", False))
                    segments.append((sys_content, True))
            emit_text_segments(segments, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False, is_content=False)
            emit_text("\n", sys_idx, is_sampled=False, is_content=False)
        elif first_is_system:
            sys_content = self._render_content(messages[0].get("content")).strip()
            emit_special(self._im_start, 0, is_sampled=False, is_content=False)
            sys_segments: list[tuple[str, bool]] = [("system\n", False)]
            if sys_content:
                sys_segments.append((sys_content, True))
            emit_text_segments(sys_segments, 0, is_sampled=False)
            emit_special(self._im_end, 0, is_sampled=False, is_content=False)
            emit_text("\n", 0, is_sampled=False, is_content=False)

        # ── 2. Compute last_query_index ─────────────────────────────
        last_qi = self._last_query_index(messages)

        # ── 3. Iterate messages ─────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = self._render_content(msg.get("content")).strip()

            if role == "system":
                if i != 0:
                    raise ValueError("System message must be at the beginning.")
                continue  # Already handled above

            elif role == "user":
                raw_content = msg.get("content")
                if self._content_has_media(raw_content):
                    emit_user_with_media(raw_content, i)
                else:
                    emit_special(self._im_start, i, is_sampled=False, is_content=False)
                    user_segments: list[tuple[str, bool]] = [("user\n", False)]
                    if content:
                        user_segments.append((content, True))
                    emit_text_segments(user_segments, i, is_sampled=False)
                    emit_special(self._im_end, i, is_sampled=False, is_content=False)
                    emit_text("\n", i, is_sampled=False, is_content=False)

            elif role == "assistant":
                preserve_thinking = should_preserve_past_thinking(
                    messages,
                    i,
                    preserve_all_thinking=self.config.preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self.config.preserve_thinking_between_tool_calls,
                )
                self._render_assistant(
                    msg,
                    i,
                    content,
                    last_qi,
                    preserve_thinking=preserve_thinking,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_ids=emit_ids,
                    emit_text_segments=emit_text_segments,
                )

            elif role == "tool":
                self._render_tool(
                    messages,
                    i,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_image=emit_image,
                    emit_text_segments=emit_text_segments,
                )

            else:
                raise ValueError(f"Unexpected message role: {role}")

        # ── 4. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_start, -1, is_sampled=False, is_content=False)
            emit_text("assistant\n", -1, is_sampled=False, is_content=False)
            if self.config.enable_thinking:
                emit_special(self._think, -1, is_sampled=False, is_content=False)
                emit_text("\n", -1, is_sampled=False, is_content=False)
            else:
                emit_special(self._think, -1, is_sampled=False, is_content=False)
                emit_text("\n\n", -1, is_sampled=False, is_content=False)
                emit_special(self._think_end, -1, is_sampled=False, is_content=False)
                emit_text("\n\n", -1, is_sampled=False, is_content=False)

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
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
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
        tools: list[ToolSpec] | None = None,
    ) -> ParsedResponse:
        return parse_qwen35(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            think_id=self._think,
            think_end_id=self._think_end,
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
            tools=tools,
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

        # ``add_vision_id`` numbers placeholders across the whole
        # conversation. The bridge can only seed that counter from
        # ``previous_multi_modal_data`` (raw prior token ids don't carry
        # the image/video count back), so if the caller asks for
        # ``add_vision_id=True`` while omitting prior mm-data on a
        # conversation that already contains images, the bridged
        # output would silently emit ``Picture 1:`` again. Refuse the
        # bridge in that case — the caller falls back to a full
        # re-render, which has the full message list and counts from
        # scratch correctly.
        if (
            self.config.add_vision_id
            and previous_multi_modal_data is None
            and self._vision_start in previous_ids
        ):
            return None

        # Seed combined-token list with prior turn so placeholder offsets
        # are absolute in the bridged sequence (matching ``render()``).
        # Parallel ``indices``/``sampled`` are seeded with ``-1``/``False``
        # for the prior portion — the bridge has no attribution info for
        # ``previous_ids``. Bridge-added tokens get proper ``msg_idx``
        # (relative to ``new_messages``) and uniformly ``False``
        # ``sampled``: nothing the bridge emits was model-sampled.
        # ``is_content`` follows the same rules as in :meth:`render` so
        # consumers can walk the trajectory and read each step's own
        # body mask; the prior portion is uniformly False since we have
        # no attribution info for it.
        tokens: list[int] = list(previous_ids)
        indices: list[int] = [-1] * len(previous_ids)
        sampled: list[bool] = [False] * len(previous_ids)
        content_mask: list[bool] = [False] * len(previous_ids)
        new_hashes: dict[str, list[str]] = {}
        new_placeholders: dict[str, list[PlaceholderRange]] = {}
        new_items: dict[str, list[dict[str, Any]]] = {}
        # Seed the ``add_vision_id`` counters from prior-turn images / videos
        # so the bridged turn's first placeholder gets ``Picture {prev+1}``.
        # Bridges can't recover the count from raw token ids, so callers
        # must thread ``previous_multi_modal_data`` through to keep
        # ``add_vision_id`` parity across turns.
        prev_image_count = 0
        prev_video_count = 0
        if previous_multi_modal_data is not None:
            prev_image_count = len(previous_multi_modal_data.mm_items.get("image", []))
            prev_video_count = len(previous_multi_modal_data.mm_items.get("video", []))
        vision_counts = {"image": prev_image_count, "video": prev_video_count}

        def emit_special(
            token_id: int,
            msg_idx: int = -1,
            *,
            is_sampled: bool = False,
            is_content: bool = False,
        ) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(
            text: str,
            msg_idx: int = -1,
            *,
            is_sampled: bool = False,
            is_content: bool = False,
        ) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_text_segments(
            segments: list[tuple[str, bool]],
            msg_idx: int = -1,
            *,
            is_sampled: bool = False,
        ) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        def emit_image(part: dict[str, Any], msg_idx: int = -1) -> None:
            _, out, n, h = self._process_image(part)
            vision_counts["image"] += 1
            if self.config.add_vision_id:
                emit_text(f"Picture {vision_counts['image']}: ", msg_idx)
            emit_special(self._vision_start, msg_idx)
            offset = len(tokens)
            for _ in range(n):
                emit_special(self._image_pad, msg_idx, is_content=True)
            emit_special(self._vision_end, msg_idx)
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

        def emit_user_with_media(content_list: list[Any], msg_idx: int) -> None:
            emit_special(self._im_start, msg_idx)
            buf_segments: list[tuple[str, bool]] = [("user\n", False)]

            def flush_buf() -> None:
                if buf_segments:
                    emit_text_segments(buf_segments, msg_idx)
                    buf_segments.clear()

            for item in content_list:
                if isinstance(item, str):
                    if item:
                        buf_segments.append((item, True))
                elif isinstance(item, dict):
                    if _is_image_part(item):
                        flush_buf()
                        emit_image(item, msg_idx)
                    elif _is_video_part(item):
                        raise NotImplementedError(
                            "Video parts are not yet supported by Qwen35Renderer."
                        )
                    elif "text" in item:
                        if item["text"]:
                            buf_segments.append((item["text"], True))
                    else:
                        raise ValueError(f"Unexpected content item: {item}")
                else:
                    raise ValueError(f"Unexpected content item: {item}")
            flush_buf()
            emit_special(self._im_end, msg_idx)
            emit_text("\n", msg_idx)

        # Trailing ``\n`` after ``<|im_end|>`` — ``render()`` emits it as
        # part of the prior turn, but vLLM stops on ``<|im_end|>`` so the
        # ``\n`` never makes it into prev_completion.
        emit_text("\n", -1)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            raw_content = msg.get("content")
            content = self._render_content(raw_content).strip()
            if role == "user":
                if self._content_has_media(raw_content):
                    emit_user_with_media(raw_content, i)
                else:
                    emit_special(self._im_start, i)
                    user_segments: list[tuple[str, bool]] = [("user\n", False)]
                    if content:
                        user_segments.append((content, True))
                    emit_text_segments(user_segments, i)
                    emit_special(self._im_end, i)
                    emit_text("\n", i)
            elif role == "system":
                emit_special(self._im_start, i)
                sys_segments: list[tuple[str, bool]] = [("system\n", False)]
                if content:
                    sys_segments.append((content, True))
                emit_text_segments(sys_segments, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)
            elif role == "tool":
                self._render_tool(
                    new_messages,
                    i,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_image=emit_image,
                    emit_text_segments=emit_text_segments,
                )
            else:
                return None

        # Generation prompt — matches the gen-prompt branch of ``render()``.
        emit_special(self._im_start, -1)
        emit_text("assistant\n", -1)
        if self.config.enable_thinking:
            emit_special(self._think, -1)
            emit_text("\n", -1)
        else:
            emit_special(self._think, -1)
            emit_text("\n\n", -1)
            emit_special(self._think_end, -1)
            emit_text("\n\n", -1)

        # Merge prev mm_data (images from earlier turns) with the new turn's.
        merged_hashes: dict[str, list[str]] = (
            dict(previous_multi_modal_data.mm_hashes)
            if previous_multi_modal_data
            else {}
        )
        merged_placeholders: dict[str, list[PlaceholderRange]] = (
            dict(previous_multi_modal_data.mm_placeholders)
            if previous_multi_modal_data
            else {}
        )
        merged_items: dict[str, list[dict[str, Any]]] = (
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

        bridge_roles = [m.get("role") or "" for m in new_messages]
        bridge_tool_names = extract_message_tool_names(new_messages)
        if not (merged_hashes or merged_placeholders or merged_items):
            return RenderedTokens(
                token_ids=tokens,
                message_indices=indices,
                sampled_mask=sampled,
                is_content=content_mask,
                message_roles=bridge_roles,
                message_tool_names=bridge_tool_names,
            )

        mm_data = MultiModalData(
            mm_hashes=merged_hashes,
            mm_placeholders=merged_placeholders,
            mm_items=merged_items,
        )
        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=bridge_roles,
            message_tool_names=bridge_tool_names,
            multi_modal_data=mm_data,
        )

    # ------------------------------------------------------------------
    # Assistant message rendering
    # ------------------------------------------------------------------

    def _should_render_thinking(self, msg_idx: int, last_query_index: int) -> bool:
        """Whether to emit a ``<think>`` block for the assistant message at ``msg_idx``.

        Qwen3.5 only emits thinking for assistant turns that sit after the
        last real user query. Subclasses (Qwen3.6) may override to mirror
        the newer template's ``preserve_thinking`` knob.
        """
        return msg_idx > last_query_index

    @staticmethod
    def _render_arg_value(arg_value: Any) -> str:
        """Serialise a single tool-call argument value to text.

        Mirrors the Jinja args-value branch: dicts/lists → compact JSON;
        everything else (including bool, int, None) → ``str(...)``. That
        matches Qwen3.5's ``tojson`` for mapping/sequence / ``string`` for
        the rest. Qwen3.6 overrides this to push non-strings through JSON
        so bools round-trip as ``true``/``false`` instead of ``True``/``False``.
        """
        if isinstance(arg_value, (dict, list)):
            return json.dumps(arg_value, ensure_ascii=False)
        return str(arg_value)

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        last_query_index: int,
        *,
        preserve_thinking: bool = False,
        emit_special,
        emit_text,
        emit_ids,
        emit_text_segments,
    ) -> None:
        # Extract reasoning_content
        reasoning_content = ""
        if isinstance(msg.get("reasoning_content"), str):
            reasoning_content = msg["reasoning_content"]
        elif "</think>" in content:
            # Split on </think> to separate reasoning from content
            before_think_end, after_think_end = content.split("</think>", 1)
            # Extract text after <think> (if present)
            if "<think>" in before_think_end:
                reasoning_content = before_think_end.split("<think>")[-1].lstrip("\n")
            else:
                reasoning_content = before_think_end.lstrip("\n")
            reasoning_content = reasoning_content.rstrip("\n")
            content = after_think_end.lstrip("\n")

        reasoning_content = reasoning_content.strip()

        # ``<|im_start|>assistant\n`` is template-injected scaffolding —
        # at inference the chat template emits these as the generation
        # prompt and the model never samples them. Marking the role tag
        # as ``is_sampled=False`` keeps the SFT loss mask aligned with
        # what the model would actually have produced. ``is_content`` is
        # also False here — the role tag isn't part of any message's
        # body, on any role.
        emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
        emit_text("assistant\n", msg_idx, is_sampled=False, is_content=False)

        # Build the model-sampled portion (think block + content + tool
        # calls). Text segments stay contiguous within each is_sampled
        # span to preserve BPE merges. For assistant messages the
        # invariant ``is_content == sampled_mask`` holds — every sampled
        # token is body, every scaffold token isn't. The XML-style tool
        # call tags (``<function=...>``, ``<parameter=...>``, etc.) are
        # part of the model's emitted output too — keep them
        # ``is_content=True`` per the assistant rule.
        emit_thinking = self._should_render_thinking(msg_idx, last_query_index) or (
            preserve_thinking and bool(reasoning_content)
        )
        if emit_thinking:
            # Include thinking block
            emit_special(self._think, msg_idx, is_sampled=True, is_content=True)
            emit_text(
                "\n" + reasoning_content + "\n",
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
            emit_special(self._think_end, msg_idx, is_sampled=True, is_content=True)
            emit_text("\n\n" + content, msg_idx, is_sampled=True, is_content=True)
        else:
            emit_text(content, msg_idx, is_sampled=True, is_content=True)

        # Tool calls
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for tc_idx, tc in enumerate(tool_calls):
                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})

                # Separator before <tool_call>
                if tc_idx == 0:
                    if content.strip():
                        emit_text("\n\n", msg_idx, is_sampled=True, is_content=True)
                    # else: no separator
                else:
                    emit_text("\n", msg_idx, is_sampled=True, is_content=True)

                emit_special(self._tool_call, msg_idx, is_sampled=True, is_content=True)
                emit_text(
                    "\n<function=" + name + ">\n",
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )

                # Render arguments
                # OpenAI canonical form: arguments is a JSON string. Parse it so the
                # per-argument rendering below still works.
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if isinstance(arguments, dict):
                    for arg_name, arg_value in arguments.items():
                        value_str = self._render_arg_value(arg_value)
                        emit_text(
                            "<parameter="
                            + arg_name
                            + ">\n"
                            + value_str
                            + "\n</parameter>\n",
                            msg_idx,
                            is_sampled=True,
                            is_content=True,
                        )

                emit_text("</function>\n", msg_idx, is_sampled=True, is_content=True)
                emit_special(
                    self._tool_call_end, msg_idx, is_sampled=True, is_content=True
                )

        # ``<|im_end|>`` is the model's stop signal — it samples this to
        # end its turn, so it is part of the sampled stream (and the
        # assistant's body). The trailing ``\n`` is template-appended
        # between turns and never sampled — scaffold for is_content too.
        emit_special(self._im_end, msg_idx, is_sampled=True, is_content=True)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

    # ------------------------------------------------------------------
    # Tool message rendering
    # ------------------------------------------------------------------

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        *,
        emit_special,
        emit_text,
        emit_image,
        emit_text_segments,
    ) -> None:
        # Consecutive tool messages share a single <|im_start|>user ... <|im_end|>
        # envelope. Whether to open and close the envelope depends only on the
        # neighbouring roles, never on the content type of this or any other
        # tool message — keep this predicate text/media-agnostic.
        # Tool messages are conversation history injected by the runtime
        # between assistant turns — the model never samples any of these
        # tokens, so every emission is is_sampled=False. The ``content``
        # field's body bytes get ``is_content=True``; everything else —
        # the ``<|im_start|>user`` wrap, the inter-section ``\n``s, the
        # ``<|tool_response>`` specials — is scaffold so the SFT mask
        # for tool body never trains the model to emit them.
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )
        raw_content = messages[msg_idx].get("content")

        if not prev_is_tool:
            emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
            emit_text("user", msg_idx, is_sampled=False, is_content=False)

        emit_text("\n", msg_idx, is_sampled=False, is_content=False)
        emit_special(self._tool_response, msg_idx, is_sampled=False, is_content=False)

        if self._content_has_media(raw_content):
            # Mirror the chat template's ``render_content`` macro for list
            # content: text segments BPE-encode together up to a media
            # boundary, then each image emits ``<|vision_start|>`` + N×
            # ``<|image_pad|>`` + ``<|vision_end|>`` inline.
            # ``_content_has_media`` returns False unless content is a list,
            # but the type checker can't follow that through the call.
            assert isinstance(raw_content, list)
            # First flush: leading ``"\n"`` is scaffold (separates the
            # ``<|tool_response>`` special from the body); subsequent
            # text items in this run are body. After a media break, the
            # buffer resets to pure body until the next media break or
            # end-of-content.
            buf_segments: list[tuple[str, bool]] = [("\n", False)]

            def flush_buf() -> None:
                if buf_segments:
                    emit_text_segments(buf_segments, msg_idx, is_sampled=False)
                    buf_segments.clear()

            for item in raw_content:
                if isinstance(item, str):
                    if item:
                        buf_segments.append((item, True))
                elif isinstance(item, dict):
                    if _is_image_part(item):
                        flush_buf()
                        emit_image(item, msg_idx)
                    elif _is_video_part(item):
                        raise NotImplementedError(
                            "Video parts are not yet supported by Qwen35Renderer."
                        )
                    elif "text" in item:
                        if item["text"]:
                            buf_segments.append((item["text"], True))
                    else:
                        raise ValueError(f"Unexpected content item: {item}")
                else:
                    raise ValueError(f"Unexpected content item: {item}")
            flush_buf()
            emit_text("\n", msg_idx, is_sampled=False, is_content=False)
        else:
            content = self._render_content(raw_content).strip()
            # ``\n`` + content + ``\n`` — body is the middle segment only.
            # Single BPE pass over the joined text preserves boundary
            # merges.
            emit_text_segments(
                [("\n", False), (content, True), ("\n", False)],
                msg_idx,
                is_sampled=False,
            )

        emit_special(
            self._tool_response_end, msg_idx, is_sampled=False, is_content=False
        )

        if not next_is_tool:
            emit_special(self._im_end, msg_idx, is_sampled=False, is_content=False)
            emit_text("\n", msg_idx, is_sampled=False, is_content=False)
