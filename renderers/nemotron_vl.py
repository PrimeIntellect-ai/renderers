"""Nemotron-VL renderer: Nemotron-3 chat template + Omni-style image inputs.

Extends :class:`renderers.nemotron3.Nemotron3Renderer` for the vision-grafted
Nemotron checkpoints (Nemotron-3-Super-VL / Ultra-VL). Text-only message lists
render byte-identically to the base class (it is called directly). Messages
with image content parts follow Nano Omni's processor semantics:

- Each image is resized to a SINGLE dynamic-resolution tile whose 16px patch
  grid ``(th, tw)`` lands in ``[min_num_patches, max_num_patches]``, aspect
  preserved, dims snapped to even (pixel-shuffle divisor 2). Port of
  ``NemotronH_Nano_Omni_Reasoning_V3ImageProcessor`` (bicubic+antialias
  resize, CLIP normalization) — byte-equivalent geometry validated against
  the reference in prime-rl ``models/validation/check_tiling_parity.py``.
- The token stream gets ``<img>`` + N×``<image>`` + ``</img>`` where
  ``N = th*tw // 4`` (Omni's ``processing.py`` replacement), all
  ``is_sampled=False``; the ``<image>`` run is body content, markers scaffold.
- ``multi_modal_data.mm_items["image"]`` entries carry PATCHIFIED pixels:
  ``pixel_values`` of shape ``(th*tw, 3*16*16)`` (row-major patch order,
  identical to RADIO's Im2Patches rearrange) plus ``image_grids`` ``[[th, tw]]``.
  Patchified layout keeps tensors concat-able along dim 0 across images of
  different tile sizes — required by prime-rl's sample packing — and the
  NemotronVL model reconstructs each tile from the grid dims.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    MultiModalData,
    PlaceholderRange,
    RenderedTokens,
    ToolSpec,
    attribute_text_segments,
    extract_message_tool_names,
)
from renderers.configs import NemotronVLRendererConfig
from renderers.nemotron3 import Nemotron3Renderer
from renderers.qwen3_vl import _image_hash, _is_image_part, _is_video_part, _load_pil_image


def _messages_have_images(messages: list[Message]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if _is_image_part(item):
                    return True
    return False


class NemotronVLRenderer(Nemotron3Renderer):
    """Deterministic message → token renderer for Nemotron-VL models."""

    _config_cls = NemotronVLRendererConfig
    _ultra = False

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: NemotronVLRendererConfig | None = None,
    ):
        super().__init__(tokenizer, config)
        self._image = self._token_id("<image>")
        self._img_start = self._token_id("<img>")
        self._img_end = self._token_id("</img>")
        # hash -> (patches ndarray, th, tw, num_tokens); FIFO-bounded.
        self._image_cache: dict[str, tuple[np.ndarray, int, int, int]] = {}

    @property
    def mm_token_type_id_map(self) -> dict[int, int]:
        """1 = image placeholder; the trainer builds ``mm_token_type_ids`` from this."""
        return {self._image: 1}

    # ------------------------------------------------------------------
    # Omni dynamic-resolution image preprocessing (processor port)
    # ------------------------------------------------------------------

    def _target_patch_grid(self, width: int, height: int) -> tuple[int, int]:
        """Port of the Omni processor's ``_compute_target_patches`` (returns (th, tw))."""
        import math

        cfg = self.config
        budget = (cfg.max_model_len - 4) * 4  # vLLM reserve, x4 pre-shuffle patches
        tokens_available = max(min(budget, cfg.max_num_patches), cfg.min_num_patches)

        closest_h = round(height / cfg.patch_size + 0.5)
        closest_w = round(width / cfg.patch_size + 0.5)
        factor = min(math.sqrt(tokens_available / (closest_h * closest_w)), 1.0)
        th, tw = math.floor(factor * closest_h), math.floor(factor * closest_w)

        if tokens_available > cfg.min_num_patches and th * tw < cfg.min_num_patches:
            up = math.sqrt(cfg.min_num_patches / (th * tw))
            th, tw = math.ceil(up * th), math.ceil(up * tw)

        divisor = 2  # pixel-shuffle factor for downsample_ratio 0.5
        rem = th % divisor
        if rem:
            th = th + (divisor - rem) if (th + divisor - rem) * tw <= tokens_available else max(divisor, th - rem)
        rem = tw % divisor
        if rem:
            tw = tw + (divisor - rem) if th * (tw + divisor - rem) <= tokens_available else max(divisor, tw - rem)
        return th, tw

    def _preprocess_image(self, pil) -> tuple[np.ndarray, int, int]:
        """PIL RGB -> ((th*tw, 3*p*p) float32 patches, th, tw).

        Resize (torch bicubic, antialiased — matches Omni/vLLM bit-exactly) +
        CLIP normalization + Im2Patches rearrange
        ``b c (py p) (px p) -> b (py px) (c p p)``.
        """
        import torch
        import torch.nn.functional as F

        cfg = self.config
        p = cfg.patch_size
        th, tw = self._target_patch_grid(pil.width, pil.height)
        target_h, target_w = th * p, tw * p

        arr = np.asarray(pil, dtype=np.uint8)
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        if t.shape[-2] != target_h or t.shape[-1] != target_w:
            t = F.interpolate(t, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor(cfg.norm_mean).view(1, 3, 1, 1)
        std = torch.tensor(cfg.norm_std).view(1, 3, 1, 1)
        t = (t / 255.0 - mean) / std

        t = t.reshape(1, 3, th, p, tw, p).permute(0, 2, 4, 1, 3, 5).reshape(1, th * tw, 3 * p * p)
        return t.squeeze(0).numpy(), th, tw

    def _process_image(self, part: dict[str, Any]) -> tuple[np.ndarray, int, int, int, str]:
        """Resolve + preprocess one image part -> (patches, th, tw, num_tokens, hash)."""
        pil = _load_pil_image(part)
        h = _image_hash(pil)
        cached = self._image_cache.get(h)
        if cached is not None:
            return (*cached, h)
        patches, th, tw = self._preprocess_image(pil)
        num_tokens = (th * tw) // 4
        if len(self._image_cache) >= self.config.image_cache_max:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[h] = (patches, th, tw, num_tokens)
        return patches, th, tw, num_tokens, h

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
        if not _messages_have_images(messages):
            return super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        if tools:
            raise NotImplementedError("NemotronVLRenderer does not support tools together with images yet.")

        original_messages = list(messages)
        messages, auto_system_injected = self._normalize_messages(messages)
        idx_offset = -1 if auto_system_injected else 0

        def orig_idx(i: int) -> int:
            return -1 if (auto_system_injected and i == 0) else i + idx_offset

        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []
        mm_hashes: dict[str, list[str]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        mm_items: dict[str, list[dict[str, Any]]] = {}

        def emit_special(token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text_segments(segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool) -> None:
            for tok_id, is_content in attribute_text_segments(self._tokenizer, segments):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        def emit_text(text: str, msg_idx: int, *, is_sampled: bool, is_content: bool) -> None:
            for tok_id in self._encode(text):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        def emit_image(part: dict[str, Any], msg_idx: int) -> None:
            # Omni processing.py: <image> in text -> <img> + N x <image> + </img>.
            # Markers are renderer scaffold; the <image> run represents
            # caller-provided image data (body content). Nothing is sampled.
            patches, th, tw, n, h = self._process_image(part)
            emit_special(self._img_start, msg_idx, is_sampled=False, is_content=False)
            offset = len(tokens)
            tokens.extend([self._image] * n)
            indices.extend([msg_idx] * n)
            sampled.extend([False] * n)
            content_mask.extend([True] * n)
            emit_special(self._img_end, msg_idx, is_sampled=False, is_content=False)
            mm_hashes.setdefault("image", []).append(h)
            mm_placeholders.setdefault("image", []).append(PlaceholderRange(offset=offset, length=n))
            mm_items.setdefault("image", []).append(
                {"pixel_values": patches, "image_grids": np.array([[th, tw]], dtype=np.int64)}
            )

        def emit_media_content(content: Any, msg_idx: int, lead_segments: list[tuple[str, bool]], tail: str) -> None:
            """Emit scaffold + interleaved text/image content for one user message.

            Buffered text segments flush (single BPE pass, scaffold/body
            attributed) whenever an image marker — an atomic BPE boundary —
            is emitted.
            """
            segments = list(lead_segments)

            def flush() -> None:
                nonlocal segments
                if segments:
                    emit_text_segments(segments, msg_idx, is_sampled=False)
                    segments = []

            if isinstance(content, str):
                if content:
                    segments.append((content, True))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        if item:
                            segments.append((item, True))
                    elif isinstance(item, dict):
                        if _is_image_part(item):
                            flush()
                            emit_image(item, msg_idx)
                        elif _is_video_part(item):
                            raise NotImplementedError("Video parts are not supported by NemotronVLRenderer.")
                        elif item.get("text"):
                            segments.append((item["text"], True))
            elif content is not None:
                raise TypeError(f"Unexpected content type: {type(content)}")
            if tail:
                segments.append((tail, False))
            flush()

        # ── 1. System message (tools rejected above) ─────────────────
        first_is_system = messages[0].get("role") == "system"
        if first_is_system:
            sys_idx = orig_idx(0)
            sys_content = self._render_content(messages[0].get("content"))
            emit_special(self._im_start, sys_idx, is_sampled=False, is_content=False)
            sys_segments: list[tuple[str, bool]] = [("system\n", False)]
            if sys_content:
                sys_segments.append((sys_content, True))
            emit_text_segments(sys_segments, sys_idx, is_sampled=False)
            emit_special(self._im_end, sys_idx, is_sampled=False, is_content=False)
            emit_text("\n", sys_idx, is_sampled=False, is_content=False)

        last_user_idx_norm = -1
        for j in range(len(messages) - 1, -1, -1):
            if messages[j].get("role") == "user":
                last_user_idx_norm = j
                break

        # ── 2. Iterate messages ─────────────────────────────────────
        for i, msg in enumerate(messages):
            role = msg["role"]
            msg_orig_idx = orig_idx(i)

            if role == "system":
                if i != 0:
                    raise ValueError("System message must be at the beginning.")
                continue

            if role == "user":
                emit_special(self._im_start, msg_orig_idx, is_sampled=False, is_content=False)
                tail = self._effort_hint if (self._effort_hint and i == last_user_idx_norm) else ""
                emit_media_content(msg.get("content"), msg_orig_idx, [("user\n", False)], tail)
                emit_special(self._im_end, msg_orig_idx, is_sampled=False, is_content=False)
                emit_text("\n", msg_orig_idx, is_sampled=False, is_content=False)

            elif role == "assistant":
                include_content = not self.config.truncate_history_thinking or i >= last_user_idx_norm
                self._render_assistant(
                    msg,
                    msg_orig_idx,
                    self._render_content(msg.get("content")),
                    include_content=include_content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            elif role == "tool":
                content = msg.get("content")
                if isinstance(content, list) and any(_is_image_part(item) for item in content):
                    raise NotImplementedError("Images in tool messages are not supported by NemotronVLRenderer.")
                self._render_tool(
                    messages,
                    i,
                    self._render_content(content),
                    msg_orig_idx=msg_orig_idx,
                    auto_system_injected=auto_system_injected,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

            else:
                raise ValueError(f"Unexpected message role: {role}")

        # ── 3. Generation prompt ────────────────────────────────────
        if add_generation_prompt:
            emit_special(self._im_start, -1, is_sampled=False, is_content=False)
            emit_text("assistant\n", -1, is_sampled=False, is_content=False)
            emit_special(self._think, -1, is_sampled=False, is_content=False)
            if self.config.enable_thinking:
                emit_text("\n", -1, is_sampled=False, is_content=False)
            else:
                emit_special(self._think_end, -1, is_sampled=False, is_content=False)

        mm_data = MultiModalData(mm_hashes=mm_hashes, mm_placeholders=mm_placeholders, mm_items=mm_items)
        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in original_messages],
            message_tool_names=extract_message_tool_names(original_messages),
            multi_modal_data=mm_data,
        )


__all__ = ["NemotronVLRenderer"]
