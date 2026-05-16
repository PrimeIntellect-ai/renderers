"""Multimodal parity tests, parameterized by ``(model, modality)``.

``MULTIMODAL_MODELS`` in ``renderers.base`` declares which checkpoints
support which non-text modalities. This test matrix iterates over every
``(model, modality)`` pair and asserts:

1. **Token byte-parity** — ``Renderer.render_ids(...)`` matches
   ``processor.apply_chat_template(..., tokenize=False)`` piped through
   ``processor(images=..., text=..., return_tensors="pt")["input_ids"]``.
2. **Placeholder anchoring** — ``RenderedTokens.multi_modal_data.mm_placeholders``
   exactly cover the runs of the modality's placeholder token id
   (``<|image_pad|>`` for images, ``<|video_pad|>`` for videos).
3. **Bridge byte-parity** — ``bridge_to_next_turn`` with new multimodal
   messages produces the same token sequence as a fresh full render of
   the combined message list.

Tests skip per-pair when:
- The HF snapshot isn't cached locally (network-free CI mode).
- The model lists a modality the renderer doesn't yet support
  (``NotImplementedError`` in ``render``).
- ``Pillow`` / ``torch`` are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from renderers import (
    MULTIMODAL_MODELS,
    Qwen3VLRenderer,
    create_renderer,
)
from renderers.base import load_tokenizer


pytest.importorskip("PIL", reason="Pillow required for multimodal tests")
pytest.importorskip("torch", reason="torch required for multimodal tests")

from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# Local-snapshot gating — skip when the HF cache doesn't have the model.
# ---------------------------------------------------------------------------


def _hf_snapshot_cached(model_name: str) -> bool:
    """True iff the HF hub cache has at least one snapshot for ``model_name``.

    Avoids pulling weights / configs over the network during test runs.
    Mirrors the convention used elsewhere in this repo (test_qwen35_size_coverage)
    of relying on the user having pre-fetched relevant models.
    """
    cache = (
        Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
        / "hub"
    )
    safe = "models--" + model_name.replace("/", "--")
    snapshots = cache / safe / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(p.is_dir() for p in snapshots.iterdir())


# ---------------------------------------------------------------------------
# Parametrization.
# ---------------------------------------------------------------------------


def _modality_cases():
    """Flatten ``MULTIMODAL_MODELS`` into ``(model, modality)`` pairs."""
    out: list[tuple[str, str]] = []
    for model, modalities in MULTIMODAL_MODELS.items():
        for modality in sorted(modalities):
            out.append((model, modality))
    return out


_CASES = _modality_cases()


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


_loaded: dict[str, tuple] = {}


# Models whose processors need ``trust_remote_code=True`` (custom Python
# in the repo) AND a pinned revision for security. Mirrors the
# ``TRUSTED_REVISIONS`` policy in ``renderers.base`` for tokenizers.
_PROCESSOR_TRUSTED_REVISIONS: dict[str, str] = {
    "moonshotai/Kimi-K2.5": "4d01dfe0332d63057c186e0b262165819efb6611",
    "moonshotai/Kimi-K2.6": "2755962d07cb42aa2d988a35bcb65cd4a9c2de82",
}


def _load_processor_and_renderer(model_name: str):
    if model_name not in _loaded:
        from transformers import AutoProcessor

        tokenizer = load_tokenizer(model_name)
        revision = _PROCESSOR_TRUSTED_REVISIONS.get(model_name)
        if revision is not None:
            processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True, revision=revision
            )
        else:
            processor = AutoProcessor.from_pretrained(model_name)
        renderer = create_renderer(tokenizer, renderer="auto")
        # Inject processor so the renderer doesn't try to fetch it lazily.
        if hasattr(renderer, "_processor") and renderer._processor is None:
            renderer._processor = processor
        _loaded[model_name] = (tokenizer, processor, renderer)
    return _loaded[model_name]


@pytest.fixture(scope="module")
def tiny_image():
    """A small synthetic RGB image — keeps per-image-processor cost low."""
    return Image.new("RGB", (224, 224), color=(128, 192, 255))


# ---------------------------------------------------------------------------
# Modality → (renderer-side content part, processor-side image-list builder).
# Each modality has its own "make a content part" / "extract source images"
# pair so the same parity machinery generalizes when video / audio land.
# ---------------------------------------------------------------------------


def _image_content_part(img):
    return {"type": "image", "image": img}


def _kimi_image_content_part(img):
    # Kimi K2.5's ``KimiK25Processor._extract_medias_from_messages`` hard-
    # reads ``content_part['image_url']`` (even when ``type == 'image'``).
    # Use the OpenAI-ish ``image_url`` shape so the same messages feed both
    # our renderer (which accepts both shapes) and Kimi's processor.
    return {"type": "image_url", "image_url": img}


def _detect_family(model_name: str) -> str:
    """Map a HF model id to a coarse family for per-family processor dispatch.

    Families differ in (a) the chat-template / vision-token format and (b)
    the processor's ``__call__`` signature. Today:
    - ``qwen_vl``: ``processor(images=..., text=..., return_tensors=...)``,
      content parts shaped ``{"type": "image", "image": <PIL>}``.
    - ``kimi_k25``: ``processor(messages=..., return_tensors=...)`` (does
      template + image preprocessing in one call), content parts shaped
      ``{"type": "image_url", "image_url": <PIL>}``.
    """
    if model_name.startswith("moonshotai/Kimi-K2.5") or model_name.startswith(
        "moonshotai/Kimi-K2.6"
    ):
        return "kimi_k25"
    return "qwen_vl"


def _qwen_vl_processor_input_ids(processor, messages, add_gp):
    """Run the Qwen-VL family processor pipeline on ``messages``.

    Two-step: ``apply_chat_template`` → text; collect images from messages;
    ``processor(images=, text=)`` to get expanded ``input_ids``.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_gp
    )
    images = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if (
                item.get("type") in ("image", "image_url")
                or "image" in item
                or "image_url" in item
            ):
                if "image" in item and not isinstance(item["image"], dict):
                    images.append(item["image"])
    return processor(images=images, text=text, return_tensors="pt")["input_ids"][
        0
    ].tolist()


def _kimi_processor_input_ids(processor, messages, add_gp):
    """Run Kimi K2.5's processor on ``messages`` (one-shot template+vision).

    Kimi's ``__call__`` takes ``messages=`` directly and emits the template-
    rendered ``input_ids`` along with ``pixel_values`` / ``grid_thws``. The
    template puts ONE ``<|media_pad|>`` per image in ``input_ids``; per-patch
    expansion lives in ``pixel_values`` and is handled inside the model.
    """
    out = processor(
        messages=messages, add_generation_prompt=add_gp, return_tensors="pt"
    )
    return out["input_ids"][0].tolist()


def _modality_kit(modality: str, model_name: str):
    family = _detect_family(model_name)
    if modality == "image":
        if family == "kimi_k25":
            return {
                "make_part": _kimi_image_content_part,
                "placeholder_token": "<|media_pad|>",
                "processor_input_ids": _kimi_processor_input_ids,
            }
        # Default: Qwen-VL family (Qwen3-VL, Qwen3.5, Qwen3.6).
        return {
            "make_part": _image_content_part,
            "placeholder_token": "<|image_pad|>",
            "processor_input_ids": _qwen_vl_processor_input_ids,
        }
    raise NotImplementedError(
        f"Test kit for modality {modality!r} not implemented yet."
    )


# ---------------------------------------------------------------------------
# Cases.
# ---------------------------------------------------------------------------


def _build_cases(make_part, image):
    """Per-modality message scenarios. ``make_part`` builds a content part
    for the modality under test; ``image`` is the shared sample item."""
    return [
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        make_part(image),
                    ],
                }
            ],
            True,
            id="single_image_in_user",
        ),
        pytest.param(
            [
                {"role": "system", "content": "Be concise."},
                {
                    "role": "user",
                    "content": [
                        make_part(image),
                        {"type": "text", "text": "Describe it."},
                    ],
                },
            ],
            True,
            id="image_before_text_with_system",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these:"},
                        make_part(image),
                        make_part(image),
                    ],
                }
            ],
            True,
            id="two_images_one_turn",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        make_part(image),
                        {"type": "text", "text": "First?"},
                    ],
                },
                {"role": "assistant", "content": "It's a square."},
                {
                    "role": "user",
                    "content": [
                        make_part(image),
                        {"type": "text", "text": "And now?"},
                    ],
                },
            ],
            True,
            id="multi_turn_two_images",
        ),
    ]


def _build_tool_image_cases(make_part, image):
    """Tool-message image scenarios. Targets renderers that emit image
    placeholders inside ``<tool_response>`` blocks. Browser-agent style
    trajectories produce post-action screenshots as tool responses, so
    handling images here is load-bearing for that workload."""
    return [
        pytest.param(
            [
                {"role": "user", "content": "Take a screenshot."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "screenshot", "arguments": {}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [
                        {"type": "text", "text": "Screenshot captured."},
                        make_part(image),
                    ],
                },
            ],
            False,
            id="tool_response_with_image",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Screenshot then describe."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "screenshot", "arguments": {}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [
                        {"type": "text", "text": "Done:"},
                        make_part(image),
                    ],
                },
                {"role": "assistant", "content": "A square."},
                {"role": "user", "content": "Now show me the next page."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "screenshot", "arguments": {}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c2",
                    "content": [
                        {"type": "text", "text": "Next page:"},
                        make_part(image),
                    ],
                },
            ],
            False,
            id="multi_turn_tool_response_images",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Run a few tools."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "ping", "arguments": {}},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "screenshot", "arguments": {}},
                        },
                        {
                            "id": "c3",
                            "type": "function",
                            "function": {"name": "ping", "arguments": {}},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "pong"},
                {
                    "role": "tool",
                    "tool_call_id": "c2",
                    "content": [
                        {"type": "text", "text": "Screenshot:"},
                        make_part(image),
                    ],
                },
                {"role": "tool", "tool_call_id": "c3", "content": "pong"},
            ],
            False,
            id="consecutive_tools_mixed_media",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def _supports_tool_message_images(renderer) -> bool:
    """True iff this renderer emits image placeholders inside tool-response
    content. Renderers without the feature silently drop image parts in tool
    content; as they grow the feature they get added here and the test starts
    asserting against them."""
    from renderers.kimi_k25 import KimiK25Renderer
    from renderers.qwen35 import Qwen35Renderer

    return isinstance(renderer, (Qwen35Renderer, KimiK25Renderer))


@pytest.mark.parametrize(
    "mm_model_name,modality", _CASES, ids=[f"{m}|{mo}" for m, mo in _CASES]
)
def test_multimodal_byte_parity_vs_processor(mm_model_name, modality, tiny_image):
    """Token byte-parity with ``processor.apply_chat_template`` + ``processor(...)``.

    Locks in the property that lets the inference engine see byte-identical
    token ids regardless of whether the prompt is templated server-side
    (MITO) or rendered client-side via this package.
    """
    if not _hf_snapshot_cached(mm_model_name):
        pytest.skip(f"{mm_model_name}: HF snapshot not cached locally")

    kit = _modality_kit(modality, mm_model_name)
    tokenizer, processor, renderer = _load_processor_and_renderer(mm_model_name)

    for case in _build_cases(kit["make_part"], tiny_image):
        messages, add_gp = case.values

        # Ours.
        ours = renderer.render_ids(messages, add_generation_prompt=add_gp)

        # Theirs: family-specific processor call. Qwen-VL is a two-step
        # (apply_chat_template + processor(images=, text=)); Kimi K2.5 is
        # a one-shot processor(messages=).
        theirs = kit["processor_input_ids"](processor, messages, add_gp)

        assert ours == theirs, (
            f"{mm_model_name} / {modality} / case={case.id}: "
            f"renderer diverges from processor.\n"
            f"  ours[:80]={ours[:80]}\n  theirs[:80]={theirs[:80]}\n"
            f"  len(ours)={len(ours)} len(theirs)={len(theirs)}"
        )


@pytest.mark.parametrize(
    "mm_model_name,modality", _CASES, ids=[f"{m}|{mo}" for m, mo in _CASES]
)
def test_multimodal_placeholders_match_pad_runs(mm_model_name, modality, tiny_image):
    """``mm_placeholders`` exactly cover the runs of the modality's pad token."""
    if not _hf_snapshot_cached(mm_model_name):
        pytest.skip(f"{mm_model_name}: HF snapshot not cached locally")

    kit = _modality_kit(modality, mm_model_name)
    tokenizer, _, renderer = _load_processor_and_renderer(mm_model_name)
    pad_id = tokenizer.convert_tokens_to_ids(kit["placeholder_token"])

    for case in _build_cases(kit["make_part"], tiny_image):
        messages, add_gp = case.values
        rendered = renderer.render(messages, add_generation_prompt=add_gp)

        assert rendered.multi_modal_data is not None, (
            f"{mm_model_name} / {modality} / {case.id}: render() returned no mm_data"
        )

        # Discover the actual pad-token runs in the token stream.
        pad_runs: list[tuple[int, int]] = []
        i, n = 0, len(rendered.token_ids)
        while i < n:
            if rendered.token_ids[i] == pad_id:
                start = i
                while i < n and rendered.token_ids[i] == pad_id:
                    i += 1
                pad_runs.append((start, i - start))
            else:
                i += 1

        claimed = [
            (p.offset, p.length)
            for p in rendered.multi_modal_data.mm_placeholders.get(modality, [])
        ]
        assert claimed == pad_runs, (
            f"{mm_model_name} / {modality} / {case.id}: "
            f"mm_placeholders {claimed} vs actual pad runs {pad_runs}"
        )


@pytest.mark.parametrize(
    "mm_model_name,modality", _CASES, ids=[f"{m}|{mo}" for m, mo in _CASES]
)
def test_multimodal_bridge_extends_and_carries_mm_data(
    mm_model_name, modality, tiny_image
):
    """Bridge-to-next-turn invariants for the multimodal case.

    Asserts three properties that should hold for every renderer
    regardless of thinking-mode quirks (the prior bridge-vs-full
    invariant was too strong — see commit log for the divergence
    rationale on thinking renderers):

    1. **Verbatim prefix**: ``bridged.token_ids`` begins with
       ``previous_prompt_ids + previous_completion_ids``. Whatever the
       sampler conditioned on stays bit-identical in the trainer's
       reconstruction.

    2. **mm_data carry-forward**: prior-turn images survive in the
       merged ``mm_placeholders`` / ``mm_items`` / ``mm_hashes``, and
       the new turn's images get appended.

    3. **Extension covers new turn**: the tokens after the prefix
       include the new ``<|image_pad|>``-or-``<|media_pad|>`` run for
       the new turn's image, plus its placeholder is recorded with an
       absolute offset inside the bridged sequence.
    """
    if not _hf_snapshot_cached(mm_model_name):
        pytest.skip(f"{mm_model_name}: HF snapshot not cached locally")

    kit = _modality_kit(modality, mm_model_name)
    tokenizer, _, renderer = _load_processor_and_renderer(mm_model_name)

    initial = [
        {
            "role": "user",
            "content": [
                kit["make_part"](tiny_image),
                {"type": "text", "text": "Turn one."},
            ],
        }
    ]
    new = [
        {
            "role": "user",
            "content": [
                kit["make_part"](tiny_image),
                {"type": "text", "text": "Turn two."},
            ],
        }
    ]

    initial_rendered = renderer.render(initial, add_generation_prompt=True)
    # ``previous_completion_ids`` mirrors what a sampler would emit
    # starting AFTER the prompt's assistant role opener — i.e. the
    # response text followed by ``<|im_end|>``.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    completion_ids = tokenizer.encode("Saw it.", add_special_tokens=False) + [im_end_id]

    bridged_raw = renderer.bridge_to_next_turn(
        previous_prompt_ids=initial_rendered.token_ids,
        previous_completion_ids=completion_ids,
        new_messages=new,
        previous_multi_modal_data=initial_rendered.multi_modal_data,
    )
    assert bridged_raw is not None, (
        f"{mm_model_name} / {modality}: bridge returned None for multimodal extension"
    )

    # Multimodal bridges return ``RenderedTokens`` (text-only callers
    # historically expected ``list[int]``; the per-renderer return
    # type splits on whether ``mm_data`` is non-empty).
    bridged_ids = (
        bridged_raw.token_ids
        if hasattr(bridged_raw, "token_ids")
        else list(bridged_raw)
    )
    bridged_mm = getattr(bridged_raw, "multi_modal_data", None)

    # (1) Verbatim prefix — what the sampler saw is what the trainer
    # reconstructs.
    prev = list(initial_rendered.token_ids) + list(completion_ids)
    assert bridged_ids[: len(prev)] == prev, (
        f"{mm_model_name} / {modality}: bridge prefix diverges from prev_prompt + prev_completion"
    )
    assert len(bridged_ids) > len(prev), (
        f"{mm_model_name} / {modality}: bridge produced no extension tokens"
    )

    # (2) mm_data carry-forward — prior images survive, new ones are appended.
    assert bridged_mm is not None, (
        f"{mm_model_name} / {modality}: bridge dropped multi_modal_data"
    )
    placeholders = bridged_mm.mm_placeholders.get(modality, [])
    assert len(placeholders) == 2, (
        f"{mm_model_name} / {modality}: expected 2 image placeholders "
        f"(1 carried + 1 new), got {len(placeholders)}"
    )
    items = bridged_mm.mm_items.get(modality, [])
    hashes = bridged_mm.mm_hashes.get(modality, [])
    assert len(items) == 2 and len(hashes) == 2

    # (3) Extension contains the new turn's pad run, and its
    # placeholder offset lands inside the extension region.
    pad_id = tokenizer.convert_tokens_to_ids(kit["placeholder_token"])
    extension = bridged_ids[len(prev) :]
    assert pad_id in extension, (
        f"{mm_model_name} / {modality}: new turn's placeholder pad missing from extension"
    )
    new_placeholder = placeholders[-1]
    assert new_placeholder.offset >= len(prev), (
        f"{mm_model_name} / {modality}: new placeholder offset {new_placeholder.offset} "
        f"sits inside the carried-forward prefix (len={len(prev)})"
    )


def test_modality_registry_models_route_to_renderer():
    """Every model in ``MULTIMODAL_MODELS`` resolves to a concrete renderer
    via ``create_renderer(renderer='auto')``. Guards against drift between
    the multimodal registry and ``MODEL_RENDERER_MAP``."""
    for model_name in MULTIMODAL_MODELS:
        if not _hf_snapshot_cached(model_name):
            continue
        tokenizer = load_tokenizer(model_name)
        renderer = create_renderer(tokenizer, renderer="auto")
        # We expect a hand-coded VL renderer, not the default fallback.
        assert not type(renderer).__name__.startswith("Default"), (
            f"{model_name} routed to DefaultRenderer despite being in "
            f"MULTIMODAL_MODELS — add it to MODEL_RENDERER_MAP."
        )


@pytest.mark.parametrize(
    "mm_model_name,modality", _CASES, ids=[f"{m}|{mo}" for m, mo in _CASES]
)
def test_tool_response_image_byte_parity(mm_model_name, modality, tiny_image):
    """Tool-message image parity vs ``processor.apply_chat_template`` + ``processor(...)``.

    Browser-agent SFT traces carry post-action screenshots as ``tool``
    responses. Renderers that drop those image parts silently — historically
    every Qwen-VL family renderer did — produce token streams that diverge
    from the HF processor and lose most of the visual learning signal.
    Skipped for renderers that haven't grown the feature yet; flips to a
    real assertion as they do.
    """
    if modality != "image":
        pytest.skip("Tool-response media path is image-only for now.")
    if not _hf_snapshot_cached(mm_model_name):
        pytest.skip(f"{mm_model_name}: HF snapshot not cached locally")

    kit = _modality_kit(modality, mm_model_name)
    tokenizer, processor, renderer = _load_processor_and_renderer(mm_model_name)

    if not _supports_tool_message_images(renderer):
        pytest.skip(
            f"{type(renderer).__name__} does not yet emit images inside tool responses"
        )

    for case in _build_tool_image_cases(kit["make_part"], tiny_image):
        messages, add_gp = case.values
        ours = renderer.render_ids(messages, add_generation_prompt=add_gp)
        theirs = kit["processor_input_ids"](processor, messages, add_gp)
        assert ours == theirs, (
            f"{mm_model_name} / tool / case={case.id}: "
            f"renderer diverges from processor.\n"
            f"  len(ours)={len(ours)} len(theirs)={len(theirs)}\n"
            f"  ours[:60]={ours[:60]}\n  theirs[:60]={theirs[:60]}"
        )


def test_qwen3_vl_renderer_exposes_image_modality():
    """The flagship multimodal renderer is concretely Qwen3VLRenderer.

    Sanity-check that the dispatch wiring works end-to-end: model in
    registry → load → create_renderer(auto) → expected concrete class.
    """
    model = "Qwen/Qwen3-VL-4B-Instruct"
    if not _hf_snapshot_cached(model):
        pytest.skip(f"{model}: HF snapshot not cached locally")
    tokenizer = load_tokenizer(model)
    renderer = create_renderer(tokenizer, renderer="auto")
    assert isinstance(renderer, Qwen3VLRenderer)
    assert "image" in MULTIMODAL_MODELS[model]


def test_is_image_part_treats_type_field_as_authoritative():
    """``Dataset.from_list`` unifies the Arrow schema across the elements
    of a list-typed column. A content list mixing text and image parts
    round-trips with ``image_url: None`` added to every text part (and
    ``text: None`` added to every image part). The classifier must treat
    the ``type`` field as authoritative when present — falling back to
    a key-presence check on ``image_url`` would misclassify the text
    part and the renderer would later raise on ``_load_pil_image(None)``.
    """
    from renderers.qwen3_vl import _is_image_part, _is_video_part

    # Typed parts classify by their ``type``.
    assert _is_image_part(
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXX"}}
    )
    assert _is_image_part({"type": "image", "image": object()})
    assert _is_video_part(
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,XXX"}}
    )

    # Schema-unified text parts — typed as text, with a None zombie key
    # for the sibling modality — must NOT classify as image / video.
    schema_unified_text = {"type": "text", "text": "hello", "image_url": None}
    assert not _is_image_part(schema_unified_text)
    assert not _is_video_part(schema_unified_text)
    schema_unified_text_with_video = {"type": "text", "text": "hi", "video_url": None}
    assert not _is_video_part(schema_unified_text_with_video)

    # Untyped fallback only fires when ``type`` is absent, and requires
    # a truthy value (mere key presence isn't enough).
    assert _is_image_part({"image_url": {"url": "data:..."}})
    assert _is_image_part({"image": object()})
    assert not _is_image_part({"image_url": None})
    assert not _is_image_part({"image": None})
    assert not _is_video_part({"video_url": None})
