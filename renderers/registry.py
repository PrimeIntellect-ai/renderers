"""Declarative source of truth for renderer registration and model routing.

Each :class:`RendererSpec` ties together the renderer/config pair and the
canonical model IDs that auto-resolve to it.  A :class:`ModelSpec` names the
checkpoint used by the parity suite and any routing-equivalent aliases.  The
runtime registry, config lookup, lazy public imports, model routing, and
multimodal catalog are all derived from this manifest.

Import paths are stored as data to keep this module dependency-free.  That is
what lets ``renderers.base``, ``renderers.configs``, and ``renderers.__init__``
all consume the same manifest without introducing import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A parity-tested model and checkpoints that intentionally share its route.

    ``model`` is the representative checkpoint exercised by the parity suite.
    ``aliases`` inherit its renderer and modalities without multiplying the
    full Cartesian parity matrix for byte-compatible or routing-equivalent
    checkpoints.
    """

    model: str
    aliases: tuple[str, ...] = ()
    modalities: frozenset[str] = frozenset()

    @property
    def model_ids(self) -> tuple[str, ...]:
        return (self.model, *self.aliases)


@dataclass(frozen=True)
class RendererSpec:
    """All registration metadata for one concrete renderer."""

    name: str
    module: str
    renderer_class: str
    config_class: str
    models: tuple[ModelSpec, ...] = ()


def _model(
    model: str,
    *,
    aliases: tuple[str, ...] = (),
    modalities: frozenset[str] = frozenset(),
) -> ModelSpec:
    return ModelSpec(model=model, aliases=aliases, modalities=modalities)


IMAGE = frozenset({"image"})
IMAGE_AUDIO = frozenset({"image", "audio"})


RENDERER_SPECS = (
    RendererSpec(
        "default",
        "renderers.default",
        "DefaultRenderer",
        "DefaultRendererConfig",
    ),
    RendererSpec(
        "qwen3",
        "renderers.qwen3",
        "Qwen3Renderer",
        "Qwen3RendererConfig",
        (
            # These checkpoints share the same Qwen3 chat-template grammar.
            _model(
                "Qwen/Qwen3-8B",
                aliases=(
                    "Qwen/Qwen3-0.6B",
                    "Qwen/Qwen3-1.7B",
                    "Qwen/Qwen3-4B",
                    "Qwen/Qwen3-4B-Instruct-2507",
                    "Qwen/Qwen3-4B-Thinking-2507",
                    "Qwen/Qwen3-14B",
                    "Qwen/Qwen3-32B",
                    "Qwen/Qwen3-30B-A3B",
                    "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "Qwen/Qwen3-30B-A3B-Thinking-2507",
                    "Qwen/Qwen3-235B-A22B",
                ),
            ),
        ),
    ),
    RendererSpec(
        "prime-qwen3",
        "renderers.prime_qwen3",
        "PrimeQwen3Renderer",
        "PrimeQwen3RendererConfig",
        (
            _model("PrimeIntellect/Qwen3-0.6B"),
            _model("PrimeIntellect/Qwen3-1.7B"),
        ),
    ),
    RendererSpec(
        "qwen3.5",
        "renderers.qwen35",
        "Qwen35Renderer",
        "Qwen35RendererConfig",
        tuple(
            _model(model, modalities=IMAGE)
            for model in (
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-4B",
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3.5-35B-A3B",
                "Qwen/Qwen3.5-122B-A10B",
                "Qwen/Qwen3.5-397B-A17B",
            )
        ),
    ),
    RendererSpec(
        "qwen3.6",
        "renderers.qwen36",
        "Qwen36Renderer",
        "Qwen36RendererConfig",
        (_model("Qwen/Qwen3.6-35B-A3B", modalities=IMAGE),),
    ),
    RendererSpec(
        "qwen3.8",
        "renderers.qwen38",
        "Qwen38Renderer",
        "Qwen38RendererConfig",
        (
            _model("Qwen/Qwen3.8-27B", modalities=IMAGE),
            _model("Qwen/Qwen3.8-Flash-Next", modalities=IMAGE),
        ),
    ),
    RendererSpec(
        "qwen3-vl",
        "renderers.qwen3_vl",
        "Qwen3VLRenderer",
        "Qwen3VLRendererConfig",
        (
            _model(
                "Qwen/Qwen3-VL-4B-Instruct",
                aliases=(
                    "Qwen/Qwen3-VL-8B-Instruct",
                    "Qwen/Qwen3-VL-30B-A3B-Instruct",
                ),
                modalities=IMAGE,
            ),
        ),
    ),
    RendererSpec(
        "gemma4",
        "renderers.gemma4",
        "Gemma4Renderer",
        "Gemma4RendererConfig",
        tuple(
            _model(model, modalities=IMAGE)
            for model in (
                "google/gemma-4-E2B-it",
                "google/gemma-4-E4B-it",
                "google/gemma-4-26B-A4B-it",
                "google/gemma-4-31B-it",
            )
        ),
    ),
    RendererSpec(
        "glm-5",
        "renderers.glm5",
        "GLM5Renderer",
        "GLM5RendererConfig",
        (
            _model("zai-org/GLM-5", aliases=("zai-org/GLM-5-FP8",)),
            _model("zai-org/GLM-4.7-Flash"),
        ),
    ),
    RendererSpec(
        "glm-5.1",
        "renderers.glm5",
        "GLM51Renderer",
        "GLM51RendererConfig",
        (_model("zai-org/GLM-5.1"),),
    ),
    RendererSpec(
        "glm-4.5",
        "renderers.glm45",
        "GLM45Renderer",
        "GLM45RendererConfig",
        (
            _model(
                "THUDM/GLM-4.5-Air",
                aliases=("zai-org/GLM-4.5-Air",),
            ),
        ),
    ),
    RendererSpec(
        "minimax-m2",
        "renderers.minimax_m2",
        "MiniMaxM2Renderer",
        "MiniMaxM2RendererConfig",
        (
            _model(
                "MiniMaxAI/MiniMax-M2.5",
                aliases=("MiniMaxAI/MiniMax-M2",),
            ),
        ),
    ),
    RendererSpec(
        "deepseek-v3",
        "renderers.deepseek_v3",
        "DeepSeekV3Renderer",
        "DeepSeekV3RendererConfig",
        (
            _model(
                "deepseek-ai/DeepSeek-V3",
                aliases=("deepseek-ai/DeepSeek-V3-Base",),
            ),
        ),
    ),
    RendererSpec(
        "deepseek-r1",
        "renderers.deepseek_r1",
        "DeepSeekR1Renderer",
        "DeepSeekR1RendererConfig",
        (
            _model(
                "deepseek-ai/DeepSeek-R1",
                aliases=("deepseek-ai/DeepSeek-R1-0528",),
            ),
        ),
    ),
    RendererSpec(
        "deepseek-v4",
        "renderers.deepseek_v4",
        "DeepSeekV4Renderer",
        "DeepSeekV4RendererConfig",
        (_model("deepseek-ai/DeepSeek-V4-Flash-0731"),),
    ),
    RendererSpec(
        "kimi-k2",
        "renderers.kimi_k2",
        "KimiK2Renderer",
        "KimiK2RendererConfig",
        (_model("moonshotai/Kimi-K2-Instruct"),),
    ),
    RendererSpec(
        "kimi-k2.5",
        "renderers.kimi_k25",
        "KimiK25Renderer",
        "KimiK25RendererConfig",
        (
            _model("moonshotai/Kimi-K2.5", modalities=IMAGE),
            _model("moonshotai/Kimi-K2.6", modalities=IMAGE),
        ),
    ),
    RendererSpec(
        "nemotron-3",
        "renderers.nemotron3",
        "Nemotron3Renderer",
        "Nemotron3RendererConfig",
        (
            _model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
            _model("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
        ),
    ),
    RendererSpec(
        "nemotron-3-ultra",
        "renderers.nemotron3",
        "Nemotron3UltraRenderer",
        "Nemotron3UltraRendererConfig",
        (
            _model(
                "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
                aliases=("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",),
            ),
        ),
    ),
    RendererSpec(
        "nemotron-3.5",
        "renderers.nemotron3",
        "Nemotron35Renderer",
        "Nemotron35RendererConfig",
        (_model("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"),),
    ),
    RendererSpec(
        "llama-3",
        "renderers.llama_3",
        "Llama3Renderer",
        "Llama3RendererConfig",
        (
            _model("meta-llama/Llama-3.2-1B-Instruct"),
            _model("meta-llama/Llama-3.2-3B-Instruct"),
        ),
    ),
    RendererSpec(
        "laguna-xs.2",
        "renderers.laguna_xs2",
        "LagunaXS2Renderer",
        "LagunaXS2RendererConfig",
        (_model("poolside/Laguna-XS.2"),),
    ),
    RendererSpec(
        "laguna-m.1",
        "renderers.laguna_xs2",
        "LagunaM1Renderer",
        "LagunaM1RendererConfig",
        (_model("poolside/Laguna-M.1"),),
    ),
    RendererSpec(
        "laguna-xs-2.1",
        "renderers.laguna_xs2",
        "LagunaXS21Renderer",
        "LagunaXS21RendererConfig",
        (_model("poolside/Laguna-XS-2.1"),),
    ),
    RendererSpec(
        "laguna-s-2.1",
        "renderers.laguna_s21",
        "LagunaS21Renderer",
        "LagunaS21RendererConfig",
        (_model("poolside/Laguna-S-2.1"),),
    ),
    RendererSpec(
        "gpt-oss",
        "renderers.gpt_oss",
        "GptOssRenderer",
        "GptOssRendererConfig",
        (
            _model(
                "openai/gpt-oss-20b",
                aliases=("openai/gpt-oss-120b",),
            ),
        ),
    ),
    RendererSpec(
        "inkling",
        "renderers.inkling",
        "InklingRenderer",
        "InklingRendererConfig",
        (
            _model("thinkingmachines/Inkling", modalities=IMAGE_AUDIO),
            _model("thinkingmachines/Inkling-Small", modalities=IMAGE_AUDIO),
        ),
    ),
    RendererSpec(
        "hy3",
        "renderers.hy3",
        "Hy3Renderer",
        "Hy3RendererConfig",
        (
            _model(
                "tencent/Hy3",
                aliases=("tencent/Hy3-FP8",),
            ),
        ),
    ),
)


def _validate_specs() -> None:
    renderer_names: set[str] = set()
    renderer_classes: set[str] = set()
    model_ids: set[str] = set()

    for renderer in RENDERER_SPECS:
        if renderer.name in renderer_names:
            raise ValueError(f"Duplicate renderer name: {renderer.name!r}")
        renderer_names.add(renderer.name)

        if renderer.renderer_class in renderer_classes:
            raise ValueError(
                f"Renderer class registered more than once: {renderer.renderer_class!r}"
            )
        renderer_classes.add(renderer.renderer_class)

        for model in renderer.models:
            if not model.modalities <= {"image", "video", "audio"}:
                raise ValueError(
                    f"Unknown modalities for {model.model!r}: {model.modalities!r}"
                )
            for model_id in model.model_ids:
                if model_id in model_ids:
                    raise ValueError(
                        f"Model ID registered more than once: {model_id!r}"
                    )
                model_ids.add(model_id)


_validate_specs()


RENDERER_SPEC_BY_NAME = {spec.name: spec for spec in RENDERER_SPECS}

# Every mapped ID points to the representative that supplies its parity claim.
MODEL_PARITY_REPRESENTATIVES = {
    model_id: model.model
    for renderer in RENDERER_SPECS
    for model in renderer.models
    for model_id in model.model_ids
}


__all__ = [
    "MODEL_PARITY_REPRESENTATIVES",
    "ModelSpec",
    "RENDERER_SPEC_BY_NAME",
    "RENDERER_SPECS",
    "RendererSpec",
]
