try:
    from renderers._version import __version__
except ImportError:
    # Source checkout without a built artifact (e.g. editable install
    # before the first ``uv build`` populates ``_version.py``). Real
    # installs always have it.
    __version__ = "0+unknown"

from renderers.base import (
    Content,
    ContentPart,
    ImagePart,
    MULTIMODAL_MODELS,
    Message,
    MultiModalData,
    MultimodalRenderer,
    ParsedResponse,
    ParsedToolCall,
    PlaceholderRange,
    RenderedConversation,
    RenderedTokens,
    Renderer,
    RendererPool,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolCallFunction,
    ToolCallParseStatus,
    ToolSpec,
    VideoPart,
    attribute_text_segments,
    build_training_sample,
    build_trajectory_step,
    create_renderer,
    create_renderer_pool,
    is_multimodal,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.client import OverlongPromptError
from renderers.configs import (
    AutoRendererConfig,
    BaseRendererConfig,
    config_from_name,
    DefaultRendererConfig,
    DeepSeekV3RendererConfig,
    GLM45RendererConfig,
    GLM51RendererConfig,
    GLM5RendererConfig,
    GptOssRendererConfig,
    KimiK25RendererConfig,
    KimiK2RendererConfig,
    LagunaXS2RendererConfig,
    MiniMaxM2RendererConfig,
    Nemotron3RendererConfig,
    Qwen35RendererConfig,
    Qwen36RendererConfig,
    Qwen3RendererConfig,
    Qwen3VLRendererConfig,
    RendererConfig,
)

# Concrete renderer classes are lazy-loaded so that consumers needing
# only the config layer (``RendererConfig`` discriminated union) don't
# pay the ``transformers`` import cost. Each renderer module does
# ``from transformers.tokenization_utils import PreTrainedTokenizer``
# at module level, so eager imports here would drag ``transformers``
# into every downstream ``import renderers``. ``__getattr__`` (PEP 562)
# resolves the names on first attribute access, so ``from renderers
# import DefaultRenderer`` and ``renderers.DefaultRenderer`` both work
# transparently. ``create_renderer`` doesn't depend on these eager
# imports — ``renderers.base._populate_registry`` lazy-imports the
# concrete classes itself when a renderer is instantiated.
_LAZY_RENDERERS: dict[str, str] = {
    "DeepSeekV3Renderer": "renderers.deepseek_v3",
    "DefaultRenderer": "renderers.default",
    "GLM45Renderer": "renderers.glm45",
    "GLM51Renderer": "renderers.glm5",
    "GLM5Renderer": "renderers.glm5",
    "GptOssRenderer": "renderers.gpt_oss",
    "KimiK25Renderer": "renderers.kimi_k25",
    "KimiK2Renderer": "renderers.kimi_k2",
    "LagunaXS2Renderer": "renderers.laguna_xs2",
    "MiniMaxM2Renderer": "renderers.minimax_m2",
    "Nemotron3Renderer": "renderers.nemotron3",
    "Qwen35Renderer": "renderers.qwen35",
    "Qwen36Renderer": "renderers.qwen36",
    "Qwen3Renderer": "renderers.qwen3",
    "Qwen3VLRenderer": "renderers.qwen3_vl",
}


def __getattr__(name: str):
    if name in _LAZY_RENDERERS:
        import importlib

        module = importlib.import_module(_LAZY_RENDERERS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_RENDERERS))


__all__ = [
    "AutoRendererConfig",
    "BaseRendererConfig",
    "Content",
    "ContentPart",
    "DeepSeekV3Renderer",
    "DeepSeekV3RendererConfig",
    "DefaultRenderer",
    "DefaultRendererConfig",
    "GLM45Renderer",
    "GLM45RendererConfig",
    "GLM51Renderer",
    "GLM51RendererConfig",
    "GLM5Renderer",
    "GLM5RendererConfig",
    "GptOssRenderer",
    "GptOssRendererConfig",
    "ImagePart",
    "KimiK25Renderer",
    "KimiK25RendererConfig",
    "KimiK2Renderer",
    "KimiK2RendererConfig",
    "LagunaXS2Renderer",
    "LagunaXS2RendererConfig",
    "MULTIMODAL_MODELS",
    "Message",
    "MiniMaxM2Renderer",
    "MiniMaxM2RendererConfig",
    "MultiModalData",
    "MultimodalRenderer",
    "Nemotron3Renderer",
    "Nemotron3RendererConfig",
    "OverlongPromptError",
    "ParsedResponse",
    "ParsedToolCall",
    "PlaceholderRange",
    "Qwen35Renderer",
    "Qwen35RendererConfig",
    "Qwen36Renderer",
    "Qwen36RendererConfig",
    "Qwen3Renderer",
    "Qwen3RendererConfig",
    "Qwen3VLRenderer",
    "Qwen3VLRendererConfig",
    "RenderedConversation",
    "RenderedTokens",
    "Renderer",
    "RendererConfig",
    "RendererPool",
    "TextPart",
    "ThinkingPart",
    "ToolCall",
    "ToolCallFunction",
    "ToolCallParseStatus",
    "ToolSpec",
    "VideoPart",
    "__version__",
    "attribute_text_segments",
    "build_training_sample",
    "build_trajectory_step",
    "config_from_name",
    "create_renderer",
    "create_renderer_pool",
    "is_multimodal",
    "reject_assistant_in_extension",
    "trim_to_turn_close",
]
