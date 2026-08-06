"""Fast, offline checks for the renderer registration surfaces.

Adding a renderer touches several deliberately explicit public/type surfaces.
These assertions make drift fail in a small unit test instead of at runtime for
one model family or in downstream config deserialization.
"""

from pydantic import TypeAdapter

import renderers
from renderers import RendererConfig
from renderers import base as base_module
from renderers import configs as configs_module


def test_runtime_registry_matches_config_registry():
    base_module._populate_registry()

    runtime_names = set(base_module.RENDERER_REGISTRY)
    config_names = set(configs_module._CONFIG_BY_NAME) - {"auto"}

    assert runtime_names == config_names


def test_renderer_config_union_matches_config_registry():
    schema = TypeAdapter(RendererConfig).json_schema()
    discriminator_names = set(schema["discriminator"]["mapping"])

    assert discriminator_names == set(configs_module._CONFIG_BY_NAME)


def test_lazy_public_renderer_exports_match_runtime_classes():
    base_module._populate_registry()

    runtime_class_names = {
        renderer_class.__name__
        for renderer_class in base_module.RENDERER_REGISTRY.values()
    }

    assert runtime_class_names == set(renderers._LAZY_RENDERERS)
    assert runtime_class_names <= set(renderers.__all__)


def test_model_and_multimodal_maps_only_reference_registered_renderers():
    base_module._populate_registry()
    runtime_names = set(base_module.RENDERER_REGISTRY)

    assert set(base_module.MODEL_RENDERER_MAP.values()) <= runtime_names
    assert set(base_module.MULTIMODAL_MODELS) <= set(base_module.MODEL_RENDERER_MAP)
