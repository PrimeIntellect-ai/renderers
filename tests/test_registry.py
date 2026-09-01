"""Structural tests for the declarative renderer registry."""

from __future__ import annotations

import importlib
from typing import get_args

import renderers
import renderers.base as base
from parity import MODEL_CATALOG
from renderers.configs import (
    AutoRendererConfig,
    RendererConfig,
    _CONFIG_BY_NAME,
)
from renderers.registry import MODEL_PARITY_REPRESENTATIVES, RENDERER_SPECS


def test_runtime_metadata_is_derived_from_renderer_specs():
    expected_routes = {
        model_id: renderer.name
        for renderer in RENDERER_SPECS
        for model in renderer.models
        for model_id in model.model_ids
    }
    expected_modalities = {
        model_id: set(model.modalities)
        for renderer in RENDERER_SPECS
        for model in renderer.models
        if model.modalities
        for model_id in model.model_ids
    }
    expected_lazy_renderers = {
        renderer.renderer_class: renderer.module for renderer in RENDERER_SPECS
    }

    assert base.MODEL_RENDERER_MAP == expected_routes
    assert base.MULTIMODAL_MODELS == expected_modalities
    assert renderers._LAZY_RENDERERS == expected_lazy_renderers


def test_every_mapped_model_has_a_parity_representative():
    catalog_models = {case.model for case in MODEL_CATALOG}

    assert MODEL_PARITY_REPRESENTATIVES.keys() == base.MODEL_RENDERER_MAP.keys()
    for model_id, representative in MODEL_PARITY_REPRESENTATIVES.items():
        assert representative in catalog_models, (
            f"{model_id!r} routes through {representative!r}, but that "
            "representative is absent from MODEL_CATALOG"
        )
        assert (
            base.MODEL_RENDERER_MAP[model_id] == base.MODEL_RENDERER_MAP[representative]
        )


def test_config_registry_and_discriminated_union_cover_manifest():
    expected_configs = {
        spec.name: getattr(
            importlib.import_module("renderers.configs"), spec.config_class
        )
        for spec in RENDERER_SPECS
    }
    assert _CONFIG_BY_NAME == {"auto": AutoRendererConfig, **expected_configs}

    config_union = get_args(RendererConfig)[0]
    assert set(get_args(config_union)) == {
        AutoRendererConfig,
        *expected_configs.values(),
    }


def test_every_renderer_loader_resolves(monkeypatch):
    monkeypatch.setattr(base, "RENDERER_REGISTRY", {})
    base._populate_registry()

    assert base.RENDERER_REGISTRY.keys() == {spec.name for spec in RENDERER_SPECS}
    for spec in RENDERER_SPECS:
        expected_class = getattr(
            importlib.import_module(spec.module), spec.renderer_class
        )
        assert base.RENDERER_REGISTRY[spec.name] is expected_class
