"""Renderer plugins loaded from local files or importable modules."""

from __future__ import annotations

import pickle
import textwrap

import pytest
from pydantic import TypeAdapter, ValidationError

from renderers import (
    PluginRendererConfig,
    RendererConfig,
    create_renderer,
    load_plugin_renderer,
)

PLUGIN_SOURCE = textwrap.dedent(
    """
    from typing import Literal

    from renderers.configs import BaseRendererConfig


    class ToyRendererConfig(BaseRendererConfig):
        name: Literal["toy"] = "toy"
        effort: int = 1
        cache_size: int = 8
        _template_fields = frozenset({"effort"})
        _internal_fields = frozenset({"cache_size"})


    class ToyRenderer:
        config_class = ToyRendererConfig

        def __init__(self, tokenizer, config):
            self.tokenizer = tokenizer
            self.config = config


    class NoConfigRenderer:
        pass
    """
)


@pytest.fixture
def plugin_file(tmp_path):
    path = tmp_path / "toy_renderer.py"
    path.write_text(PLUGIN_SOURCE)
    return path


def target(path, name="ToyRenderer"):
    return f"{path}:{name}"


def test_plugin_fields_validate_through_the_plugin_config(plugin_file):
    config = PluginRendererConfig(target=target(plugin_file), effort=3)
    assert config.plugin_config.effort == 3
    assert config.plugin_config.name == "toy"
    with pytest.raises(ValidationError):
        PluginRendererConfig(target=target(plugin_file), unknown=1)


def test_shared_renderer_fields_reach_the_plugin_config(plugin_file):
    config = PluginRendererConfig(target=target(plugin_file), thinking_retention="all")
    assert config.plugin_config.thinking_retention == "all"


def test_discriminated_union_parses_plugin_configs(plugin_file):
    config = TypeAdapter(RendererConfig).validate_python(
        {"name": "plugin", "target": target(plugin_file), "effort": 2}
    )
    assert isinstance(config, PluginRendererConfig)
    assert config.plugin_config.effort == 2


def test_create_renderer_builds_the_plugin_with_its_own_config(plugin_file):
    tokenizer = object()
    renderer = create_renderer(
        tokenizer, PluginRendererConfig(target=target(plugin_file), effort=4)
    )
    assert type(renderer).__name__ == "ToyRenderer"
    assert renderer.tokenizer is tokenizer
    assert type(renderer.config).__name__ == "ToyRendererConfig"
    assert renderer.config.effort == 4


def test_chat_template_kwargs_use_the_plugin_allowlist(plugin_file):
    config = PluginRendererConfig(target=target(plugin_file))
    renderer = create_renderer(object(), config, chat_template_kwargs={"effort": 5})
    assert renderer.config.effort == 5
    with pytest.raises(ValueError, match="cache_size"):
        create_renderer(object(), config, chat_template_kwargs={"cache_size": 2})


def test_module_targets_import_by_name(plugin_file, monkeypatch):
    monkeypatch.syspath_prepend(str(plugin_file.parent))
    config = PluginRendererConfig(target="toy_renderer:ToyRenderer", effort=6)
    assert create_renderer(object(), config).config.effort == 6


def test_file_targets_load_once_per_process(plugin_file):
    assert load_plugin_renderer(target(plugin_file)) is load_plugin_renderer(
        target(plugin_file)
    )


def test_invalid_targets_fail_clearly(plugin_file, tmp_path):
    with pytest.raises(ValueError, match="must look like"):
        PluginRendererConfig(target=str(plugin_file))
    with pytest.raises(ValueError, match="has no attribute"):
        PluginRendererConfig(target=target(plugin_file, "Missing"))
    with pytest.raises(TypeError, match="config_class"):
        PluginRendererConfig(target=target(plugin_file, "NoConfigRenderer"))
    with pytest.raises(FileNotFoundError):
        PluginRendererConfig(target=f"{tmp_path / 'missing.py'}:ToyRenderer")


def test_plugin_configs_pickle_and_key_caches(plugin_file):
    low = PluginRendererConfig(target=target(plugin_file), effort=1)
    high = PluginRendererConfig(target=target(plugin_file), effort=9)
    assert pickle.loads(pickle.dumps(high)) == high
    cache = {low: "low", high: "high"}
    assert cache[PluginRendererConfig(target=target(plugin_file), effort=9)] == "high"
    assert low != high
