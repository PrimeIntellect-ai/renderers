"""Custom renderers loaded from local files or importable modules."""

from __future__ import annotations

import pickle
import textwrap

import pytest
from pydantic import TypeAdapter, ValidationError

from renderers import (
    CustomRendererConfig,
    GLM53RendererConfig,
    RendererConfig,
    create_renderer,
    custom_renderer_config,
    load_custom_renderer,
    merge_chat_template_kwargs,
    template_field_names,
)

TOY_SOURCE = textwrap.dedent(
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
def toy_file(tmp_path):
    path = tmp_path / "toy_renderer.py"
    path.write_text(TOY_SOURCE)
    return path


def import_path(path, name="ToyRenderer"):
    return f"{path}:{name}"


def test_custom_fields_validate_through_the_renderer_config(toy_file):
    config = CustomRendererConfig(import_path=import_path(toy_file), effort=3)
    assert custom_renderer_config(config).effort == 3
    assert custom_renderer_config(config).name == "toy"
    with pytest.raises(ValidationError):
        CustomRendererConfig(import_path=import_path(toy_file), unknown=1)


def test_shared_renderer_fields_reach_the_renderer_config(toy_file):
    config = CustomRendererConfig(
        import_path=import_path(toy_file), thinking_retention="all"
    )
    assert custom_renderer_config(config).thinking_retention == "all"


def test_discriminated_union_parses_renderer_configs(toy_file):
    config = TypeAdapter(RendererConfig).validate_python(
        {"name": "custom", "import_path": import_path(toy_file), "effort": 2}
    )
    assert isinstance(config, CustomRendererConfig)
    assert custom_renderer_config(config).effort == 2


def test_create_renderer_builds_the_custom_renderer_with_its_own_config(toy_file):
    tokenizer = object()
    renderer = create_renderer(
        tokenizer, CustomRendererConfig(import_path=import_path(toy_file), effort=4)
    )
    assert type(renderer).__name__ == "ToyRenderer"
    assert renderer.tokenizer is tokenizer
    assert type(renderer.config).__name__ == "ToyRendererConfig"
    assert renderer.config.effort == 4


def test_chat_template_kwargs_use_the_custom_renderer_allowlist(toy_file):
    config = CustomRendererConfig(import_path=import_path(toy_file))
    renderer = create_renderer(object(), config, chat_template_kwargs={"effort": 5})
    assert renderer.config.effort == 5
    with pytest.raises(ValueError, match="cache_size"):
        create_renderer(object(), config, chat_template_kwargs={"cache_size": 2})


def test_module_import_paths_import_by_name(toy_file, monkeypatch):
    monkeypatch.syspath_prepend(str(toy_file.parent))
    config = CustomRendererConfig(import_path="toy_renderer.ToyRenderer", effort=6)
    assert create_renderer(object(), config).config.effort == 6


def test_file_import_paths_load_once_per_process(toy_file):
    assert load_custom_renderer(import_path(toy_file)) is load_custom_renderer(
        import_path(toy_file)
    )


def test_invalid_import_paths_fail_clearly(toy_file, tmp_path):
    with pytest.raises(ValueError, match="must look like"):
        CustomRendererConfig(import_path=str(toy_file))
    with pytest.raises(ValueError, match="has no attribute"):
        CustomRendererConfig(import_path=import_path(toy_file, "Missing"))
    with pytest.raises(TypeError, match="config_class"):
        CustomRendererConfig(import_path=import_path(toy_file, "NoConfigRenderer"))
    with pytest.raises(FileNotFoundError):
        CustomRendererConfig(import_path=f"{tmp_path / 'missing.py'}:ToyRenderer")


def test_renderer_configs_pickle_and_key_caches(toy_file):
    low = CustomRendererConfig(import_path=import_path(toy_file), effort=1)
    high = CustomRendererConfig(import_path=import_path(toy_file), effort=9)
    assert pickle.loads(pickle.dumps(high)) == high
    cache = {low: "low", high: "high"}
    assert (
        cache[CustomRendererConfig(import_path=import_path(toy_file), effort=9)]
        == "high"
    )
    assert low != high


def test_public_helpers_resolve_custom_template_fields(toy_file):
    config = CustomRendererConfig(import_path=import_path(toy_file))
    assert template_field_names(config) == frozenset({"effort"})
    merged = merge_chat_template_kwargs(config, {"effort": 7})
    assert isinstance(merged, CustomRendererConfig)
    assert custom_renderer_config(merged).effort == 7
    with pytest.raises(ValueError, match="cache_size"):
        merge_chat_template_kwargs(config, {"cache_size": 1})


GLM53_NO_THINKING_SOURCE = textwrap.dedent(
    """
    from typing import Literal

    from renderers.configs import GLM53RendererConfig
    from renderers.glm5 import GLM53Renderer


    class NoThinkingGLM53Config(GLM53RendererConfig):
        name: Literal["glm-5.3-no-thinking"] = "glm-5.3-no-thinking"


    class NoThinkingGLM53Renderer(GLM53Renderer):
        config_class = NoThinkingGLM53Config

        def _emit_generation_prompt(self, emit_special) -> None:
            emit_special(self._assistant, -1, is_sampled=False, is_content=False)
            emit_special(self._think, -1, is_sampled=False, is_content=False)
            emit_special(self._think_end, -1, is_sampled=False, is_content=False)
    """
)


def test_readme_example_prefills_the_think_block(tmp_path):
    from renderers.base import load_tokenizer

    path = tmp_path / "glm53_no_thinking.py"
    path.write_text(GLM53_NO_THINKING_SOURCE)
    tokenizer = load_tokenizer("zai-org/GLM-5.3-BF16")
    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    custom = create_renderer(
        tokenizer,
        CustomRendererConfig(
            import_path=import_path(path, "NoThinkingGLM53Renderer"),
            clear_thinking=True,
        ),
    )
    builtin = create_renderer(
        tokenizer, GLM53RendererConfig(enable_thinking=False, clear_thinking=True)
    )

    ids = custom.render_ids(messages, add_generation_prompt=True)
    assert tokenizer.decode(ids[-3:]) == "<|assistant|><think></think>"
    assert ids == builtin.render_ids(messages, add_generation_prompt=True)
    assert custom.config.clear_thinking is True
