"""DeepSeek-V4.1-Flash parity against the checkpoint's shipped Python encoder."""

from __future__ import annotations

import importlib.util
from functools import lru_cache

import pytest
from huggingface_hub import hf_hub_download
from pydantic import TypeAdapter, ValidationError
from renderers import (
    DeepSeekV41Renderer,
    DeepSeekV41RendererConfig,
    RendererConfig,
    create_renderer,
)
from renderers.base import load_tokenizer
from renderers.configs import config_from_name

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

CONVERSATIONS = {
    "user": [{"role": "user", "content": "What is 2+2?"}],
    "system_user": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    "multi_turn": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
        {"role": "user", "content": "2+2?"},
    ],
    "system_multi_turn_ending_in_assistant": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add"},
    ],
}


@lru_cache(maxsize=1)
def _tokenizer():
    return load_tokenizer(MODEL)


@lru_cache(maxsize=1)
def _reference_encoder():
    path = hf_hub_download(MODEL, "encoding/encoding.py")
    spec = importlib.util.spec_from_file_location("deepseek_v41_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode(renderer, messages, **kwargs):
    return _tokenizer().decode(
        renderer.render_ids(messages, **kwargs), skip_special_tokens=False
    )


def test_registration_and_native_defaults():
    # Selected by name only: the shared parity suites need tool support for auto-mapping.
    renderer = create_renderer(_tokenizer(), config_from_name("deepseek-v4.1"))

    assert isinstance(renderer, DeepSeekV41Renderer)
    assert renderer.config.enable_thinking is True
    assert renderer.config.reasoning_effort == "high"


def test_config_discriminator_and_effort_validation():
    parsed = TypeAdapter(RendererConfig).validate_python(
        {"name": "deepseek-v4.1", "reasoning_effort": 30}
    )
    assert isinstance(parsed, DeepSeekV41RendererConfig)
    assert parsed.reasoning_effort == 30

    with pytest.raises(ValidationError):
        DeepSeekV41RendererConfig(reasoning_effort=101)


@pytest.mark.parametrize("name", CONVERSATIONS)
@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max", 30])
def test_matches_reference_encoder(name, enable_thinking, reasoning_effort):
    messages = CONVERSATIONS[name]
    add_generation_prompt = messages[-1]["role"] != "assistant"
    renderer = DeepSeekV41Renderer(
        _tokenizer(),
        DeepSeekV41RendererConfig(
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
        ),
    )

    expected = _reference_encoder().encode_messages(
        [dict(message) for message in messages],
        thinking_mode="thinking" if enable_thinking else "chat",
        reasoning_effort=reasoning_effort,
    )

    assert (
        _decode(renderer, messages, add_generation_prompt=add_generation_prompt)
        == expected
    )


def test_inline_reasoning_moves_into_think_block():
    renderer = DeepSeekV41Renderer(_tokenizer())
    inline = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "<think>add</think>4"},
    ]
    split = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add"},
    ]

    assert renderer.render_ids(inline) == renderer.render_ids(split)


def test_rejects_tools_and_mid_conversation_system():
    renderer = DeepSeekV41Renderer(_tokenizer())
    user = [{"role": "user", "content": "Hi"}]
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}

    with pytest.raises(NotImplementedError):
        renderer.render(user, tools=[tool])
    with pytest.raises(NotImplementedError):
        renderer.render([*user, {"role": "system", "content": "Be terse."}])
