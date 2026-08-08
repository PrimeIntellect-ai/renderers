"""Tests for the Transformers-free tokenizer path."""

from __future__ import annotations

import builtins
import subprocess
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from renderers import DefaultRendererConfig, create_renderer
from renderers.base import attribute_text_segments, load_tokenizer
from renderers.tokenizer import TokenizersTokenizer


def _write_word_tokenizer(path):
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))


def test_tokenizers_backend_adapts_ids_decode_and_offsets(tmp_path):
    _write_word_tokenizer(tmp_path)

    tokenizer = load_tokenizer(str(tmp_path), backend="tokenizers")

    assert isinstance(tokenizer, TokenizersTokenizer)
    assert tokenizer.name_or_path == str(tmp_path)
    assert tokenizer.unk_token_id == 0
    assert tokenizer.convert_tokens_to_ids("hello") == 1
    assert tokenizer.convert_tokens_to_ids("missing") is None
    assert tokenizer.encode("hello world", add_special_tokens=False) == [1, 2]
    assert tokenizer.decode([1, 2], skip_special_tokens=False) == "hello world"
    assert tokenizer(
        "hello world",
        add_special_tokens=False,
        return_offsets_mapping=True,
    ) == {
        "input_ids": [1, 2],
        "offset_mapping": [(0, 5), (6, 11)],
    }


def test_tokenizers_backend_drives_segment_attribution(tmp_path):
    _write_word_tokenizer(tmp_path)
    tokenizer = load_tokenizer(str(tmp_path), backend="tokenizers")

    assert attribute_text_segments(
        tokenizer,
        [("hello ", False), ("world", True)],
    ) == [(1, False), (2, True)]


def test_auto_backend_uses_tokenizers_when_transformers_is_absent(
    tmp_path, monkeypatch
):
    _write_word_tokenizer(tmp_path)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ModuleNotFoundError("blocked for test", name="transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    tokenizer = load_tokenizer(str(tmp_path))
    assert isinstance(tokenizer, TokenizersTokenizer)


def test_default_renderer_rejects_tokenizer_without_chat_templates(tmp_path):
    _write_word_tokenizer(tmp_path)
    tokenizer = load_tokenizer(str(tmp_path), backend="tokenizers")

    with pytest.raises(TypeError, match=r"apply_chat_template.*renderers\[hf\]"):
        create_renderer(tokenizer, DefaultRendererConfig())


def test_qwen_transformers_and_tokenizers_ids_and_offsets_match():
    hf = load_tokenizer("Qwen/Qwen3-0.6B", backend="transformers")
    rust = load_tokenizer("Qwen/Qwen3-0.6B", backend="tokenizers")

    samples = [
        "hello world",
        "hello 👋🏽 café 中文",
        '<|im_start|>assistant\n{"name":"f","arguments":{}}',
    ]
    for sample in samples:
        assert hf.encode(sample, add_special_tokens=False) == rust.encode(
            sample,
            add_special_tokens=False,
        )
        hf_encoding = hf(
            sample,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        rust_encoding = rust(
            sample,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        assert list(hf_encoding["input_ids"]) == rust_encoding["input_ids"]
        assert list(hf_encoding["offset_mapping"]) == rust_encoding["offset_mapping"]


def test_every_renderer_module_imports_without_transformers():
    script = """
import builtins

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "transformers" or name.startswith("transformers."):
        raise ModuleNotFoundError("blocked for core-install smoke test", name="transformers")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import renderers
for renderer_name in renderers._LAZY_RENDERERS:
    getattr(renderers, renderer_name)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
