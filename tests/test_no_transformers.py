"""Regression tests for the lightweight install boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap


_PREAMBLE = r"""
import importlib.abc
import sys


class BlockTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "transformers" or fullname.startswith("transformers."):
            raise ImportError("blocked optional dependency")
        return None


sys.meta_path.insert(0, BlockTransformers())


class FakeTokenizer:
    name_or_path = "Qwen/Qwen3-8B"
    unk_token_id = 0
    eos_token_id = 1

    def __init__(self):
        specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|message_system|>",
            "<|message_tool|>",
            "<|content_text|>",
            "<|content_thinking|>",
            "<|content_image|>",
            "<|content_audio_input|>",
            "<|content_xml|>",
            "<|content_invoke_tool_json|>",
            "<|content_invoke_tool_text|>",
            "<|content_model_end_sampling|>",
            "<|end_message|>",
            "<|audio_end|>",
            "<|unused_200054|>",
            "<|unused_200053|>",
        ]
        self.specials = {token: index + 100 for index, token in enumerate(specials)}
        self.reverse = {value: key for key, value in self.specials.items()}
        self.eos_token_id = self.specials["<|im_end|>"]

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token, self.unk_token_id)

    def encode(self, text, **kwargs):
        return [200_000 + ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(
            self.reverse[token_id]
            if token_id in self.reverse
            else chr(token_id - 200_000)
            for token_id in token_ids
        )

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(str(message.get("content") or "") for message in messages)
        return self.encode(text)
"""


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE + textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def test_all_renderer_modules_import_without_transformers():
    result = _run(
        """
        import renderers

        for name in renderers._LAZY_RENDERERS:
            getattr(renderers, name)

        assert not any(
            name == "transformers" or name.startswith("transformers.")
            for name in sys.modules
        )
        """
    )
    _assert_ok(result)


def test_byo_text_tokenizer_create_render_parse_without_transformers():
    result = _run(
        """
        from renderers import Qwen3Renderer, create_renderer

        tokenizer = FakeTokenizer()
        renderer = create_renderer(tokenizer)
        assert isinstance(renderer, Qwen3Renderer)

        prompt = renderer.render_ids(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
        )
        assert prompt

        parsed = renderer.parse_response(
            tokenizer.encode("world") + [tokenizer.eos_token_id]
        )
        assert parsed.content == "world"
        """
    )
    _assert_ok(result)


def test_text_only_inkling_works_without_transformers_or_processor():
    result = _run(
        """
        from renderers import InklingRenderer, create_renderer

        tokenizer = FakeTokenizer()
        tokenizer.name_or_path = "thinkingmachines/Inkling"
        renderer = create_renderer(tokenizer)
        assert isinstance(renderer, InklingRenderer)
        assert renderer._processor is None

        rendered = renderer.render(
            [{"role": "user", "content": "text only"}],
            add_generation_prompt=True,
        )
        assert rendered.token_ids
        assert rendered.multi_modal_data is None
        assert renderer._processor is None
        assert not any(
            name == "transformers" or name.startswith("transformers.")
            for name in sys.modules
        )
        """
    )
    _assert_ok(result)


def test_load_tokenizer_points_to_optional_extra():
    result = _run(
        """
        from renderers.base import load_tokenizer

        try:
            load_tokenizer("Qwen/Qwen3-8B")
        except ImportError as exc:
            assert "renderers[transformers]" in str(exc), str(exc)
        else:
            raise AssertionError("load_tokenizer unexpectedly succeeded")
        """
    )
    _assert_ok(result)


def test_unknown_auto_resolution_requires_extra_or_explicit_config():
    result = _run(
        """
        from renderers import DefaultRenderer, DefaultRendererConfig, create_renderer

        tokenizer = FakeTokenizer()
        tokenizer.name_or_path = "example/unknown-text-model"

        try:
            create_renderer(tokenizer)
        except ImportError as exc:
            assert "DefaultRendererConfig" in str(exc), str(exc)
        else:
            raise AssertionError("unknown auto-resolution unexpectedly succeeded")

        renderer = create_renderer(tokenizer, DefaultRendererConfig())
        assert isinstance(renderer, DefaultRenderer)
        """
    )
    _assert_ok(result)


def test_multimodal_lazy_paths_point_to_optional_extra():
    result = _run(
        """
        from renderers import MultiModalData, Qwen3VLRenderer
        from renderers.client import _build_qwen_vl_features

        renderer = Qwen3VLRenderer(FakeTokenizer())
        for operation in (
            renderer._get_processor,
            lambda: _build_qwen_vl_features(
                MultiModalData(), spatial_merge_size=2
            ),
        ):
            try:
                operation()
            except ImportError as exc:
                assert "renderers[transformers]" in str(exc), str(exc)
            else:
                raise AssertionError("multimodal operation unexpectedly succeeded")
        """
    )
    _assert_ok(result)
