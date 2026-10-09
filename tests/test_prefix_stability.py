"""Declared full-render stability must agree with token prefixes."""

from functools import lru_cache

import pytest
from parity import TOOLS, models_for
from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import _config_class_for


_STABLE_DEFAULTS = {
    "prime-qwen3",
    "qwen3-vl",
    "qwen3.8",
    "glm-5.3",
    "kimi-k2",
    "laguna-xs.2",
    "laguna-m.1",
    "laguna-xs-2.1",
    "laguna-s-2.1",
    "llama-3",
    "deepseek-v3",
    "deepseek-r1",
}
_VARIANTS = {
    "qwen3": [
        ({"enable_thinking": False}, False),
        ({"thinking_retention": "all"}, False),
    ],
    "qwen3.5": [({"enable_thinking": False}, False)],
    "qwen3.6": [({"preserve_thinking": True}, True)],
    "qwen3.8": [({"preserve_thinking": False}, False)],
    "glm-5": [({"clear_thinking": False}, True), ({"enable_thinking": False}, False)],
    "glm-5.1": [
        ({"clear_thinking": False}, False),
        ({"clear_thinking": False, "enable_thinking": False}, True),
    ],
    "glm-5.3": [({"clear_thinking": True}, False)],
    "gpt-oss": [({"auto_drop_analysis": False}, False)],
    "gemma4": [
        ({"preserve_thinking": True}, False),
        ({"enable_thinking": False}, False),
    ],
    "hy3": [
        ({"preserved_thinking": True}, True),
        ({"is_training": True}, True),
        ({"preserved_thinking": True, "raw_last_assistant": True}, False),
    ],
    "kimi-k2.5": [({"thinking": False}, False)],
    "deepseek-v4": [
        ({"enable_thinking": True}, False),
        ({"enable_thinking": True, "drop_thinking": False}, False),
    ],
    "nemotron-3": [
        ({"truncate_history_thinking": False}, True),
        ({"enable_thinking": False}, False),
    ],
    "nemotron-3-ultra": [
        ({"truncate_history_thinking": False}, True),
        ({"truncate_history_thinking": False, "medium_effort": True}, False),
    ],
    "nemotron-3.5": [({"truncate_history_thinking": False}, True)],
}


def _cases():
    for case in models_for("shared"):
        name = case.resolved_renderer
        variants = [({}, name in _STABLE_DEFAULTS), *_VARIANTS.get(name, [])]
        if "Nemotron-3-Super" in case.model:
            variants.append(
                ({"truncate_history_thinking": False, "low_effort": True}, False)
            )
        for kwargs, stable in variants:
            yield pytest.param(
                case.model, name, kwargs, stable, id=f"{case.model}-{kwargs}"
            )


@lru_cache(maxsize=None)
def _tokenizer(model):
    return load_tokenizer(model)


def _conversations(name=None):
    if name == "deepseek-v4":
        for task in ("query", "action"):
            yield (
                [
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Answer.", "task": task},
                    {"role": "user", "content": "Follow up?"},
                ],
                None,
            )
        yield (
            [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "a", "wo_eos": True},
                {"role": "assistant", "content": "b"},
            ],
            None,
        )
    if name == "inkling":
        yield (
            [
                {"role": "user", "content": "First?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_0", "function": {"name": "first", "arguments": {}}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "Result."},
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Second?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "function": {"name": "second", "arguments": {}},
                        }
                    ],
                },
            ],
            None,
        )

    for reasoning in ("First reasoning.", ""):
        assistant = {
            "role": "assistant",
            "content": "First answer.",
            "reasoning_content": reasoning,
        }
        yield (
            [
                {"role": "user", "content": "First question?"},
                assistant,
                {"role": "user", "content": "Second question?"},
                {
                    "role": "assistant",
                    "content": "Second answer.",
                    "reasoning_content": "Second reasoning.",
                },
            ],
            None,
        )
        yield (
            [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    **assistant,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_weather",
                    "tool_call_id": "call_0",
                    "content": "Sunny.",
                },
                {
                    "role": "assistant",
                    "content": "It is sunny.",
                    "reasoning_content": "Summarize.",
                },
                {"role": "user", "content": "Thanks. What next?"},
                {
                    "role": "assistant",
                    "content": "Go outside.",
                    "reasoning_content": "Suggest an activity.",
                },
            ],
            TOOLS,
        )


@pytest.mark.parametrize("model, name, kwargs, stable", _cases())
def test_prefix_stability_matches_rendering(model, name, kwargs, stable):
    config_cls = _config_class_for(name)
    template_kwargs = {
        k: v for k, v in kwargs.items() if k in config_cls.template_field_names()
    }
    internal_kwargs = {k: v for k, v in kwargs.items() if k not in template_kwargs}
    renderer = create_renderer(
        _tokenizer(model),
        config_cls(**internal_kwargs),
        chat_template_kwargs=template_kwargs,
    )
    assert renderer.is_prefix_stable is stable
    if name == "default":
        return  # Opaque means unknown, even if these particular inputs are stable.
    changed = False
    for messages, tools in _conversations(name):
        for end, message in enumerate(messages[:-1], 1):
            if message["role"] != "assistant":
                continue
            before = renderer.render_ids(messages[:end], tools=tools)
            for extended_end in range(end + 1, len(messages) + 1):
                after = renderer.render_ids(messages[:extended_end], tools=tools)
                extends = after[: len(before)] == before
                if stable:
                    assert extends, (model, kwargs, end, extended_end)
                changed |= not extends
    if not stable:
        assert changed, (
            "Exercise a counterexample for every known-unstable configuration"
        )


def test_stability_uses_auto_resolved_template_kwargs():
    tokenizer = _tokenizer("Qwen/Qwen3.8-27B")
    assert create_renderer(tokenizer).is_prefix_stable is True
    assert (
        create_renderer(
            tokenizer, chat_template_kwargs={"preserve_thinking": False}
        ).is_prefix_stable
        is False
    )


# The shared parity catalog enumerates every template field. Include bridge
# overrides and internal Harmony switches as well: neither may accidentally
# turn a full-render history transformation into a stable declaration.
def _all_configs():
    from itertools import product

    from parity import MODEL_CATALOG, kwarg_combinations
    from pydantic import ValidationError

    for case in MODEL_CATALOG:
        config_cls = _config_class_for(case.resolved_renderer)
        internal = [
            {"thinking_retention": retention}
            for retention in (None, "tool_cycle", "all")
        ]
        if case.resolved_renderer == "default":
            internal = [{"thinking_retention": None}]
        if case.resolved_renderer == "gpt-oss":
            internal = [
                {**override, "use_system_prompt": system, "auto_drop_analysis": drop}
                for override, system, drop in product(
                    internal, (False, True), (False, True)
                )
            ]
        for template, override in product(kwarg_combinations(case), internal):
            try:
                config = config_cls(**template, **override)
            except ValidationError:
                continue  # Explicitly conflicting template/bridge policies are invalid.
            yield pytest.param(
                case,
                config,
                id=f"{case.model}-{config.model_dump(exclude_defaults=True)}",
            )


def _prefix_pairs(messages):
    for end, message in enumerate(messages, 1):
        if message["role"] != "assistant":
            continue
        before = messages[:end]
        for extended_end in range(end + 1, len(messages) + 1):
            yield before, messages[:extended_end]
        # Consecutive assistant messages are accepted by these renderers.
        yield before, before + [{"role": "assistant", "content": "Continuation."}]
        # Exercise the assistant that originally ended the scenario, too.
        yield (
            before,
            before
            + [
                {"role": "user", "content": "Follow up?"},
                {
                    "role": "assistant",
                    "content": "Answer.",
                    "reasoning_content": "Reason.",
                },
            ],
        )


def _assert_prefix(renderer, before, after, tools, label):
    prefix = renderer.render_ids(before, tools=tools)
    extended = renderer.render_ids(after, tools=tools)
    assert extended[: len(prefix)] == prefix, (label, before, after)


@pytest.mark.parametrize("case, config", _all_configs())
def test_all_configurations_against_shared_corpus(case, config):
    from parity import SCENARIOS

    renderer = create_renderer(_tokenizer(case.model), config)
    assert type(renderer.is_prefix_stable) is bool
    if not renderer.is_prefix_stable:
        if case.resolved_renderer != "default":
            # Require a real witness for known-unstable built-ins under every
            # config cell, including bridge overrides. Jinja remains unknown.
            changed = False
            for messages, tools in _conversations(case.resolved_renderer):
                for before, after in _prefix_pairs(messages):
                    prefix = renderer.render_ids(before, tools=tools)
                    extended = renderer.render_ids(after, tools=tools)
                    changed |= extended[: len(prefix)] != prefix
            assert changed, (case.model, config)
        return
    for scenario in SCENARIOS:
        if scenario.id in case.excluded_scenarios:
            continue
        if (
            scenario.only_renderers
            and case.resolved_renderer not in scenario.only_renderers
        ):
            continue
        messages = [dict(message) for message in scenario.messages]
        for before, after in _prefix_pairs(messages):
            _assert_prefix(
                renderer, before, after, list(scenario.tools) or None, scenario.id
            )

    # Alternate reasoning representations, BPE-sensitive suffixes, and fixed
    # empty/nonempty system preambles are independent of reference parity.
    for content in (
        "",
        " ",
        "a\n",
        "<think>Reason.</think>Answer.",
        [{"type": "text", "text": "Structured answer."}],
        [
            {"type": "thinking", "thinking": "Reason."},
            {"type": "text", "text": "Answer."},
        ],
    ):
        if isinstance(content, list) and case.resolved_renderer == "prime-qwen3":
            continue  # Prime-Qwen3 only accepts string content.
        if (
            isinstance(content, list)
            and any(part["type"] == "thinking" for part in content)
            and case.resolved_renderer
            in {
                "prime-qwen3",
                "qwen3.5",
                "qwen3.6",
                "inkling",
                "nemotron-3",
                "nemotron-3-ultra",
                "nemotron-3.5",
                "llama-3",
            }
        ):
            # These grammars reject structured thinking parts.
            continue
        for system in (
            [],
            [{"role": "system", "content": ""}],
            [{"role": "system", "content": "Be concise."}],
        ):
            before = system + [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": content},
            ]
            for tools in (None, TOOLS):
                _assert_prefix(
                    renderer,
                    before,
                    before + [{"role": "user", "content": "Next?"}],
                    tools,
                    "content/preamble",
                )


@pytest.fixture(scope="module")
def tiny_images():
    from PIL import Image

    return [
        Image.new("RGB", size, color=color)
        for size, color in [((28, 28), "red"), ((56, 28), "blue")]
    ]


class _ImageProcessor:
    """Deterministic processor output; test renderer expansion, not model preprocessing."""

    merge_size = 2

    def __call__(self, *, images, return_tensors):
        import numpy as np

        assert return_tensors == "np"
        width, height = images[0].size
        return {
            "image_grid_thw": np.array([[1, height // 14, width // 14]]),
            "pixel_values": np.zeros((1, 3)),
            "num_tokens": [width // 14],
            "num_patches": [1],
            "imgs_sizes": [(height, width)],
        }


def _media_configs():
    for parameter in _all_configs():
        case, config = parameter.values
        if case.resolved_renderer in {"qwen3-vl", "qwen3.6", "qwen3.8", "nemotron-3.5"}:
            yield parameter


@pytest.mark.parametrize("case, config", _media_configs())
def test_image_prefixes_across_configurations(case, config, tiny_images):
    from types import SimpleNamespace

    # A small cache forces repeated renders through both cache hits and eviction.
    if "image_cache_max" in type(config).model_fields:
        config = config.model_copy(update={"image_cache_max": 1})
    tokenizer = _tokenizer(case.model)
    if case.resolved_renderer == "nemotron-3.5":
        # The public Lightning tokenizer is text-only. Exercise the VLM path
        # with the same BYO vocabulary fixture as the dedicated VLM tests.
        from tests.test_nemotron35_vlm import _Tokenizer

        tokenizer = _Tokenizer()
    renderer = create_renderer(tokenizer, config)
    if not renderer.is_prefix_stable:
        return
    renderer._processor = SimpleNamespace(image_processor=_ImageProcessor())
    messages = []
    for image in tiny_images:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "An image.",
                    "reasoning_content": "Inspect pixels.",
                },
            ]
        )
    prefix = renderer.render(messages[:2])
    for tools in (None, TOOLS):
        _assert_prefix(renderer, messages[:2], messages, tools, "image expansion")
    assert renderer.render(messages[:2]).token_ids == prefix.token_ids
    assert prefix.multi_modal_data is not None
    assert prefix.multi_modal_data.mm_placeholders["image"][0].length > 0


def test_matrix_covers_every_builtin_renderer():
    from parity import MODEL_CATALOG
    from renderers.base import RENDERER_REGISTRY, _populate_registry

    _populate_registry()
    assert {case.resolved_renderer for case in MODEL_CATALOG} == set(RENDERER_REGISTRY)


@pytest.mark.parametrize(
    "model, name",
    [
        ("Qwen/Qwen3-VL-4B-Instruct", "qwen3-vl"),
        ("Qwen/Qwen3.6-35B-A3B", "qwen3.6"),
        ("Qwen/Qwen3.8-27B", "qwen3.8"),
        ("Qwen/Qwen3.8-Flash-Next", "qwen3.8"),
    ],
)
def test_image_prefixes_with_checkpoint_processor(model, name, tiny_images):
    from transformers import AutoProcessor

    config_cls = _config_class_for(name)
    kwargs = {"add_vision_id": True, "image_cache_max": 1}
    if "preserve_thinking" in config_cls.model_fields:
        kwargs["preserve_thinking"] = True
    renderer = create_renderer(_tokenizer(model), config_cls(**kwargs))
    renderer._processor = AutoProcessor.from_pretrained(model)
    assert renderer.is_prefix_stable
    messages = []
    for image in tiny_images:
        messages.extend(
            [
                {"role": "user", "content": [{"type": "image", "image": image}]},
                {
                    "role": "assistant",
                    "content": "Description.",
                    "reasoning_content": "Inspect.",
                },
            ]
        )
    for tools in (None, TOOLS):
        _assert_prefix(
            renderer, messages[:2], messages, tools, "checkpoint image processor"
        )


def test_matrix_enumerates_finite_template_domains():
    from types import UnionType
    from typing import Literal, Union, get_args, get_origin

    from parity import KWARG_VALUES, MODEL_CATALOG

    def finite_values(annotation):
        if annotation is bool:
            return {False, True}
        if annotation is type(None):
            return {None}
        if get_origin(annotation) is Literal:
            return set(get_args(annotation))
        if get_origin(annotation) in (Union, UnionType):
            domains = [finite_values(part) for part in get_args(annotation)]
            if all(domain is not None for domain in domains):
                return set().union(*domains)
        return None  # Strings/numbers have unbounded domains, sampled in the catalog.

    for case in MODEL_CATALOG:
        config_cls = _config_class_for(case.resolved_renderer)
        for field in config_cls.template_field_names():
            domain = finite_values(config_cls.model_fields[field].annotation)
            if domain is not None:
                assert domain <= set(KWARG_VALUES[field]), (
                    case.resolved_renderer,
                    field,
                )
