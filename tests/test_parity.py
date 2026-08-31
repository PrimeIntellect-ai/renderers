"""Unified renderer parity matrix.

Every valid cell is the product of one model, one shared conversation
scenario, and all explicit reference-controlled values accepted by that
model's typed config. Unsupported cells are excluded declaratively in the
model catalog rather than discovered at runtime through skips or xfails. The
reference is model-aware: most models use Hugging Face Jinja, DeepSeek V4 uses
its shipped Python encoder, and GPT-OSS uses Harmony.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import Annotated, Any, Literal, Mapping, Union, cast, get_args, get_origin

import pytest

from parity import (
    KWARG_VALUES,
    MODEL_CATALOG,
    SCENARIOS,
    ModelCase,
    Scenario,
    kwarg_combinations,
    scenario_is_valid,
)
from renderers import create_renderer
from renderers.base import MODEL_RENDERER_MAP, load_tokenizer
from renderers.configs import RendererConfig, _config_class_for
from tests.reference_rendering import (
    DEFAULT_REFERENCE_ORACLE,
    REFERENCE_ORACLES,
    RENDERER_ORACLE_ROUTES,
    reference_oracle_for_model,
    reference_oracle_for_renderer,
    render_reference,
)


def _id(case: ModelCase, scenario: Scenario, kwargs: Mapping[str, Any]) -> str:
    values = ",".join(f"{key}={value!r}" for key, value in kwargs.items())
    suffix = values or "defaults"
    return f"{case.model}-{scenario.id}-{suffix}"


def _matrix():
    for case in MODEL_CATALOG:
        for kwargs in kwarg_combinations(case):
            for scenario in SCENARIOS:
                if scenario_is_valid(case, scenario, kwargs):
                    yield pytest.param(
                        case,
                        scenario,
                        kwargs,
                        id=_id(case, scenario, kwargs),
                    )


@lru_cache(maxsize=None)
def _tokenizer(model: str):
    return load_tokenizer(model)


@lru_cache(maxsize=None)
def _renderer(model: str, renderer_name: str, items: tuple[tuple[str, Any], ...]):
    tokenizer = _tokenizer(model)
    resolved = (
        MODEL_RENDERER_MAP.get(model, "default")
        if renderer_name == "auto"
        else renderer_name
    )
    config = cast(RendererConfig, _config_class_for(resolved)())
    kwargs = dict(items)
    return create_renderer(tokenizer, config, chat_template_kwargs=kwargs or None)


def _required_annotation_values(annotation: Any) -> tuple[Any, ...]:
    """Return every finite value explicitly declared by a field annotation.

    Open domains such as ``str`` and ``float`` intentionally contribute no
    values; ``KWARG_VALUES`` still supplies representative samples for them.
    Finite branches (``Literal``, ``Enum``, ``bool``, and ``None``) are
    exhaustive, even when nested inside ``Annotated`` or a union with an open
    domain.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _required_annotation_values(get_args(annotation)[0])
    if origin is Literal:
        return get_args(annotation)
    if annotation is bool:
        return (True, False)
    if annotation is type(None):
        return (None,)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(member.value for member in annotation)
    if origin in {Union, UnionType}:
        return tuple(
            value
            for member in get_args(annotation)
            for value in _required_annotation_values(member)
        )
    return ()


def test_catalog_covers_every_declared_kwarg():
    declared = {
        field
        for case in MODEL_CATALOG
        for field in _config_class_for(case.resolved_renderer).template_field_names()
    }
    assert declared <= KWARG_VALUES.keys()


def test_catalog_covers_every_finite_declared_kwarg_value():
    missing = []
    config_classes = {
        _config_class_for(case.resolved_renderer) for case in MODEL_CATALOG
    }
    for config_cls in sorted(config_classes, key=lambda cls: cls.__name__):
        for field in sorted(config_cls.template_field_names()):
            annotation = config_cls.model_fields[field].annotation
            absent = tuple(
                value
                for value in _required_annotation_values(annotation)
                if value not in KWARG_VALUES[field]
            )
            if absent:
                missing.append(f"{config_cls.__name__}.{field}: {absent!r}")

    assert not missing, "KWARG_VALUES omits declared finite values:\n" + "\n".join(
        missing
    )


def test_oracle_routes_resolve_to_registered_adapters():
    assert DEFAULT_REFERENCE_ORACLE in REFERENCE_ORACLES
    assert set(RENDERER_ORACLE_ROUTES.values()) <= REFERENCE_ORACLES.keys()
    assert all(name == oracle.name for name, oracle in REFERENCE_ORACLES.items())
    for renderer_name in RENDERER_ORACLE_ROUTES:
        _config_class_for(renderer_name)
    for case in MODEL_CATALOG:
        assert case.oracle == reference_oracle_for_renderer(case.resolved_renderer)


@pytest.mark.parametrize(
    ("model", "oracle"),
    (
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-v4"),
        ("openai/gpt-oss-20b", "harmony"),
        ("openai/gpt-oss-120b", "harmony"),
        ("unmapped/example", DEFAULT_REFERENCE_ORACLE),
    ),
)
def test_model_oracle_routing(model: str, oracle: str):
    assert reference_oracle_for_model(model) == oracle


def test_catalog_routes_every_auto_model_to_its_declared_renderer():
    for case in MODEL_CATALOG:
        if case.renderer == "auto":
            assert case.model in MODEL_RENDERER_MAP


@pytest.mark.parametrize(
    "model",
    (
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-Flash-Next",
    ),
)
def test_preserved_disabled_thinking_history_remains_in_parity_matrix(model: str):
    case = next(case for case in MODEL_CATALOG if case.model == model)
    scenario = next(scenario for scenario in SCENARIOS if scenario.id == "multi-turn")

    assert scenario_is_valid(
        case,
        scenario,
        {"enable_thinking": False, "preserve_thinking": True},
    )
    assert not scenario_is_valid(
        case,
        scenario,
        {"enable_thinking": False, "preserve_thinking": False},
    )


@pytest.mark.parametrize("case,scenario,kwargs", tuple(_matrix()))
def test_renderer_matches_reference(
    case: ModelCase,
    scenario: Scenario,
    kwargs: Mapping[str, Any],
):
    tokenizer = _tokenizer(case.model)
    renderer = _renderer(case.model, case.renderer, tuple(kwargs.items()))
    for key, value in kwargs.items():
        if value is not None:
            assert getattr(renderer.config, key) == value

    oracle_kwargs = dict(case.oracle_defaults)
    oracle_kwargs.update(kwargs)
    # Config ``None`` values mean "defer to the oracle default"; omission
    # expresses the same state for every registered adapter.
    oracle_kwargs = {
        key: value for key, value in oracle_kwargs.items() if value is not None
    }
    expected = render_reference(
        tokenizer,
        [dict(message) for message in scenario.messages],
        oracle=case.oracle,
        **oracle_kwargs,
        **scenario.render_kwargs,
    )
    got = renderer.render_ids(list(scenario.messages), **scenario.render_kwargs)
    assert got == expected, (
        f"{case.model} / {scenario.id} / {dict(kwargs)!r}: renderer diverged "
        f"from {case.oracle} oracle (got {len(got)} tokens, expected "
        f"{len(expected)})"
    )
