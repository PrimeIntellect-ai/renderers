"""Completed traces exercise render(), including attribution, against the encoder."""

import json
from pathlib import Path
from typing import get_args

import pytest

from renderers.configs import RendererConfig
from tests.conformance.harness import (
    assert_attribution,
    assert_tokens,
    reference,
    renderer,
    unmodified_generation_reference,
)
from tests.conformance.test_rollouts import matrix as inference_matrix
from tests.parity import MODEL_CATALOG, SCENARIOS, kwarg_combinations, scenario_is_valid


def matrix():
    for case in MODEL_CATALOG:
        for kwargs in kwarg_combinations(case):
            for scenario in SCENARIOS:
                if scenario_is_valid(case, scenario, kwargs):
                    yield pytest.param(
                        case,
                        kwargs,
                        scenario,
                        id=f"{case.model}/{scenario.id}/{kwargs!r}",
                    )


def test_every_renderer_has_a_reference_checkpoint():
    union = get_args(RendererConfig)[0]
    declared = {cls.model_fields["name"].default for cls in get_args(union)}
    # Factory choices are not wire formats.
    assert declared - {"auto", "custom"} == {
        case.resolved_renderer for case in MODEL_CATALOG
    }
    pins = json.loads(Path(__file__).with_name("tokenizers.json").read_text())
    assert set(pins) == {case.model for case in MODEL_CATALOG}
    assert all(len(pin["revision"]) == 40 for pin in pins.values())


@pytest.mark.parametrize("case,kwargs,scenario", tuple(matrix()))
def test_finished_trace(case, kwargs, scenario):
    native = renderer(case, kwargs)
    messages = [dict(message) for message in scenario.messages]
    rendered = native.render(messages, **scenario.render_kwargs)
    expected = reference(case, messages, kwargs, **scenario.render_kwargs)
    assert_tokens(rendered.token_ids, expected, case)
    assert_attribution(rendered)
    assert_tokens(native.render_ids(messages, **scenario.render_kwargs), expected, case)


@pytest.mark.parametrize("case,kwargs", tuple(inference_matrix()))
def test_unmodified_generation_prefill(case, kwargs):
    messages = [{"role": "user", "content": "Test."}]
    assert_tokens(
        renderer(case, kwargs).render_ids(messages, add_generation_prompt=True),
        unmodified_generation_reference(case, messages, kwargs),
        case,
    )
