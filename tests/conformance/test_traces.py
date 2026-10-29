"""Reference coverage and the exact supported generation profiles."""

from typing import get_args

import pytest

from renderers.configs import RendererConfig
from tests.conformance.harness import (
    assert_tokens,
    reference,
    renderer,
)
from tests.conformance.test_rollouts import matrix as inference_matrix
from tests.model_assets import MODEL_REVISIONS
from tests.parity import MODEL_CATALOG

pytestmark = pytest.mark.model


def test_every_renderer_has_a_reference_checkpoint():
    union = get_args(RendererConfig)[0]
    declared = {cls.model_fields["name"].default for cls in get_args(union)}
    # Factory choices are not wire formats.
    assert declared - {"auto", "custom"} == {
        case.resolved_renderer for case in MODEL_CATALOG
    }
    assert {case.model for case in MODEL_CATALOG} <= MODEL_REVISIONS.keys()


@pytest.mark.parametrize("case,kwargs", tuple(inference_matrix()))
def test_generation_prefill_matches_reference_profile(case, kwargs):
    messages = [{"role": "user", "content": "Test."}]
    assert_tokens(
        renderer(case, kwargs).render_ids(messages, add_generation_prompt=True),
        reference(case, messages, kwargs, add_generation_prompt=True),
        case,
    )
