"""All renderers treat function-compatible provider shapes identically."""

from __future__ import annotations

from functools import lru_cache

import pytest

from tests.golden_corpus import GOLDEN_CASES, SYSTEM_AND_USER, GoldenCase, _renderer_for
from tests.model_assets import load_test_tokenizer
from tests.test_tool_normalization import TOOL_SHAPES


pytestmark = [pytest.mark.network, pytest.mark.model_parity]


@lru_cache(maxsize=None)
def _load_case(case: GoldenCase):
    tokenizer = load_test_tokenizer(case.model_name)
    return _renderer_for(case, tokenizer)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize("shape", TOOL_SHAPES)
def test_provider_tool_shapes_render_identically(case, shape):
    renderer = _load_case(case)
    expected = renderer.render(
        SYSTEM_AND_USER,
        tools=[TOOL_SHAPES["openai-chat"]],
        add_generation_prompt=True,
    )

    actual = renderer.render(
        SYSTEM_AND_USER,
        tools=[TOOL_SHAPES[shape]],
        add_generation_prompt=True,
    )

    assert actual == expected
