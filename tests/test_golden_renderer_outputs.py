"""Golden contract for each registered renderer's public behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.golden_corpus import GOLDEN_CASES, build_golden_case


GOLDEN_PATH = Path(__file__).with_name("golden_renderer_outputs.json")


@pytest.fixture(scope="module")
def expected_cases():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.slug)
def test_renderer_behavior_matches_golden(case, expected_cases):
    assert build_golden_case(case) == expected_cases[case.slug]
