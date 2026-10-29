"""Release real tokenizer assets between conformance modules."""

import gc

import pytest

from tests.conformance.harness import tokenizer


@pytest.fixture(scope="module", autouse=True)
def release_reference_assets():
    yield
    tokenizer.cache_clear()
    gc.collect()
