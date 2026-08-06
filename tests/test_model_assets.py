"""Offline validation of immutable model revisions used by parity tests."""

import re

from renderers.base import TRUSTED_REVISIONS
from tests.model_assets import MODEL_REVISIONS, model_revision


def test_model_revisions_are_full_shas():
    sha_re = re.compile(r"^[0-9a-f]{40}$")
    for model_name, revision in MODEL_REVISIONS.items():
        assert sha_re.fullmatch(revision), (
            f"{model_name}: test revision {revision!r} is not a full commit SHA"
        )


def test_remote_code_revisions_match_production_security_policy():
    for model_name, revision in TRUSTED_REVISIONS.items():
        assert model_revision(model_name) == revision


def test_unknown_test_model_fails_loudly():
    try:
        model_revision("unreviewed/example")
    except KeyError as exc:
        assert "No immutable test revision" in str(exc)
    else:
        raise AssertionError("unreviewed model unexpectedly resolved")
