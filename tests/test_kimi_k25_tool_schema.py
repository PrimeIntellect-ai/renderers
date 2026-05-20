"""Kimi K2.5 tool-schema TypeScript rendering: bool/null literal correctness.

Regression for PR #55: ``_function_to_typescript`` used to leak Python
literals (``True`` / ``False`` / ``None``) into output advertised as
TypeScript. Both call sites — ``_EnumType.to_typescript_style`` (for
non-string enum members) and ``_TypedParam.to_typescript_style`` (for
bool defaults) — now route non-string values through ``json.dumps`` so
booleans and ``None`` render as ``true`` / ``false`` / ``null``.
"""

from __future__ import annotations

from renderers.kimi_k25 import _function_to_typescript


def test_bool_default_renders_as_json_literal():
    """Bool defaults must render as ``true`` / ``false``, not ``True`` / ``False``."""
    out = _function_to_typescript(
        {
            "name": "set_logging",
            "parameters": {
                "type": "object",
                "properties": {
                    "verbose": {"type": "boolean", "default": True},
                    "quiet": {"type": "boolean", "default": False},
                },
            },
        }
    )
    assert "// Default: true" in out
    assert "// Default: false" in out
    assert "Default: True" not in out
    assert "Default: False" not in out


def test_enum_non_string_members_render_as_json_literals():
    """``null`` / bool / int enum members must render as JSON literals, not Python ``repr``."""
    out = _function_to_typescript(
        {
            "name": "set_mode",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"enum": ["auto", None, True, False, 0]},
                },
            },
        }
    )
    assert '"auto" | null | true | false | 0' in out
    # Python literals must not leak through.
    assert "None" not in out
    assert "True" not in out
    assert "False" not in out
