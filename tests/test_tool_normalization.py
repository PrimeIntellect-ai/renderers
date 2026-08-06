"""Provider tool definitions normalize to one renderer-facing contract."""

from __future__ import annotations

from copy import deepcopy

import pytest
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.responses import FunctionToolParam

from renderers import (
    ToolSpec,
    ToolSpecError,
    UnsupportedToolSpecError,
    normalize_tool_spec,
    normalize_tool_specs,
)
from renderers.parsing import _build_param_type_index, _extract_tool_names


FUNCTION = {
    "name": "get_weather",
    "description": "Return the weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

CANONICAL = {"type": "function", "function": FUNCTION}

TOOL_SHAPES = {
    "flat": FUNCTION,
    "openai-chat": CANONICAL,
    "openai-responses": {"type": "function", **FUNCTION},
    "anthropic": {
        "name": FUNCTION["name"],
        "description": FUNCTION["description"],
        "input_schema": FUNCTION["parameters"],
    },
    "mcp": {
        "name": FUNCTION["name"],
        "description": FUNCTION["description"],
        "inputSchema": FUNCTION["parameters"],
    },
}


@pytest.mark.parametrize(
    "tool",
    TOOL_SHAPES.values(),
    ids=TOOL_SHAPES,
)
def test_function_tool_wire_shapes_normalize_identically(tool):
    assert normalize_tool_spec(tool) == CANONICAL


def test_legacy_tool_spec_remains_constructible():
    assert ToolSpec(**FUNCTION) == FUNCTION


def test_openai_sdk_chat_and_responses_request_types_are_supported():
    chat = ChatCompletionFunctionToolParam(type="function", function=FUNCTION)
    responses = FunctionToolParam(type="function", **FUNCTION)

    assert normalize_tool_spec(chat) == CANONICAL
    assert normalize_tool_spec(responses) == CANONICAL


@pytest.mark.parametrize("tool", TOOL_SHAPES.values(), ids=TOOL_SHAPES)
def test_provider_shapes_feed_schema_aware_parsing(tool):
    assert _build_param_type_index([tool]) == {
        "get_weather": {"city": {"type": "string"}}
    }
    assert _extract_tool_names([tool]) == {"get_weather"}


def test_normalization_preserves_function_options_and_extensions():
    tool = {
        "type": "function",
        "name": "lookup",
        "parameters": None,
        "strict": False,
        "defer_loading": True,
        "allowed_callers": ["programmatic_tool_calling"],
    }

    assert normalize_tool_spec(tool) == {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": None,
            "strict": False,
            "defer_loading": True,
            "allowed_callers": ["programmatic_tool_calling"],
        },
    }


def test_normalization_does_not_retain_or_mutate_caller_data():
    original = deepcopy(CANONICAL)
    normalized = normalize_tool_spec(original)

    normalized["function"]["parameters"]["properties"]["city"]["type"] = "integer"

    assert original == CANONICAL


class _ToolModel:
    def __init__(self):
        self.kwargs = None

    def model_dump(self, **kwargs):
        self.kwargs = kwargs
        return {"type": "function", **FUNCTION, "strict": None}


def test_pydantic_style_tool_objects_are_supported():
    tool = _ToolModel()

    assert normalize_tool_spec(tool) == {
        "type": "function",
        "function": {**FUNCTION, "strict": None},
    }
    assert tool.kwargs == {"mode": "python", "exclude_none": True}


@pytest.mark.parametrize(
    "tool_type",
    [
        "apply_patch",
        "code_interpreter",
        "computer",
        "computer_use_preview",
        "custom",
        "file_search",
        "image_generation",
        "local_shell",
        "mcp",
        "namespace",
        "programmatic_tool_calling",
        "shell",
        "skills",
        "tool_search",
        "web_search",
        "web_search_preview",
    ],
)
def test_non_function_tool_protocols_fail_loudly(tool_type):
    with pytest.raises(UnsupportedToolSpecError, match=repr(tool_type)):
        normalize_tool_spec({"type": tool_type})


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        ({"description": "missing name"}, "name"),
        ({"name": ""}, "name"),
        ({"name": "x", "description": 42}, "description"),
        ({"name": "x", "parameters": []}, "parameters"),
        ({"name": "x", "strict": "yes"}, "strict"),
        ({"name": "x", "defer_loading": 1}, "defer_loading"),
        ({"name": "x", "allowed_callers": "direct"}, "allowed_callers"),
        (
            {"name": "x", "parameters": {}, "input_schema": {"type": "object"}},
            "conflicting",
        ),
    ],
)
def test_invalid_function_tools_fail_at_the_boundary(tool, message):
    with pytest.raises(ToolSpecError, match=message):
        normalize_tool_spec(tool)


def test_collection_errors_identify_the_bad_tool_index():
    with pytest.raises(UnsupportedToolSpecError, match=r"tools\[1\]"):
        normalize_tool_specs([FUNCTION, {"type": "web_search"}])


def test_none_and_empty_collections_remain_distinct():
    assert normalize_tool_specs(None) is None
    assert normalize_tool_specs([]) == []
