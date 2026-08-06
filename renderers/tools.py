"""Tool-definition types and normalization.

Renderers only know how to describe client-executed, JSON-schema function
tools in a model prompt.  Provider APIs expose that same concept through
several wire shapes, so normalize those shapes once before model-specific
formatting begins.

The canonical representation deliberately matches the OpenAI Chat
Completions envelope.  Existing renderer templates already consume that
shape, which lets callers use other provider shapes without changing the
rendered token stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, cast


class _OptionalFunctionFields(TypedDict, total=False):
    description: str
    parameters: dict[str, Any] | None
    strict: bool | None
    defer_loading: bool
    allowed_callers: list[str]


class FunctionSpec(_OptionalFunctionFields):
    """Canonical body of a client-executed JSON-schema function tool."""

    name: str


class ChatCompletionToolSpec(TypedDict):
    """OpenAI Chat Completions function-tool envelope."""

    type: Literal["function"]
    function: FunctionSpec


class ResponsesFunctionToolSpec(_OptionalFunctionFields):
    """OpenAI Responses flat function-tool definition."""

    type: Literal["function"]
    name: str


class _OptionalDescription(TypedDict, total=False):
    description: str


class _OptionalAnthropicFields(_OptionalDescription, total=False):
    strict: bool
    defer_loading: bool
    input_examples: list[dict[str, Any]]
    cache_control: dict[str, Any]


class AnthropicToolSpec(_OptionalAnthropicFields):
    """Anthropic Messages function-tool definition."""

    name: str
    input_schema: dict[str, Any]


class _OptionalMCPFields(_OptionalDescription, total=False):
    title: str
    outputSchema: dict[str, Any]
    annotations: dict[str, Any]
    _meta: dict[str, Any]


class MCPToolSpec(_OptionalMCPFields):
    """Model Context Protocol function-tool definition."""

    name: str
    inputSchema: dict[str, Any]


class ToolSpec(_OptionalFunctionFields, total=False):
    """Provider-agnostic tool mapping accepted by renderer entry points.

    This remains a constructible ``TypedDict`` for compatibility with the
    original flat tool contract, while also describing the keys used by Chat
    Completions, Responses, Anthropic, and MCP function tools.
    """

    type: str
    function: FunctionSpec
    name: str
    input_schema: dict[str, Any]
    inputSchema: dict[str, Any]
    input_examples: list[dict[str, Any]]
    cache_control: dict[str, Any]
    title: str
    outputSchema: dict[str, Any]
    annotations: dict[str, Any]
    _meta: dict[str, Any]


class ToolSpecModel(Protocol):
    """Pydantic-style object that can expose a tool definition mapping."""

    def model_dump(self, **kwargs: Any) -> Mapping[str, Any]: ...


KnownToolSpec: TypeAlias = (
    FunctionSpec
    | ChatCompletionToolSpec
    | ResponsesFunctionToolSpec
    | AnthropicToolSpec
    | MCPToolSpec
)
ToolSpecInput: TypeAlias = ToolSpec | Mapping[str, Any] | ToolSpecModel
CanonicalToolSpec: TypeAlias = ChatCompletionToolSpec


class ToolSpecError(ValueError):
    """A tool definition cannot be normalized safely."""


class UnsupportedToolSpecError(ToolSpecError):
    """The tool requires a protocol the renderer cannot serialize."""


def _as_mapping(tool: ToolSpecInput) -> Mapping[str, Any]:
    if isinstance(tool, Mapping):
        return cast(Mapping[str, Any], tool)

    model_dump = getattr(tool, "model_dump", None)
    if not callable(model_dump):
        raise ToolSpecError("tool definitions must be mappings or expose model_dump()")
    try:
        dumped = model_dump(mode="python", exclude_none=True)
    except TypeError:
        dumped = model_dump()
    if not isinstance(dumped, Mapping):
        raise ToolSpecError("tool model_dump() must return a mapping")
    return dumped


def _function_body(raw_tool: Mapping[str, Any]) -> dict[str, Any]:
    tool_type = raw_tool.get("type")
    nested = raw_tool.get("function")

    if nested is not None:
        if tool_type not in (None, "function"):
            raise ToolSpecError(
                f"a nested function definition cannot use tool type {tool_type!r}"
            )
        if not isinstance(nested, Mapping):
            raise ToolSpecError("tool.function must be a mapping")
        function = deepcopy(dict(nested))
    else:
        if tool_type not in (None, "function"):
            raise UnsupportedToolSpecError(
                f"tool type {tool_type!r} is not a client-executed JSON-schema "
                "function tool; hosted, custom-text, and namespace tools need "
                "a model-specific execution protocol"
            )
        function = deepcopy(dict(raw_tool))
        function.pop("type", None)

    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ToolSpecError("function tool name must be a non-empty string")

    description = function.get("description")
    if description is None:
        function.pop("description", None)
    elif not isinstance(description, str):
        raise ToolSpecError("function tool description must be a string")

    schema_aliases = [
        key for key in ("parameters", "input_schema", "inputSchema") if key in function
    ]
    if len(schema_aliases) > 1:
        first = function[schema_aliases[0]]
        if any(function[key] != first for key in schema_aliases[1:]):
            raise ToolSpecError(
                "function tool provides conflicting parameter schemas: "
                + ", ".join(schema_aliases)
            )
    if schema_aliases:
        schema = function[schema_aliases[0]]
        if schema is not None and not isinstance(schema, Mapping):
            raise ToolSpecError("function tool parameters must be a mapping or None")
        for key in ("parameters", "input_schema", "inputSchema"):
            function.pop(key, None)
        function["parameters"] = deepcopy(dict(schema)) if schema is not None else None

    strict = function.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ToolSpecError("function tool strict must be bool or None")

    defer_loading = function.get("defer_loading")
    if defer_loading is not None and not isinstance(defer_loading, bool):
        raise ToolSpecError("function tool defer_loading must be bool")

    allowed_callers = function.get("allowed_callers")
    if allowed_callers is not None and (
        not isinstance(allowed_callers, list)
        or not all(isinstance(caller, str) for caller in allowed_callers)
    ):
        raise ToolSpecError("function tool allowed_callers must be a list of strings")

    return function


def normalize_tool_spec(tool: ToolSpecInput) -> CanonicalToolSpec:
    """Return a detached Chat-style envelope for one function tool.

    Supported inputs are the legacy/verifiers flat shape, OpenAI Chat
    Completions and Responses function tools, Anthropic ``input_schema``
    tools, MCP ``inputSchema`` tools, and Pydantic-style objects containing
    any of those mappings.
    """

    function = cast(FunctionSpec, _function_body(_as_mapping(tool)))
    return {"type": "function", "function": function}


def normalize_tool_specs(
    tools: Iterable[ToolSpecInput] | None,
) -> list[ToolSpec] | None:
    """Normalize a tool collection without retaining caller-owned objects."""

    if tools is None:
        return None

    normalized: list[ToolSpec] = []
    for index, tool in enumerate(tools):
        try:
            normalized.append(cast(ToolSpec, normalize_tool_spec(tool)))
        except ToolSpecError as exc:
            raise type(exc)(f"tools[{index}]: {exc}") from exc
    return normalized
