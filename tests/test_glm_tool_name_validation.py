"""GLM tool-name validation against the provided tools list.

vLLM 0.24 re-aliased ``glm45`` from the lenient ``Glm4MoeModelToolParser``
to ``Glm47MoeModelToolParser``, whose engine config sets
``validate_tool_names=True`` (``vllm/parser/glm47_moe.py``): a completed
``<tool_call>`` block whose name isn't in the request's tool list is
silently dropped — no tool call is emitted and the response reads as a
voluntary stop. Before this, unknown-name calls were surfaced to the
harness, which answered with a recoverable "unknown tool" error.

``parse_glm`` mirrors the ≥ 0.24 behavior when ``tools`` is passed:
unknown names get ``status=UNKNOWN_TOOL`` — never ``OK`` — so the
client-side stop→tool_calls finish-reason promotion
(``renderers/client.py``) agrees with what an OpenAI chat-completions
client sees from the engine, while the attempt itself stays visible for
verifier / RL-loss code. Without ``tools`` there is no validation, also
matching vLLM (``ParserEngine._is_valid_tool_name`` returns ``True``
when the request has no tools).

Cross-validated end-to-end against the real vLLM v0.23.0 and v0.24.0
parsers (vendored from the release tags) on identical completions.
"""

from __future__ import annotations

from functools import lru_cache

from renderers.base import ToolCallParseStatus

# Both GLM renderers share ``parse_glm`` and are served by the same
# strict vLLM parser (the ``glm45`` and ``glm47`` aliases both resolve
# to ``Glm47MoeModelToolParser`` in vLLM ≥ 0.24).
_MODELS = [
    ("THUDM/GLM-4.5-Air", "auto"),
    ("zai-org/GLM-5", "auto"),
]

_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
    # A declared tool with no parameters must still count as a known
    # name (name lookup is independent of the param-type index).
    {"name": "submit", "description": "Finish the episode."},
]


@lru_cache(maxsize=None)
def _load(model: str, renderer_name: str):
    from renderers import config_from_name, create_renderer
    from renderers.base import load_tokenizer

    tok = load_tokenizer(model)
    return tok, create_renderer(tok, config_from_name(renderer_name))


def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        metafunc.parametrize(
            "model,renderer_name",
            _MODELS,
            ids=[m for m, _ in _MODELS],
        )


def _parse(model: str, renderer_name: str, text: str, tools):
    tok, renderer = _load(model, renderer_name)
    ids = tok.encode(text, add_special_tokens=False)
    return renderer.parse_response(ids, tools=tools)


def _statuses(parsed):
    return [tc.status for tc in parsed.tool_calls]


def test_known_name_is_ok(model, renderer_name):
    parsed = _parse(
        model,
        renderer_name,
        "<tool_call>bash\n"
        "<arg_key>command</arg_key>\n<arg_value>pwd</arg_value>\n"
        "</tool_call>",
        _TOOLS,
    )
    assert _statuses(parsed) == [ToolCallParseStatus.OK]
    assert parsed.tool_calls[0].name == "bash"
    assert parsed.tool_calls[0].arguments == {"command": "pwd"}


def test_known_name_without_parameters_is_ok(model, renderer_name):
    parsed = _parse(model, renderer_name, "<tool_call>submit\n</tool_call>", _TOOLS)
    assert _statuses(parsed) == [ToolCallParseStatus.OK]
    assert parsed.tool_calls[0].name == "submit"


def test_unknown_name_is_flagged(model, renderer_name):
    parsed = _parse(
        model,
        renderer_name,
        "<tool_call>read\n"
        "<arg_key>lines</arg_key>\n<arg_value>10</arg_value>\n"
        "</tool_call>",
        _TOOLS,
    )
    assert _statuses(parsed) == [ToolCallParseStatus.UNKNOWN_TOOL]
    # The attempt stays visible (unlike vLLM's silent drop): name and
    # parsed arguments are preserved for verifier / RL-loss consumers.
    assert parsed.tool_calls[0].name == "read"
    assert parsed.tool_calls[0].arguments == {"lines": 10}


def test_unknown_name_without_args_is_flagged(model, renderer_name):
    parsed = _parse(model, renderer_name, "<tool_call>finish\n</tool_call>", _TOOLS)
    assert _statuses(parsed) == [ToolCallParseStatus.UNKNOWN_TOOL]


def test_missing_arg_key_block_is_flagged(model, renderer_name):
    # No <arg_key> token ⇒ the whole block resolves as the name — both
    # here and in vLLM's engine (unmatched terminals in TOOL_NAME state
    # accumulate into the name) — and then fails validation.
    parsed = _parse(
        model,
        renderer_name,
        "<tool_call>bash\n<arg_value>pwd</arg_value>\n</tool_call>",
        _TOOLS,
    )
    assert _statuses(parsed) == [ToolCallParseStatus.UNKNOWN_TOOL]
    assert "bash" in (parsed.tool_calls[0].name or "")


def test_mixed_calls_flag_only_unknown(model, renderer_name):
    parsed = _parse(
        model,
        renderer_name,
        "<tool_call>bash\n"
        "<arg_key>command</arg_key>\n<arg_value>ls</arg_value>\n"
        "</tool_call>"
        "<tool_call>read\n"
        "<arg_key>path</arg_key>\n<arg_value>x</arg_value>\n"
        "</tool_call>",
        _TOOLS,
    )
    assert _statuses(parsed) == [
        ToolCallParseStatus.OK,
        ToolCallParseStatus.UNKNOWN_TOOL,
    ]


def test_no_tools_means_no_validation(model, renderer_name):
    # vLLM skips name validation when the request carries no tools; so
    # do we — this also keeps tools-less parse_response calls (the
    # common test / SFT path) byte-for-byte backward compatible.
    parsed = _parse(
        model,
        renderer_name,
        "<tool_call>read\n"
        "<arg_key>lines</arg_key>\n<arg_value>10</arg_value>\n"
        "</tool_call>",
        None,
    )
    assert _statuses(parsed) == [ToolCallParseStatus.OK]


def test_openai_envelope_tools_are_recognized(model, renderer_name):
    wrapped = [{"type": "function", "function": t} for t in _TOOLS]
    parsed = _parse(
        model,
        renderer_name,
        "<tool_call>bash\n"
        "<arg_key>command</arg_key>\n<arg_value>pwd</arg_value>\n"
        "</tool_call>",
        wrapped,
    )
    assert _statuses(parsed) == [ToolCallParseStatus.OK]
