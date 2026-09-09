"""Reasoning state follows the generation prefix, independently of termination."""

from functools import lru_cache

import pytest
from parity import MODEL_CATALOG

from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import config_from_name


# Exercise every registered renderer, including inherited family variants.
_CASES = {}
for _case in MODEL_CATALOG:
    _CASES.setdefault(_case.resolved_renderer, _case.model)


@lru_cache(None)
def _renderer(name):
    config = config_from_name(name)
    updates = {"thinking_retention": "all"}
    if "enable_thinking" in type(config).model_fields:
        updates["enable_thinking"] = True
    if "thinking" in type(config).model_fields:
        updates["thinking"] = True
    if name == "hy3":
        updates["reasoning_effort"] = "high"
    tok = load_tokenizer(_CASES[name])
    return tok, create_renderer(tok, config.model_copy(update=updates))


def _encode(tok, text):
    return list(tok.encode(text, add_special_tokens=False))


def _thinking_stream(name, tok, renderer):
    prompt = renderer.render_ids(
        [{"role": "user", "content": "Calculate carefully."}],
        add_generation_prompt=True,
    )
    if name == "gpt-oss":
        return (
            prompt,
            _encode(tok, "unfinished reasoning"),
            _encode(tok, "<|end|><|start|>assistant<|channel|>final<|message|>Answer"),
        )
    if name == "gemma4":
        return (
            prompt,
            _encode(tok, "<|channel>thought\nunfinished reasoning"),
            _encode(tok, "<channel|>Answer"),
        )
    if name in {"inkling", "inkling-small"}:
        return (
            prompt,
            _encode(tok, "<|content_thinking|>unfinished reasoning"),
            _encode(tok, "<|end_message|><|message_model|><|content_text|>Answer"),
        )
    if name == "hy3":
        return (
            prompt,
            _encode(tok, "unfinished reasoning"),
            [renderer._think_end, *_encode(tok, "Answer")],
        )
    prefix = tok.decode(prompt, skip_special_tokens=False).rstrip()
    opener = "" if prefix.endswith("<think>") else "<think>"
    return (
        prompt,
        _encode(tok, opener + "unfinished reasoning"),
        _encode(tok, "</think>Answer"),
    )


@pytest.mark.parametrize("name", sorted(set(_CASES) - {"llama-3", "default"}))
@pytest.mark.parametrize("stop", [False, True])
def test_every_reasoning_renderer_preserves_unfinished_reasoning(name, stop):
    tok, renderer = _renderer(name)
    prompt, incomplete, closed = _thinking_stream(name, tok, renderer)
    sampled = incomplete + (renderer.get_stop_token_ids()[:1] if stop else [])
    original = list(sampled)
    result = renderer.parse_response(sampled, prompt_ids=prompt)
    assert result.content == ""
    assert "unfinished reasoning" in result.reasoning_content
    assert not result.tool_calls
    assert result.reasoning_complete is False
    assert sampled == original
    completed = renderer.parse_response(incomplete + closed, prompt_ids=prompt)
    assert completed.reasoning_complete is True
    assert completed.content == "Answer"


@pytest.mark.parametrize("name", sorted(set(_CASES) - {"llama-3", "default"}))
def test_every_reasoning_bridge_preserves_prefix_or_refuses_sampled_stop(name):
    tok, renderer = _renderer(name)
    prompt, incomplete, _ = _thinking_stream(name, tok, renderer)
    new_messages = [{"role": "user", "content": "Next question."}]
    bridge = renderer.bridge_to_next_turn(prompt, incomplete, new_messages)
    assert bridge is not None
    prefix_len = len(prompt) + len(incomplete)
    assert bridge.token_ids[:prefix_len] == prompt + incomplete
    assert not any(bridge.sampled_mask[prefix_len:])
    close_ids = {
        "gpt-oss": lambda: [renderer._end],
        "gemma4": lambda: [renderer._channel_end],
        "inkling": lambda: [renderer._end_message],
        "hy3": lambda: [renderer._think_end],
    }.get(name, lambda: _encode(tok, "</think>"))()
    assert bridge.token_ids[prefix_len : prefix_len + len(close_ids)] == close_ids
    assert not any(bridge.is_content[prefix_len : prefix_len + len(close_ids)])
    for stop in renderer.get_stop_token_ids():
        stopped_bridge = renderer.bridge_to_next_turn(
            prompt, incomplete + [stop], new_messages
        )
        # Harmony's handoff token closes analysis; its return token does not.
        if name == "gpt-oss" and [stop] == _encode(tok, "<|call|>"):
            assert stopped_bridge is not None
            assert stopped_bridge.token_ids[: prefix_len + 1] == prompt + incomplete + [
                stop
            ]
        else:
            assert stopped_bridge is None


def test_qwen35_prompt_context_overrides_renderer_mode():
    tok, renderer = _renderer("qwen3.5")
    plain_prompt = _encode(tok, "<|im_start|>assistant\n<think></think>\n")
    answer = _encode(tok, "Answer")
    assert renderer.parse_response(answer, prompt_ids=plain_prompt).content == "Answer"
    thinking_prompt = _encode(tok, "<|im_start|>assistant\n<think>already thinking ")
    parsed = renderer.parse_response(answer, prompt_ids=thinking_prompt)
    assert parsed.content == ""
    assert parsed.reasoning_content == "Answer"
    assert parsed.reasoning_complete is False


def test_gemma_post_tool_prompt_supplies_reasoning_state():
    tok, renderer = _renderer("gemma4")
    messages = [
        {"role": "user", "content": "Calculate."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "calculate", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "4"},
    ]
    prompt = renderer.render_ids(messages, add_generation_prompt=True)
    assert tok.decode(prompt, skip_special_tokens=False).endswith("<|channel>thought\n")
    result = renderer.parse_response(
        _encode(tok, "Still considering"), prompt_ids=prompt
    )
    assert result.content == ""
    assert result.reasoning_content == "Still considering"
    assert result.reasoning_complete is False


@pytest.mark.parametrize(
    "name", ["qwen3", "qwen3.5", "deepseek-r1", "kimi-k2", "kimi-k2.5", "laguna-xs-2.1"]
)
def test_tool_drafts_in_unfinished_reasoning_are_not_executable(name):
    tok, renderer = _renderer(name)
    prompt, incomplete, _ = _thinking_stream(name, tok, renderer)
    draft = '<tool_call>{"name":"danger","arguments":{}}</tool_call>'
    if name.startswith("kimi"):
        draft = "<|tool_calls_section_begin|><|tool_call_begin|>functions.danger:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
    if name == "deepseek-r1":
        draft = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>danger\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    result = renderer.parse_response(
        incomplete + _encode(tok, draft), prompt_ids=prompt
    )
    assert result.content == ""
    assert "danger" in result.reasoning_content
    assert result.tool_calls == []
    assert result.reasoning_complete is False


@pytest.mark.parametrize(
    "name", ["qwen3", "qwen3.5", "deepseek-r1", "kimi-k2.5", "laguna-xs-2.1", "gemma4"]
)
def test_later_reasoning_markers_follow_the_format_grammar(name):
    tok, renderer = _renderer(name)
    prompt, incomplete, closed = _thinking_stream(name, tok, renderer)
    opening = "<|channel>thought\n" if name == "gemma4" else "<think>"
    result = renderer.parse_response(
        incomplete + closed + _encode(tok, opening + "more reasoning"),
        prompt_ids=prompt,
    )
    if name == "gemma4":
        assert result.content == ""
        assert "more reasoning" in result.reasoning_content
        assert result.reasoning_complete is False
    else:
        assert result.content == "Answer<think>more reasoning"
        assert result.reasoning_complete is True
    assert result.reasoning_content.startswith("unfinished reasoning")


def test_default_renderer_uses_prompt_before_extracting_tool_drafts():
    from renderers.configs import DefaultRendererConfig

    tok, _ = _renderer("qwen3")
    renderer = create_renderer(
        tok, DefaultRendererConfig(tool_parser="qwen3", reasoning_parser="think")
    )
    prompt = _encode(tok, "<|im_start|>assistant\n<think>")
    draft = _encode(
        tok, 'Considering <tool_call>{"name":"example","arguments":{}}</tool_call>'
    )
    result = renderer.parse_response(draft, prompt_ids=prompt)
    assert result.content == ""
    assert "example" in result.reasoning_content
    assert not result.tool_calls
    assert result.reasoning_complete is False


def test_empty_final_channel_is_not_unfinished_reasoning():
    tok, renderer = _renderer("qwen3.5")
    prompt = _encode(tok, "<|im_start|>assistant\n<think>")
    result = renderer.parse_response(_encode(tok, "</think>"), prompt_ids=prompt)
    assert result.content == ""
    assert result.reasoning_complete is True


def test_no_reasoning_format_keeps_literal_think_text():
    tok, renderer = _renderer("llama-3")
    result = renderer.parse_response(_encode(tok, "Literal <think> text"))
    assert result.content == "Literal <think> text"
    assert result.reasoning_content is None
    assert result.reasoning_complete is True


def test_catalog_covers_every_registered_renderer():
    from renderers.base import MODEL_RENDERER_MAP

    assert set(MODEL_RENDERER_MAP.values()) <= set(_CASES)


@pytest.mark.parametrize("name", ["qwen3", "qwen3.5", "deepseek-r1", "kimi-k2.5"])
def test_literal_text_lookalike_is_not_a_reasoning_token(name):
    tok, renderer = _renderer(name)
    # Tokenize separately so the decoded bytes spell the marker without its
    # structural token ID. The reasoning scanner must match the grammar IDs.
    ids = _encode(tok, "<thi") + _encode(tok, "nk>") + _encode(tok, "example")
    result = renderer.parse_response(ids, prompt_ids=[])
    assert result.reasoning_complete is True
    assert result.content == "<think>example"


def test_inkling_partial_reasoning_prefix_is_not_returned_as_content():
    tok, renderer = _renderer("inkling")
    prompt = _encode(tok, "<|message_model|><|content_thinking|>already considered ")
    sampled = _encode(tok, "the next step")
    parsed = renderer.parse_response(sampled, prompt_ids=prompt)
    assert parsed.content == ""
    assert parsed.reasoning_content == "the next step"
    assert parsed.reasoning_complete is False
    closed = renderer.parse_response(
        sampled
        + _encode(tok, "<|end_message|><|message_model|><|content_text|>Answer"),
        prompt_ids=prompt,
    )
    assert closed.reasoning_content == "the next step"
    assert closed.content == "Answer"
    assert closed.reasoning_complete is True


@pytest.mark.parametrize(
    "name", sorted((set(_CASES) - {"llama-3"}) | {"default-think"})
)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("origin", ["prompt", "completion", "closed_prompt", "both"])
@pytest.mark.parametrize("ending_kind", ["truncated", "stopped", "closed"])
def test_reasoning_markers_override_generation_mode(name, enabled, origin, ending_kind):
    """Generation settings must not hide reasoning actually present in the stream."""
    if name in {"default", "default-think"}:
        from renderers.configs import DefaultRendererConfig

        tok, _ = _renderer("qwen3")
        renderer = create_renderer(
            tok,
            DefaultRendererConfig(
                reasoning_parser="think" if name == "default-think" else None
            ),
        )
    else:
        tok, original = _renderer(name)
        updates = {}
        for key in ("enable_thinking", "thinking"):
            if key in type(original.config).model_fields:
                updates[key] = enabled
        if name == "hy3":
            updates["reasoning_effort"] = "high" if enabled else "no_think"
        renderer = create_renderer(tok, original.config.model_copy(update=updates))

    opening, ending = {
        "gpt-oss": (
            "<|start|>assistant<|channel|>analysis<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        ),
        "gemma4": ("<|channel>thought\n", "<channel|>"),
        "inkling": (
            "<|message_model|><|content_thinking|>",
            "<|end_message|><|message_model|><|content_text|>",
        ),
        "hy3": ("<think:opensource>", "</think:opensource>"),
    }.get(name, ("<think>", "</think>"))
    prompt_text = {
        "prompt": opening + "Earlier reasoning ",
        "completion": "",
        "closed_prompt": opening + "Earlier reasoning" + ending,
        "both": opening,
    }[origin]
    prompt = _encode(tok, prompt_text)
    text = (opening if origin != "prompt" else "") + "Reasoning"
    closed = ending_kind == "closed"
    if closed:
        text += ending + "Answer"
    sampled = _encode(tok, text)
    if ending_kind == "stopped":
        sampled += renderer.get_stop_token_ids()[:1]
    original = list(sampled)
    parsed = renderer.parse_response(sampled, prompt_ids=prompt)
    assert parsed.reasoning_content == "Reasoning"
    assert parsed.content == ("Answer" if closed else "")
    assert parsed.reasoning_complete is closed
    if origin == "completion":
        assert renderer.parse_response(sampled) == parsed
        assert renderer.parse_response(sampled, prompt_ids=None) == parsed
    assert not parsed.tool_calls
    assert sampled == original


@pytest.mark.parametrize(
    "name", sorted(set(_CASES) - {"llama-3", "gpt-oss", "gemma4", "inkling", "default"})
)
@pytest.mark.parametrize("closed", [False, True])
def test_late_think_markers_stay_content(name, closed):
    tok, renderer = _renderer(name)
    opening, ending = (
        ("<think:opensource>", "</think:opensource>")
        if name == "hy3"
        else ("<think>", "</think>")
    )
    literal = "Before" + opening + "literal" + (ending + "After" if closed else "")
    parsed = renderer.parse_response(_encode(tok, literal), prompt_ids=[])
    assert parsed.content == literal
    assert parsed.reasoning_content is None
    assert parsed.reasoning_complete is True
    continued = renderer.parse_response(
        _encode(tok, "continued"), prompt_ids=_encode(tok, literal)
    )
    assert continued.content == "continued"
    assert continued.reasoning_content is None
    assert continued.reasoning_complete is True

    # After a valid initial reasoning block closes, later markers stay content.
    prompt, reasoning, suffix = _thinking_stream(name, tok, renderer)
    completion = reasoning + suffix + _encode(tok, literal)
    parsed = renderer.parse_response(completion, prompt_ids=prompt)
    assert parsed.content == "Answer" + literal
    assert parsed.reasoning_complete is True
    assert parsed.reasoning_content == "unfinished reasoning"
    bridged = renderer.bridge_to_next_turn(
        prompt, completion, [{"role": "user", "content": "Next"}]
    )
    assert bridged is not None
    prefix = prompt + completion
    assert bridged.token_ids[: len(prefix)] == prefix
    close_ids = _encode(tok, ending)
    assert bridged.token_ids[len(prefix) : len(prefix) + len(close_ids)] != close_ids

    # The same content may already be part of the assistant prompt prefix.
    parsed = renderer.parse_response(_encode(tok, "continued"), prompt_ids=prefix)
    assert parsed.content == "continued"
    assert parsed.reasoning_content is None
    assert parsed.reasoning_complete is True


@pytest.mark.parametrize("reasoning_parser", [None, "think"])
@pytest.mark.parametrize(
    "literal",
    [
        "Before<think>literal",
        "Before<think>literal</think>After",
        "Before</think>After",
    ],
)
def test_default_renderer_preserves_late_think_markers(reasoning_parser, literal):
    from renderers.configs import DefaultRendererConfig

    tok, _ = _renderer("qwen3")
    renderer = create_renderer(
        tok, DefaultRendererConfig(reasoning_parser=reasoning_parser)
    )
    parsed = renderer.parse_response(_encode(tok, literal), prompt_ids=[])
    assert parsed.content == literal
    assert parsed.reasoning_content is None
    assert parsed.reasoning_complete is not False


@pytest.mark.parametrize("name", sorted(_CASES))
def test_missing_and_empty_prompt_are_self_contained(name):
    if name == "default":
        from renderers.configs import DefaultRendererConfig

        tok, _ = _renderer("qwen3")
        renderer = create_renderer(tok, DefaultRendererConfig())
    else:
        tok, renderer = _renderer(name)
    bare = _encode(tok, "Answer")
    expected = renderer.parse_response(bare, prompt_ids=[])
    assert expected.content == "Answer"
    assert expected.reasoning_content is None
    assert expected.reasoning_complete is not False
    assert renderer.parse_response(bare) == expected
    assert renderer.parse_response(bare, prompt_ids=None) == expected

    if name not in {"llama-3", "default"}:
        prompt, incomplete, closed = _thinking_stream(name, tok, renderer)
        # Missing context is equivalent for both closed and truncated streams.
        for sampled in (incomplete, incomplete + closed):
            expected = renderer.parse_response(sampled, prompt_ids=[])
            assert renderer.parse_response(sampled) == expected
            assert renderer.parse_response(sampled, prompt_ids=None) == expected
        with_prompt = renderer.parse_response(incomplete, prompt_ids=prompt)
        assert with_prompt.reasoning_complete is False
        assert with_prompt.content == ""
