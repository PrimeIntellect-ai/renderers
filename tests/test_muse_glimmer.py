"""Muse Glimmer (ATEM) renderer coverage.

Covers ``MuseGlimmerRenderer`` and the ``meta-models/Muse-Glimmer-30B`` entry in
``MODEL_RENDERER_MAP``. Every render is checked byte-for-byte against the model's own
``apply_chat_template``; ``current_date`` is pinned on both sides so the template's
``strftime_now`` branch stays deterministic.
"""

from __future__ import annotations

import pytest
from huggingface_hub.errors import GatedRepoError
from renderers import create_renderer
from renderers.base import MODEL_RENDERER_MAP, ToolCallParseStatus, load_tokenizer
from renderers.configs import MuseGlimmerRendererConfig
from renderers.muse_glimmer import MuseGlimmerRenderer

MODEL = "meta-models/Muse-Glimmer-30B"
_PINNED_DATE = "2026-08-19"

TOOLS = [
    {
        "name": "search.web",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "search.news",
        "description": "Search news <b>only</b>.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return load_tokenizer(MODEL)
    except GatedRepoError:
        pytest.skip(f"{MODEL} requires Hugging Face access or a local cache")


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return MuseGlimmerRenderer(
        tokenizer, MuseGlimmerRendererConfig(current_date=_PINNED_DATE)
    )


def _expected_ids(
    tokenizer, messages, tools=None, add_generation_prompt=False, **kwargs
):
    kwargs.setdefault("current_date", _PINNED_DATE)
    return list(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )["input_ids"]
    )


CASES = {
    "user_gen_prompt": ([{"role": "user", "content": "Hi there"}], None, True),
    "user_no_gen_prompt": ([{"role": "user", "content": "Hi there"}], None, False),
    "empty_user_content": ([{"role": "user", "content": ""}], None, True),
    "unicode_and_quotes": (
        [{"role": "user", "content": 'héllo "world" <tag> & more ✓'}],
        None,
        True,
    ),
    "system_and_user": (
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}],
        None,
        True,
    ),
    # The template suppresses its own reasoning line when the prompt already has one.
    "system_mentions_reasoning_strength": (
        [
            {"role": "system", "content": "Reasoning strength: low."},
            {"role": "user", "content": "Hi"},
        ],
        None,
        True,
    ),
    # ...and normalises "Reasoning effort" to "Reasoning strength".
    "system_says_reasoning_effort": (
        [
            {"role": "system", "content": "Reasoning effort: medium. Be nice."},
            {"role": "user", "content": "Hi"},
        ],
        None,
        True,
    ),
    "assistant_reply": (
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        None,
        False,
    ),
    "assistant_with_reasoning": (
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "reasoning_content": "Add them.", "content": "4"},
        ],
        None,
        False,
    ),
    "multi_turn": (
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ],
        None,
        True,
    ),
    # Consecutive same-role messages close with <|eom|> until the last.
    "consecutive_assistants": (
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "One."},
            {"role": "assistant", "content": "Two."},
        ],
        None,
        False,
    ),
    "tools_and_user": ([{"role": "user", "content": "Search cats"}], TOOLS, True),
    "tools_and_system": (
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Go"}],
        TOOLS,
        True,
    ),
    "assistant_tool_call": (
        [
            {"role": "user", "content": "Search cats"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "search.web",
                            "arguments": {"query": "cats", "top_k": 3},
                        },
                    }
                ],
            },
        ],
        TOOLS,
        False,
    ),
    "tool_call_arg_types": (
        [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "search.web",
                            "arguments": {
                                "flag": True,
                                "off": False,
                                "nothing": None,
                                "items": [1, 2],
                                "obj": {"k": "v"},
                                "text": "plain string",
                            },
                        },
                    }
                ],
            },
        ],
        TOOLS,
        False,
    ),
    "parallel_tool_calls": (
        [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "search.web", "arguments": {"query": "a"}},
                    },
                    {
                        "id": "c2",
                        "function": {"name": "search.news", "arguments": {"q": "b"}},
                    },
                ],
            },
        ],
        TOOLS,
        False,
    ),
    "full_tool_cycle": (
        [
            {"role": "user", "content": "Search cats"},
            {
                "role": "assistant",
                "reasoning_content": "Need search.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "search.web",
                            "arguments": {"query": "cats"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "10 results"},
        ],
        TOOLS,
        True,
    ),
    "tool_response_named": (
        [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "search.news", "arguments": {"q": "x"}},
                    }
                ],
            },
            {"role": "tool", "name": "search.news", "content": "3"},
        ],
        TOOLS,
        True,
    ),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_render_matches_apply_chat_template(renderer, tokenizer, case):
    messages, tools, add_generation_prompt = CASES[case]
    expected = _expected_ids(tokenizer, messages, tools, add_generation_prompt)
    got = renderer.render_ids(
        messages, tools=tools, add_generation_prompt=add_generation_prompt
    )
    assert got == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("reasoning_strength", "low"),
        ("reasoning_strength", "medium"),
        ("knowledge_cutoff", "2025-01-01"),
        ("current_date", "2001-09-11"),
    ],
)
def test_config_fields_match_template_kwargs(tokenizer, field, value):
    """Every field in ``template_field_names`` must mirror the Jinja kwarg."""
    config_kwargs = {"current_date": _PINNED_DATE, field: value}
    renderer = MuseGlimmerRenderer(
        tokenizer, MuseGlimmerRendererConfig(**config_kwargs)
    )
    messages = [{"role": "user", "content": "Hi"}]
    expected = _expected_ids(
        tokenizer, messages, add_generation_prompt=True, **{field: value}
    )
    assert renderer.render_ids(messages, add_generation_prompt=True) == expected


def test_auto_resolution_selects_muse_glimmer(tokenizer):
    assert MODEL_RENDERER_MAP[MODEL] == "muse-glimmer"
    assert isinstance(create_renderer(tokenizer), MuseGlimmerRenderer)


@pytest.mark.parametrize("content_type", ["image", "image_url", "video", "video_url"])
def test_rejects_multimodal_content(renderer, content_type):
    messages = [{"role": "user", "content": [{"type": content_type}]}]

    with pytest.raises(ValueError, match="text-only"):
        renderer.render(messages)


def test_eom_is_not_a_stop_token(renderer, tokenizer):
    """``<|eom|>`` closes a channel; generation continues into the next one."""
    stop_ids = renderer.get_stop_token_ids()
    assert tokenizer.convert_tokens_to_ids("<|eot|>") in stop_ids
    assert tokenizer.convert_tokens_to_ids("<|end_of_text|>") in stop_ids
    assert tokenizer.convert_tokens_to_ids("<|eom|>") not in stop_ids


def test_sampled_mask_covers_exactly_the_model_emission(renderer, tokenizer):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "reasoning_content": "Add them.", "content": "4"},
    ]
    rendered = renderer.render(messages)
    prompt_ids = renderer.render_ids(messages[:1], add_generation_prompt=True)

    assert rendered.token_ids[: len(prompt_ids)] == prompt_ids
    assert not any(rendered.sampled_mask[: len(prompt_ids)])
    assert all(rendered.sampled_mask[len(prompt_ids) :])

    sampled_text = tokenizer.decode(
        [t for t, s in zip(rendered.token_ids, rendered.sampled_mask) if s]
    )
    # The generation prompt ends at a bare ``<|start|>assistant``, so the recipient,
    # the ``<|message|>`` separator and every later channel opener are model-sampled.
    assert sampled_text == (
        " to=self<|message|>Add them.<|eom|><|start|>assistant to=user<|message|>4<|eot|>"
    )


def test_is_content_tracks_sampled_mask_on_assistant_tokens(renderer):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "reasoning_content": "Add them.", "content": "4"},
    ]
    rendered = renderer.render(messages)
    assistant = [i for i, idx in enumerate(rendered.message_indices) if idx == 1]
    assert assistant
    assert all(rendered.is_content[i] == rendered.sampled_mask[i] for i in assistant)


def test_consecutive_assistant_opener_is_sampled(renderer):
    messages = [
        {"role": "user", "content": "Continue twice."},
        {"role": "assistant", "content": "One.", "end_turn": False},
        {"role": "assistant", "content": "Two."},
    ]
    rendered = renderer.render(messages)
    second_assistant = [i for i, idx in enumerate(rendered.message_indices) if idx == 2]

    assert second_assistant
    assert all(rendered.sampled_mask[i] for i in second_assistant)


def test_consecutive_assistant_opener_after_eot_is_scaffold(renderer):
    messages = [
        {"role": "user", "content": "Continue twice."},
        {"role": "assistant", "content": "One."},
        {"role": "assistant", "content": "Two."},
    ]
    rendered = renderer.render(messages)
    second_assistant = [i for i, idx in enumerate(rendered.message_indices) if idx == 2]

    assert second_assistant
    assert not any(rendered.sampled_mask[i] for i in second_assistant[:2])
    assert all(rendered.sampled_mask[i] for i in second_assistant[2:])


def test_parse_response_splits_reasoning_from_content(renderer):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "reasoning_content": "Add them.", "content": "4"},
    ]
    rendered = renderer.render(messages)
    prompt_ids = renderer.render_ids(messages[:1], add_generation_prompt=True)

    parsed = renderer.parse_response(rendered.token_ids[len(prompt_ids) :])
    assert parsed.reasoning_content == "Add them."
    assert parsed.content == "4"
    assert parsed.tool_calls == []


def test_parse_response_recovers_typed_tool_arguments(renderer, tokenizer):
    messages = [
        {"role": "user", "content": "Search cats"},
        {
            "role": "assistant",
            "reasoning_content": "Need a search.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "search.web",
                        "arguments": {"query": "cats", "top_k": 3},
                    },
                }
            ],
        },
    ]
    rendered = renderer.render(messages, tools=TOOLS)
    prompt_ids = renderer.render_ids(
        messages[:1], tools=TOOLS, add_generation_prompt=True
    )

    parsed = renderer.parse_response(rendered.token_ids[len(prompt_ids) :], tools=TOOLS)
    assert parsed.reasoning_content == "Need a search."
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "search.web"
    # ``query`` is declared a string so it stays verbatim; ``top_k`` coerces to int.
    assert call.arguments == {"query": "cats", "top_k": 3}
    assert call.token_span is not None
    start, end = call.token_span
    stripped = rendered.token_ids[len(prompt_ids) :]
    while stripped and stripped[-1] in renderer.get_stop_token_ids():
        stripped.pop()
    block = tokenizer.decode(stripped[start:end], skip_special_tokens=False)
    assert block.startswith('<atem:invoke name="search.web">')
    assert block.rstrip().endswith("</atem:invoke>")


def test_parse_response_preserves_malformed_tool_attempt(renderer, tokenizer):
    token_ids = tokenizer.encode(
        " to=search.web<|message|>malformed body<|eot|>",
        add_special_tokens=False,
    )

    parsed = renderer.parse_response(token_ids, tools=TOOLS)

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search.web"
    assert parsed.tool_calls[0].raw == "malformed body"
    assert parsed.tool_calls[0].token_span is not None
    assert parsed.tool_calls[0].status == ToolCallParseStatus.MALFORMED_STRUCTURE


def test_parse_response_preserves_valid_then_truncated_tool_attempt(
    renderer, tokenizer
):
    token_ids = tokenizer.encode(
        " to=search.web<|message|><atem:function_calls>\n"
        '<atem:invoke name="search.web">\n'
        '<atem:parameter name="query">cats</atem:parameter>\n'
        "</atem:invoke>\n"
        '<atem:invoke name="search.news">\n'
        '<atem:parameter name="q">breaking</atem:parameter>\n',
        add_special_tokens=False,
    )

    parsed = renderer.parse_response(token_ids, tools=TOOLS)

    assert [(call.name, call.status) for call in parsed.tool_calls] == [
        ("search.web", ToolCallParseStatus.OK),
        ("search.news", ToolCallParseStatus.UNCLOSED_BLOCK),
    ]
    assert parsed.tool_calls[0].arguments == {"query": "cats"}


def test_parse_response_marks_unclosed_parameter_inside_closed_invoke(
    renderer, tokenizer
):
    token_ids = tokenizer.encode(
        " to=search.web<|message|><atem:function_calls>\n"
        '<atem:invoke name="search.web">\n'
        '<atem:parameter name="top_k">3</atem:parameter>\n'
        '<atem:parameter name="query">cats\n'
        "</atem:invoke>\n"
        "</atem:function_calls><|eot|>",
        add_special_tokens=False,
    )

    parsed = renderer.parse_response(token_ids, tools=TOOLS)

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search.web"
    assert parsed.tool_calls[0].arguments == {"top_k": 3}
    assert parsed.tool_calls[0].token_span is not None
    assert parsed.tool_calls[0].status == ToolCallParseStatus.UNCLOSED_BLOCK


@pytest.mark.parametrize(
    ("recipient", "name", "status"),
    [
        ("evil.tool", "evil.tool", ToolCallParseStatus.UNKNOWN_TOOL),
        ("search.web", "search.news", ToolCallParseStatus.MALFORMED_STRUCTURE),
    ],
)
def test_parse_response_validates_tool_name_and_recipient(
    renderer, tokenizer, recipient, name, status
):
    token_ids = tokenizer.encode(
        f" to={recipient}<|message|><atem:function_calls>\n"
        f'<atem:invoke name="{name}">\n'
        '<atem:parameter name="query">cats</atem:parameter>\n'
        "</atem:invoke>\n"
        "</atem:function_calls><|eot|>",
        add_special_tokens=False,
    )

    parsed = renderer.parse_response(token_ids, tools=TOOLS)

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == name
    assert parsed.tool_calls[0].status == status


def test_bridge_matches_a_full_rerender(renderer):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "reasoning_content": "Add them.", "content": "4"},
    ]
    prompt_ids = renderer.render_ids(messages[:1], add_generation_prompt=True)
    completion = renderer.render(messages).token_ids[len(prompt_ids) :]
    follow_up = [{"role": "user", "content": "Thanks"}]

    bridged = renderer.bridge_to_next_turn(prompt_ids, completion, follow_up)
    assert bridged is not None
    assert (
        bridged.token_ids[: len(prompt_ids) + len(completion)]
        == prompt_ids + completion
    )
    assert bridged.token_ids == renderer.render_ids(
        messages + follow_up, add_generation_prompt=True
    )


def test_bridge_refuses_a_new_system_message(renderer):
    """A system message rewrites the block at the front of the prompt; it can't be appended."""
    messages = [{"role": "user", "content": "Hi"}]
    prompt_ids = renderer.render_ids(messages, add_generation_prompt=True)
    assert (
        renderer.bridge_to_next_turn(
            prompt_ids, [], [{"role": "system", "content": "New rules."}]
        )
        is None
    )


def test_bridge_refuses_tool_call_id_without_name(renderer):
    messages = [{"role": "user", "content": "Search cats"}]
    prompt_ids = renderer.render_ids(messages, add_generation_prompt=True)
    tool_response = [{"role": "tool", "tool_call_id": "call_1", "content": "results"}]

    assert renderer.bridge_to_next_turn(prompt_ids, [], tool_response) is None
