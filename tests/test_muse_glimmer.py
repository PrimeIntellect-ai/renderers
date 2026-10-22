"""Muse Glimmer (ATEM) renderer.

Text rendering is covered by the shared parity barrage against the pinned
Jinja template. These tests cover what is specific to Muse Glimmer: images as
one ``<|patch|>`` placeholder each (the logical prompt), sampled/content masks
across the several messages of one sampled turn, response parsing matched to
vLLM's ``muse_glimmer`` reasoning + tool parsers, and bridge closes.

The template recognises ``{"type": "image"}`` parts only; vLLM converts
``image_url`` parts to that form before applying it, so reference renders
normalise image parts the same way.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import pytest
from renderers.base import ToolCallParseStatus

MODEL = "meta-models/Muse-Glimmer-30B"
DATE = "2026-10-16"
IMAGE = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "units": {"type": "array"},
                    "code": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "maps.route",
            "description": "Route.",
            "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
        },
    },
]


def _call(call_id: str, name: str, arguments) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _invoke(name: str, *params: tuple[str, str]) -> str:
    body = "".join(
        f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params
    )
    return f'<atem:function_calls>\n<atem:invoke name="{name}">\n{body}</atem:invoke>\n</atem:function_calls>'


@lru_cache(maxsize=None)
def _load(**config):
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import MuseGlimmerRendererConfig

    tok = load_tokenizer(MODEL)
    return tok, create_renderer(
        tok, MuseGlimmerRendererConfig(current_date=DATE, **config)
    )


def _as_template_images(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [
                {"type": "image"}
                if isinstance(p, dict) and p.get("type") == "image_url"
                else p
                for p in content
            ]
            message = {**message, "content": content}
        out.append(message)
    return out


def _reference(
    messages, *, tools=None, add_generation_prompt=True, **kwargs
) -> list[int]:
    tok, _ = _load()
    text = tok.apply_chat_template(
        _as_template_images(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        current_date=DATE,
        **kwargs,
    )
    return tok.encode(text, add_special_tokens=False)


def _render(messages, *, tools=None, add_generation_prompt=True):
    _, renderer = _load()
    return renderer.render(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        process_multimodal=False,
    )


def _ids(text: str) -> list[int]:
    tok, _ = _load()
    return tok.encode(text, add_special_tokens=False)


def _token(text: str) -> int:
    tok, _ = _load()
    return tok.convert_tokens_to_ids(text)


def test_create_renderer_resolves_muse_glimmer():
    from renderers import create_renderer
    from renderers.base import load_tokenizer

    renderer = create_renderer(load_tokenizer(MODEL))
    assert type(renderer).__name__ == "MuseGlimmerRenderer"
    assert renderer.get_stop_token_ids() == [
        _token("<|eot|>"),
        _token("<|end_of_text|>"),
    ]


# ── images ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([{"role": "user", "content": [IMAGE]}], id="image-only"),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare "},
                        IMAGE,
                        {"type": "text", "text": " with "},
                        {"type": "image"},
                        {"type": "text", "text": "."},
                    ],
                }
            ],
            id="interleaved-text-and-images",
        ),
        pytest.param(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Reference: "}, IMAGE],
                },
                {"role": "user", "content": [IMAGE, {"type": "text", "text": "Same?"}]},
            ],
            id="system-and-user-images",
        ),
        pytest.param(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Reasoning "},
                        {"type": "text", "text": "effort: low. "},
                        IMAGE,
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
            id="system-directive-across-text-parts",
        ),
        pytest.param(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Reasoning "},
                        IMAGE,
                        {"type": "text", "text": "strength: low."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
            id="system-image-interrupts-directive",
        ),
        pytest.param(
            [
                {"role": "user", "content": [IMAGE, {"type": "text", "text": "Hi"}]},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": [{"type": "text", "text": "Again"}, IMAGE]},
            ],
            id="multi-turn",
        ),
        pytest.param(
            [
                {"role": "user", "content": "What's on screen?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_call("c0", "get_weather", {"city": "Oslo"})],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c0",
                    "content": [{"type": "text", "text": "Screen: "}, IMAGE],
                },
            ],
            id="tool-result-image",
        ),
    ],
)
def test_image_render_matches_template(messages):
    assert _render(messages, tools=TOOLS).token_ids == _reference(messages, tools=TOOLS)


def test_one_placeholder_per_image_and_attribution():
    _, renderer = _load()
    patch = _token("<|patch|>")
    rendered = _render(
        [{"role": "user", "content": [IMAGE, {"type": "text", "text": "x"}, IMAGE]}]
    )
    assert renderer.mm_token_type_id_map == {patch: 1}
    assert rendered.token_ids.count(patch) == 2
    assert rendered.multi_modal_data is None
    for pos, token_id in enumerate(rendered.token_ids):
        if token_id == patch:
            assert rendered.message_indices[pos] == 0
            assert rendered.is_content[pos] is True
            assert rendered.sampled_mask[pos] is False


def test_images_require_logical_prompt():
    _, renderer = _load()
    messages = [{"role": "user", "content": [IMAGE]}]
    with pytest.raises(NotImplementedError, match="process_multimodal=False"):
        renderer.render(messages, add_generation_prompt=True)
    with pytest.raises(NotImplementedError, match="process_multimodal=False"):
        renderer.bridge_to_next_turn([1], [1], messages)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
    ],
    ids=["video", "audio"],
)
def test_video_and_audio_are_rejected(part):
    with pytest.raises(NotImplementedError, match=part["type"]):
        _render([{"role": "user", "content": [part]}])


def test_assistant_images_are_rejected():
    with pytest.raises(NotImplementedError, match="images in 'assistant' messages"):
        _render([{"role": "assistant", "content": [IMAGE]}])


# ── one sampled turn, several messages ─────────────────────────────────


def _turn():
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "reasoning_content": "Call the tool.",
            "content": "",
            "tool_calls": [_call("c1", "get_weather", {"city": "Paris", "days": 2})],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
        {
            "role": "assistant",
            "reasoning_content": "Done.",
            "content": "Sunny in Paris.",
        },
    ]


def test_turn_masks_follow_what_the_model_samples():
    tok, renderer = _load()
    messages = _turn()
    rendered = renderer.render(messages, tools=TOOLS)
    start, eot, eom = _token("<|start|>"), _token("<|eot|>"), _token("<|eom|>")
    for index in (1, 3):
        positions = [k for k, m in enumerate(rendered.message_indices) if m == index]
        ids = [rendered.token_ids[k] for k in positions]
        sampled = [rendered.sampled_mask[k] for k in positions]
        # ``<|start|>assistant`` of the first message is the generation prompt.
        assert ids[:2] == [start, *_ids("assistant")] and sampled[:2] == [False, False]
        assert all(sampled[2:])
        # Later headers inside the same turn are sampled: the model emits them
        # after ``<|eom|>``, which does not stop generation.
        later = [p for p, t in enumerate(ids) if t == start and p > 0]
        assert later and all(sampled[p] for p in later)
        assert ids[-1] == eot and eom in ids
        assert [rendered.is_content[k] for k in positions] == sampled
    assert not any(
        s
        for s, m in zip(rendered.sampled_mask, rendered.message_indices)
        if m in (0, 2, -1)
    )


def test_render_prompt_plus_parsed_completion_round_trips():
    _, renderer = _load()
    messages = _turn()[:2]
    prompt = renderer.render_ids(messages[:1], tools=TOOLS, add_generation_prompt=True)
    full = renderer.render_ids(messages, tools=TOOLS)
    assert full[: len(prompt)] == prompt
    completion = full[len(prompt) :]
    parsed = renderer.parse_response(completion, tools=TOOLS, prompt_ids=prompt)
    assert parsed.reasoning_content == "Call the tool."
    assert [(c.name, c.arguments, c.status) for c in parsed.tool_calls] == [
        ("get_weather", {"city": "Paris", "days": 2}, ToolCallParseStatus.OK)
    ]


def test_consecutive_assistant_messages_continue_the_turn():
    """A tool-call message followed by another assistant message ends with
    ``<|eom|>``, and the next ``<|start|>assistant`` is model-sampled."""
    _, renderer = _load()
    messages = [
        {"role": "user", "content": "Plan."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("a", "maps.route", {"to": "home"})],
        },
        {"role": "assistant", "content": "Routing."},
    ]
    rendered = renderer.render(messages, tools=TOOLS)
    assert rendered.token_ids == _reference(
        messages, tools=TOOLS, add_generation_prompt=False
    )
    second = [k for k, m in enumerate(rendered.message_indices) if m == 2]
    assert all(rendered.sampled_mask[k] for k in second)


# ── parsing (matched to vLLM 0.29 muse_glimmer reasoning + tool parsers) ─

_PARSE_CASES = {
    "reasoning-then-answer": (
        " to=self<|message|>Let me think.<|eom|><|start|>assistant to=user<|message|>The answer is 4.",
        ("Let me think.", "The answer is 4.", []),
    ),
    "answer-only": (" to=user<|message|>Hello there!", (None, "Hello there!", [])),
    "answer-final-newline": (" to=user<|message|>Hello\n", (None, "Hello", [])),
    "answer-final-two-newlines": (
        " to=user<|message|>Hello\n\n",
        (None, "Hello\n", []),
    ),
    "answer-final-space-newline": (" to=user<|message|>Hello \n", (None, "Hello ", [])),
    "closed-answer-keeps-newline": (
        " to=user<|message|>Hello\n<|eom|>",
        (None, "Hello\n", []),
    ),
    "tool-and-answer-keeps-newline": (
        " to=get_weather<|message|>"
        + _invoke("get_weather", ("city", "Paris"))
        + "<|eom|><|start|>assistant to=user<|message|>Hello\n",
        (None, "Hello\n", [("get_weather", {"city": "Paris"})]),
    ),
    "tool-call": (
        " to=self<|message|>Need weather.<|eom|><|start|>assistant to=get_weather<|message|>"
        + _invoke(
            "get_weather", ("city", "Paris"), ("days", "3"), ("units", '["C", "F"]')
        ),
        (
            "Need weather.",
            "",
            [("get_weather", {"city": "Paris", "days": 3, "units": ["C", "F"]})],
        ),
    ),
    "parallel-calls": (
        " to=self<|message|>Two calls.<|eom|><|start|>assistant to=get_weather<|message|>"
        + _invoke("get_weather", ("city", "Rome"))
        + "<|eom|><|start|>assistant to=maps.route<|message|>"
        + _invoke("maps.route", ("to", "home")),
        (
            "Two calls.",
            "",
            [("get_weather", {"city": "Rome"}), ("maps.route", {"to": "home"})],
        ),
    ),
    "call-echoed-in-reasoning-is-not-executed": (
        " to=self<|message|>I might write "
        + _invoke("get_weather", ("city", "X"))
        + " but no.<|eom|><|start|>assistant to=user<|message|>No call needed.",
        (
            "I might write " + _invoke("get_weather", ("city", "X")) + " but no.",
            "No call needed.",
            [],
        ),
    ),
    "header-without-eom": (
        # Observed vLLM case: the model opens a tool channel without ``<|eom|>``.
        " to=self<|message|>Calling now. to=get_weather<|message|>"
        + _invoke("get_weather", ("city", "Oslo")),
        ("Calling now. ", "", [("get_weather", {"city": "Oslo"})]),
    ),
    "string-schema-keeps-digits": (
        " to=get_weather<|message|>"
        + _invoke("get_weather", ("city", "Paris"), ("code", "007")),
        (None, "", [("get_weather", {"city": "Paris", "code": "007"})]),
    ),
    "several-reasoning-messages": (
        " to=self<|message|>First.<|eom|><|start|>assistant to=self<|message|>Second.<|eom|>"
        "<|start|>assistant to=user<|message|>Done.",
        ("First.\nSecond.", "Done.", []),
    ),
    "doubled-bare-name": (
        " to=get_weather<|message|>"
        + _invoke("get_weather.get_weather", ("city", "Paris")),
        (None, "", [("get_weather", {"city": "Paris"})]),
    ),
    "string-schema-does-not-override-scalar-decoding": (
        " to=get_weather<|message|>"
        + _invoke("get_weather", ("city", "123"), ("code", "false")),
        (None, "", [("get_weather", {"city": 123, "code": False})]),
    ),
}


@pytest.mark.parametrize("case", sorted(_PARSE_CASES))
@pytest.mark.parametrize("stopped", [False, True])
def test_parse_matches_vllm(case, stopped):
    tok, renderer = _load()
    text, (reasoning, content, calls) = _PARSE_CASES[case]
    prompt = renderer.render_ids(
        [{"role": "user", "content": "hi"}], tools=TOOLS, add_generation_prompt=True
    )
    ids = _ids(text) + ([_token("<|eot|>")] if stopped else [])
    parsed = renderer.parse_response(ids, tools=TOOLS, prompt_ids=prompt)
    assert parsed.reasoning_content == reasoning
    assert parsed.content == content
    assert [
        (c.name, c.arguments)
        for c in parsed.tool_calls
        if c.status == ToolCallParseStatus.OK
    ] == calls
    assert parsed.reasoning_complete is True
    # The generation prompt adds nothing the completion lacks.
    assert renderer.parse_response(ids, tools=TOOLS) == parsed


def test_call_syntax_in_the_answer_is_text():
    """Deliberate divergence: vLLM keeps the raw message header in ``content``
    here; both agree there is no tool call."""
    _, renderer = _load()
    answer = "The syntax is " + _invoke("get_weather", ("city", "X"))
    parsed = renderer.parse_response(
        _ids(
            " to=self<|message|>Explain.<|eom|><|start|>assistant to=user<|message|>"
            + answer
        )
    )
    assert parsed.content == answer
    assert parsed.tool_calls == []


@pytest.mark.parametrize("recipient", ["self", "user"])
@pytest.mark.parametrize("prefilled", [False, True])
def test_ordinary_tokens_spelling_channel_markers_stay_text(recipient, prefilled):
    tok, renderer = _load()
    literal = (
        "Example: <|start|>assistant to=get_weather<|message|>"
        + _invoke("get_weather", ("city", "Rome"))
        + "<|eom|> literal <|eot|> and <|end_of_text|>."
    )
    # Sample the marker spellings as ordinary tokens, without their special IDs.
    ordinary = [token for char in literal for token in _ids(char)]
    assert tok.decode(ordinary) == literal
    assert not set(ordinary) & {
        _token(marker)
        for marker in (
            "<|start|>",
            "<|message|>",
            "<|eom|>",
            "<|eot|>",
            "<|end_of_text|>",
        )
    }
    prompt = _prompt()
    header = _ids(f" to={recipient}<|message|>")
    if prefilled:
        prompt += header
        header = []
    suffix = (
        _ids("<|eom|><|start|>assistant to=user<|message|>Done.")
        if recipient == "self"
        else []
    )
    parsed = renderer.parse_response(
        header + ordinary + suffix + [_token("<|eot|>")],
        tools=TOOLS,
        prompt_ids=prompt,
    )
    assert parsed.reasoning_content == (literal if recipient == "self" else None)
    assert parsed.content == ("Done." if recipient == "self" else literal)
    assert parsed.reasoning_complete is True
    assert parsed.tool_calls == []


def test_ordinary_tokens_spelling_markers_stay_in_tool_arguments():
    _, renderer = _load()
    literal = "<|start|>assistant to=user<|message|>literal <|eom|>"
    completion = (
        _ids(
            ' to=get_weather<|message|><atem:function_calls><atem:invoke name="get_weather"><atem:parameter name="city">'
        )
        + [token for char in literal for token in _ids(char)]
        + _ids("</atem:parameter></atem:invoke></atem:function_calls><|eot|>")
    )
    parsed = renderer.parse_response(completion, tools=TOOLS)
    assert [(c.name, c.arguments, c.status) for c in parsed.tool_calls] == [
        ("get_weather", {"city": literal}, ToolCallParseStatus.OK)
    ]


def test_unfinished_tool_call_is_unclosed():
    _, renderer = _load()
    parsed = renderer.parse_response(
        _ids(
            ' to=get_weather<|message|><atem:function_calls>\n<atem:invoke name="get_weather">\n<atem:param'
        )
    )
    assert [c.status for c in parsed.tool_calls] == [ToolCallParseStatus.UNCLOSED_BLOCK]
    assert parsed.tool_calls[0].name is None
    assert '<atem:invoke name="get_weather">' in parsed.content


def test_tool_channel_without_invoke_is_malformed():
    _, renderer = _load()
    parsed = renderer.parse_response(_ids(" to=get_weather<|message|>Paris please"))
    assert [(c.name, c.status) for c in parsed.tool_calls] == [
        (None, ToolCallParseStatus.MALFORMED_STRUCTURE)
    ]
    assert parsed.content == " to=get_weather<|message|>Paris please"


def test_prefilled_reasoning_prompt_continues_reasoning():
    tok, renderer = _load()
    prompt = renderer.render_ids(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    ) + _ids(" to=self<|message|>Already ")
    parsed = renderer.parse_response(_ids("considering"), prompt_ids=prompt)
    assert (parsed.content, parsed.reasoning_content, parsed.reasoning_complete) == (
        "",
        "considering",
        False,
    )


# ── bridges ─────────────────────────────────────────────────────────────


def _prompt():
    _, renderer = _load()
    return renderer.render_ids(
        [{"role": "user", "content": "Weather?"}],
        tools=TOOLS,
        add_generation_prompt=True,
    )


def test_bridge_tool_result_matches_full_render():
    _, renderer = _load()
    messages = _turn()[:3]
    prompt = renderer.render_ids(messages[:1], tools=TOOLS, add_generation_prompt=True)
    full = renderer.render_ids(messages, tools=TOOLS, add_generation_prompt=True)
    eot = full.index(_token("<|eot|>"), len(prompt))
    bridged = renderer.bridge_to_next_turn(
        prompt, full[len(prompt) : eot + 1], messages[2:], tools=TOOLS
    )
    assert bridged is not None
    assert bridged.token_ids == full
    assert not any(bridged.sampled_mask)


@pytest.mark.parametrize("reverse_results", [False, True])
@pytest.mark.parametrize("named_results", [False, True])
@pytest.mark.parametrize("same_function", [False, True])
def test_bridge_tool_result_names(reverse_results, named_results, same_function):
    _, renderer = _load()
    second_name = "get_weather" if same_function else "maps.route"
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call("a", "get_weather", {"city": "Rome"}),
                _call(
                    "b",
                    second_name,
                    {"city": "Paris"} if same_function else {"to": "home"},
                ),
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "Sunny"},
        {"role": "tool", "tool_call_id": "b", "content": "12 km"},
    ]
    if named_results:
        messages[2]["name"] = "get_weather"
        messages[3]["name"] = second_name
    if reverse_results:
        messages[2:] = reversed(messages[2:])
    prompt = _prompt()
    full = renderer.render_ids(messages, tools=TOOLS, add_generation_prompt=True)
    eot = full.index(_token("<|eot|>"), len(prompt))
    completion = full[len(prompt) : eot + 1]
    bridged = renderer.bridge_to_next_turn(
        prompt, completion, messages[2:], tools=TOOLS
    )
    if named_results or same_function:
        assert bridged is not None and bridged.token_ids == full
    else:
        # The sampled ATEM stream has no IDs to match distinct call names to
        # results. Equal counts cannot establish which result belongs to which.
        assert bridged is None
    # A count mismatch cannot be resolved without the assistant message.
    if not named_results:
        assert (
            renderer.bridge_to_next_turn(prompt, completion, messages[2:3], tools=TOOLS)
            is None
        )


def test_bridge_closes_truncated_reasoning_like_a_full_render():
    _, renderer = _load()
    prompt = _prompt()
    completion = _ids(" to=self<|message|>Half a thought")
    bridged = renderer.bridge_to_next_turn(
        prompt, completion, [{"role": "user", "content": "Go on."}]
    )
    assert bridged is not None
    full = renderer.render_ids(
        [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "reasoning_content": "Half a thought", "content": ""},
            {"role": "user", "content": "Go on."},
        ],
        tools=TOOLS,
        add_generation_prompt=True,
    )
    assert bridged.token_ids == full


def test_bridge_closes_truncated_answer_with_eot():
    _, renderer = _load()
    prompt = _prompt()
    completion = _ids(" to=user<|message|>Partly")
    bridged = renderer.bridge_to_next_turn(
        prompt, completion, [{"role": "user", "content": "And?"}]
    )
    assert bridged is not None
    assert bridged.token_ids[len(prompt) + len(completion)] == _token("<|eot|>")


def test_bridge_refuses_stop_inside_reasoning_and_new_system():
    _, renderer = _load()
    prompt = _prompt()
    reasoning = _ids(" to=self<|message|>Half a thought")
    assert (
        renderer.bridge_to_next_turn(
            prompt, reasoning + [_token("<|eot|>")], [{"role": "user", "content": "x"}]
        )
        is None
    )
    answer = _ids(" to=user<|message|>Done") + [_token("<|eot|>")]
    assert (
        renderer.bridge_to_next_turn(
            prompt, answer, [{"role": "system", "content": "New rules."}]
        )
        is None
    )
    assert (
        renderer.bridge_to_next_turn(
            prompt, answer, [{"role": "assistant", "content": "x"}]
        )
        is None
    )


# ── system preamble and arguments ───────────────────────────────────────


@pytest.mark.parametrize("strength", ["low", "medium", "high", "xhigh"])
def test_reasoning_strength_line(strength):
    _, renderer = _load(reasoning_strength=strength)
    messages = [{"role": "user", "content": "Hi"}]
    assert renderer.render_ids(messages, add_generation_prompt=True) == _reference(
        messages, reasoning_strength=strength
    )


@pytest.mark.parametrize(
    "phrase",
    ["Reasoning effort: low", "reasoning strength: medium", "REASONING EFFORT: high"],
)
def test_system_reasoning_directive_is_kept_and_normalised(phrase):
    messages = [
        {"role": "system", "content": f"Be brief. {phrase}"},
        {"role": "user", "content": "Hi"},
    ]
    rendered = _render(messages)
    assert rendered.token_ids == _reference(messages)
    tok, _ = _load()
    assert tok.decode(rendered.token_ids).lower().count("reasoning strength") == 1


def test_current_date_defaults_to_today_once():
    from renderers import create_renderer
    from renderers.configs import MuseGlimmerRendererConfig

    tok, _ = _load()
    renderer = create_renderer(tok, MuseGlimmerRendererConfig())
    text = tok.decode(renderer.render_ids([{"role": "user", "content": "Hi"}]))
    assert f"Current date: {datetime.now().strftime('%Y-%m-%d')}." in text


def test_json_string_arguments_render_like_a_mapping():
    _, renderer = _load()
    as_dict = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("a", "get_weather", {"city": "Rome", "days": 2})],
        },
    ]
    as_json = [
        as_dict[0],
        {
            **as_dict[1],
            "tool_calls": [_call("a", "get_weather", '{"city": "Rome", "days": 2}')],
        },
    ]
    assert renderer.render_ids(as_json, tools=TOOLS) == renderer.render_ids(
        as_dict, tools=TOOLS
    )
    with pytest.raises(ValueError, match="arguments"):
        renderer.render_ids(
            [
                as_dict[0],
                {**as_dict[1], "tool_calls": [_call("a", "get_weather", "[1, 2]")]},
            ]
        )
