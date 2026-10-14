"""GLM-5.3 Flash image rendering.

GLM-5.3 Flash shares GLM-5.3's text template (covered by the shared parity
barrage) and adds images: each image part of a system, user or tool message
renders as ``<|begin_of_image|><|image|><|end_of_image|>``. The renderer emits
the logical prompt (``process_multimodal=False``, one ``<|image|>`` per image)
and leaves expansion to the inference engine, so these tests compare against
the template's untokenized output rather than the processor's expanded ids.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import httpx
import pytest
from openai import AsyncOpenAI

from renderers.client import generate

MODEL = "zai-org/GLM-5.3-Flash-BF16"
GLM53_MODELS = ["zai-org/GLM-5.3", MODEL]

IMAGE = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": "Take a screenshot.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

_TOOL_CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "screenshot", "arguments": {}},
}


@lru_cache(maxsize=None)
def _load(model=MODEL):
    from renderers import create_renderer
    from renderers.base import load_tokenizer

    tok = load_tokenizer(model)
    return tok, create_renderer(tok)


def _reference(messages, *, tools=None, model=MODEL) -> list[int]:
    tok, _ = _load(model)
    text = tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    return tok.encode(text, add_special_tokens=False)


def _render(messages, *, tools=None):
    _, renderer = _load()
    return renderer.render(
        messages, tools=tools, add_generation_prompt=True, process_multimodal=False
    )


def _token(text: str) -> int:
    tok, _ = _load()
    return tok.convert_tokens_to_ids(text)


def _tool_turn(tool_content) -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "What's on screen?"}]},
        {"role": "assistant", "content": "", "tool_calls": [_TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_0", "content": tool_content},
    ]


def test_create_renderer_resolves_flash():
    _, renderer = _load()
    assert type(renderer).__name__ == "GLM53FlashRenderer"


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
                        IMAGE,
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
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                        IMAGE,
                    ],
                }
            ],
            id="adjacent-text-parts",
        ),
        pytest.param(
            [
                {"role": "user", "content": [IMAGE, {"type": "text", "text": "Hi"}]},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": [{"type": "text", "text": "Again"}, IMAGE]},
            ],
            id="multi-turn",
        ),
    ],
)
def test_image_render_matches_template(messages):
    assert _render(messages).token_ids == _reference(messages)


def test_tool_response_image_matches_template():
    messages = _tool_turn([{"type": "text", "text": "Screen: "}, IMAGE])
    assert _render(messages, tools=TOOLS).token_ids == _reference(messages, tools=TOOLS)


def test_one_image_placeholder_per_image():
    rendered = _render(
        [{"role": "user", "content": [IMAGE, {"type": "text", "text": "x"}, IMAGE]}]
    )
    assert rendered.token_ids.count(_token("<|image|>")) == 2
    assert rendered.multi_modal_data is None


def test_image_attribution():
    _, renderer = _load()
    image, begin, end = (
        _token("<|image|>"),
        _token("<|begin_of_image|>"),
        _token("<|end_of_image|>"),
    )
    rendered = _render(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "Look: "}, IMAGE]},
        ]
    )

    assert renderer.mm_token_type_id_map == {image: 1}
    assert image in rendered.token_ids
    for pos, token_id in enumerate(rendered.token_ids):
        if token_id == image:
            assert rendered.is_content[pos] is True
        elif token_id in (begin, end):
            assert rendered.is_content[pos] is False
        else:
            continue
        assert rendered.message_indices[pos] == 1
        assert rendered.sampled_mask[pos] is False


def test_bridge_tool_response_image_matches_full_render():
    _, renderer = _load()
    messages = _tool_turn([{"type": "text", "text": "Screen: "}, IMAGE])

    prompt_ids = _render(messages[:1], tools=TOOLS).token_ids
    full_ids = _render(messages, tools=TOOLS).token_ids
    # The sampled completion runs through the <|observation|> stop token.
    observation = full_ids.index(_token("<|observation|>"), len(prompt_ids))
    completion_ids = full_ids[len(prompt_ids) : observation + 1]

    bridged = renderer.bridge_to_next_turn(
        prompt_ids,
        completion_ids,
        messages[2:],
        tools=TOOLS,
        process_multimodal=False,
    )

    assert bridged is not None
    assert bridged.token_ids[: len(prompt_ids) + len(completion_ids)] == (
        prompt_ids + completion_ids
    )
    assert bridged.token_ids == full_ids


def test_bridge_user_image_matches_full_render():
    _, renderer = _load()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": [IMAGE, {"type": "text", "text": "And this?"}]},
    ]

    prompt_ids = _render(messages[:1]).token_ids
    full_ids = _render(messages).token_ids
    # The sampled completion runs through the <|user|> stop token.
    user = full_ids.index(_token("<|user|>"), len(prompt_ids))
    completion_ids = full_ids[len(prompt_ids) : user + 1]

    bridged = renderer.bridge_to_next_turn(
        prompt_ids, completion_ids, messages[2:], process_multimodal=False
    )

    assert bridged is not None
    assert bridged.token_ids == full_ids


def test_text_only_renders_with_default_process_multimodal():
    _, renderer = _load()
    messages = [{"role": "user", "content": "Hi"}]
    assert renderer.render(
        messages, add_generation_prompt=True
    ).token_ids == _reference(messages)


def test_images_require_logical_prompt():
    _, renderer = _load()
    messages = [{"role": "user", "content": [IMAGE]}]
    with pytest.raises(NotImplementedError, match="process_multimodal=False"):
        renderer.render(messages, add_generation_prompt=True)
    with pytest.raises(NotImplementedError, match="process_multimodal=False"):
        renderer.bridge_to_next_turn([], [], messages)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
    ],
    ids=["video", "audio"],
)
@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_video_and_audio_are_rejected(part, role):
    _, renderer = _load()
    messages = [{"role": role, "content": [part]}]
    with pytest.raises(NotImplementedError, match=part["type"]):
        _render(messages)
    with pytest.raises(NotImplementedError, match=part["type"]):
        renderer.bridge_to_next_turn([], [], messages, process_multimodal=False)


@pytest.mark.parametrize("process_multimodal", [False, True])
def test_assistant_images_are_rejected(process_multimodal):
    _, renderer = _load()
    messages = [{"role": "assistant", "content": [IMAGE]}]
    with pytest.raises(NotImplementedError, match="images in 'assistant' messages"):
        renderer.render(messages, process_multimodal=process_multimodal)
    with pytest.raises(NotImplementedError, match="images in 'assistant' messages"):
        renderer.bridge_to_next_turn(
            [], [], messages, process_multimodal=process_multimodal
        )


def _parallel_image_results(layout="separate"):
    image_b = {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}}
    messages = [
        {"role": "user", "content": [IMAGE]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                dict(_TOOL_CALL, id="call_a"),
                dict(_TOOL_CALL, id="call_b"),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "content": [{"type": "text", "text": "B: "}, image_b],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "content": [{"type": "text", "text": "A: "}, IMAGE, IMAGE],
        },
    ]
    if layout == "bundled":
        return messages[:2] + [
            {
                "role": "tool",
                "content": [
                    {"id": m["tool_call_id"], "output": m["content"]}
                    for m in messages[2:]
                ],
            }
        ]
    if layout == "mixed":
        messages[2] = {
            "role": "tool",
            "content": [{"id": "call_b", "output": messages[2]["content"]}],
        }
    return messages


@pytest.mark.parametrize("layout", ["separate", "bundled", "mixed"])
def test_bridge_declines_parallel_image_results(layout):
    _, renderer = _load()
    messages = _parallel_image_results(layout)
    prompt_ids = _render(messages[:1], tools=TOOLS).token_ids
    full_ids = _reference(messages, tools=TOOLS)
    observation = full_ids.index(_token("<|observation|>"), len(prompt_ids))
    completion_ids = full_ids[len(prompt_ids) : observation + 1]
    assert (
        renderer.bridge_to_next_turn(
            prompt_ids,
            completion_ids,
            messages[2:],
            tools=TOOLS,
            process_multimodal=False,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_source", ["render", "prebuilt", "bridge"])
@pytest.mark.parametrize("layout", ["separate", "bundled", "mixed"])
async def test_generate_keeps_parallel_images_in_prompt_order(prompt_source, layout):
    tok, renderer = _load()
    messages = _parallel_image_results(layout)
    original_messages = deepcopy(messages)
    prompt_ids = None
    if prompt_source == "prebuilt":
        prompt_ids = _reference(messages, tools=TOOLS)
    elif prompt_source == "bridge":
        prior_prompt = _render(messages, tools=TOOLS).token_ids
        completion = tok.encode("</think>Done.<|user|>", add_special_tokens=False)
        new_messages = [{"role": "user", "content": "Continue."}]
        bridged = renderer.bridge_to_next_turn(
            prior_prompt,
            completion,
            new_messages,
            tools=TOOLS,
            process_multimodal=False,
        )
        assert bridged is not None
        prompt_ids = bridged.token_ids
        messages += [{"role": "assistant", "content": "Done."}, *new_messages]

    requests = []

    def respond(request):
        import json

        body = json.loads(request.content)
        requests.append(body)
        expanded = []
        placeholders = []
        for token_id in body["token_ids"]:
            if token_id == _token("<|image|>"):
                placeholders.append({"offset": len(expanded), "length": 2})
                expanded.append(token_id)
            expanded.append(token_id)
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": expanded,
                "mm_placeholders": {"image": placeholders},
                "choices": [{"token_ids": [], "logprobs": {"content": []}}],
            },
        )

    async with AsyncOpenAI(
        api_key="replay",
        base_url="http://replay/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            prompt_ids=prompt_ids,
            process_multimodal=False,
        )

    assert requests[0]["token_ids"] == _reference(messages, tools=TOOLS)
    assert [part["url"] for part in requests[0]["content_parts"]] == [
        "https://example.com/a.png",
        "https://example.com/a.png",
        "https://example.com/a.png",
        "https://example.com/b.png",
    ]
    assert len(result["mm_placeholders"]["image"]) == 4
    assert messages[: len(original_messages)] == original_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize(
    "part",
    [
        IMAGE,
        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
    ],
)
async def test_generate_rejects_assistant_media_before_http(part, prebuilt):
    _, renderer = _load()
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": [part]},
        {"role": "user", "content": [IMAGE]},
    ]
    requests = []

    def respond(request):
        requests.append(request)
        raise AssertionError("Unsupported media must be rejected before HTTP")

    async with AsyncOpenAI(
        api_key="replay",
        base_url="http://replay/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        with pytest.raises(NotImplementedError, match="does not support"):
            await generate(
                client=client,
                renderer=renderer,
                model=MODEL,
                messages=messages,
                prompt_ids=[1, 2, 3] if prebuilt else None,
                process_multimodal=False,
            )
    assert requests == []


@pytest.mark.parametrize("model", GLM53_MODELS)
@pytest.mark.parametrize("layout", ["bundled", "mixed"])
@pytest.mark.parametrize(
    "case",
    [
        "ordered",
        "partial",
        "missing-result-id",
        "unknown-result-id",
        "duplicate-result-id",
        "missing-call-id",
        "duplicate-call-id",
        "tool-call-id-precedence",
        "no-calls",
    ],
)
def test_structured_tool_outputs_match_template(model, layout, case):
    tok, renderer = _load(model)
    calls = [dict(_TOOL_CALL, id=f"call_{i}") for i in range(3)]
    entries = [
        {"id": "call_2", "output": None},
        {"id": "call_0", "output": "FIRST_SENTINEL λ"},
        {"id": "call_1", "output": [{"type": "text", "text": "SECOND_SENTINEL"}]},
    ]
    if case == "partial":
        calls.insert(0, dict(_TOOL_CALL, id="not_returned"))
    elif case == "missing-result-id":
        entries[0].pop("id")
    elif case == "unknown-result-id":
        entries[0]["id"] = "unknown"
    elif case == "duplicate-result-id":
        entries[0]["id"] = "call_0"
    elif case == "missing-call-id":
        calls[0].pop("id")
    elif case == "duplicate-call-id":
        calls[0]["id"] = "call_1"
    elif case == "tool-call-id-precedence":
        for call in calls:
            call["tool_call_id"], call["id"] = call["id"], "ignored"
        for entry in entries:
            entry["tool_call_id"], entry["id"] = entry["id"], "ignored"
    elif case == "no-calls":
        calls = []
    messages = [
        {"role": "user", "content": "Run the tools."},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "content": entries},
    ]
    second_owner = 2
    if layout == "mixed":
        messages[2]["content"] = entries[:2]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": entries[2]["id"],
                "content": entries[2]["output"],
            }
        )
        if "tool_call_id" in entries[2]:
            messages[3]["tool_call_id"] = entries[2]["tool_call_id"]
        second_owner = 3
    original = deepcopy(messages)
    kwargs = {"process_multimodal": False} if model == MODEL else {}
    rendered = renderer.render(
        messages, tools=TOOLS, add_generation_prompt=True, **kwargs
    )
    assert rendered.token_ids == _reference(messages, tools=TOOLS, model=model)
    assert rendered.token_ids.count(_token("<tool_response>")) == 3
    for owner, text in [(2, "FIRST_SENTINEL λ"), (second_owner, "SECOND_SENTINEL")]:
        body_ids = [
            token_id
            for token_id, index, content in zip(
                rendered.token_ids, rendered.message_indices, rendered.is_content
            )
            if index == owner and content
        ]
        assert text in tok.decode(body_ids)
    observation = rendered.token_ids.index(tok.convert_tokens_to_ids("<|observation|>"))
    assert rendered.message_indices[observation] == 1
    assert rendered.sampled_mask[observation]
    assert messages == original
    prompt_ids = renderer.render(
        messages[:1], tools=TOOLS, add_generation_prompt=True, **kwargs
    ).token_ids
    completion_ids = rendered.token_ids[len(prompt_ids) : observation + 1]
    assert (
        renderer.bridge_to_next_turn(
            prompt_ids, completion_ids, messages[2:], tools=TOOLS, **kwargs
        )
        is None
    )


@pytest.mark.parametrize("model", GLM53_MODELS)
@pytest.mark.parametrize("output", ["", None, 0, False, {"key": "value"}, ["a", "b"]])
def test_structured_scalar_outputs_match_template(model, output):
    _, renderer = _load(model)
    messages = _tool_turn([{"id": "call_0", "output": output}])
    kwargs = {"process_multimodal": False} if model == MODEL else {}
    assert renderer.render(
        messages, tools=TOOLS, add_generation_prompt=True, **kwargs
    ).token_ids == _reference(messages, tools=TOOLS, model=model)


@pytest.mark.parametrize("model", GLM53_MODELS)
@pytest.mark.parametrize("bundled", [False, True])
@pytest.mark.parametrize("enveloped", [False, True])
def test_tool_reference_outputs_match_template(model, bundled, enveloped):
    tok, renderer = _load(model)
    specs = [
        {
            "name": "screenshot",
            "description": "Capture λ",
            "parameters": {},
            "defer_loading": True,
            "strict": True,
        },
        {"name": "other", "description": "Available", "parameters": {}},
    ]
    tools = (
        [{"type": "function", "function": spec} for spec in specs]
        if enveloped
        else specs
    )
    refs = [{"type": "tool_reference", "name": "screenshot"}]
    messages = _tool_turn([{"id": "call_0", "output": refs}] if bundled else refs)
    kwargs = {"process_multimodal": False} if model == MODEL else {}
    rendered = renderer.render(
        messages, tools=tools, add_generation_prompt=True, **kwargs
    )
    assert rendered.token_ids == _reference(messages, tools=tools, model=model)
    assert not any(
        sampled or content
        for index, sampled, content in zip(
            rendered.message_indices, rendered.sampled_mask, rendered.is_content
        )
        if index == 2
    )
    if not bundled:
        prompt = renderer.render(
            messages[:1], tools=tools, add_generation_prompt=True, **kwargs
        ).token_ids
        stop = rendered.token_ids.index(tok.convert_tokens_to_ids("<|observation|>"))
        bridged = renderer.bridge_to_next_turn(
            prompt,
            rendered.token_ids[len(prompt) : stop + 1],
            messages[2:],
            tools=tools,
            **kwargs,
        )
        assert bridged is not None
        assert bridged.token_ids == rendered.token_ids


@pytest.mark.parametrize(
    "part",
    [
        IMAGE,
        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
    ],
)
def test_nested_media_validation(part):
    _, renderer = _load()
    messages = _tool_turn([{"id": "call_0", "output": [part]}])
    if part == IMAGE:
        with pytest.raises(NotImplementedError, match="process_multimodal=False"):
            renderer.render(messages)
    else:
        with pytest.raises(NotImplementedError, match=part["type"]):
            _render(messages)
        with pytest.raises(NotImplementedError, match=part["type"]):
            renderer.bridge_to_next_turn([], [], messages[2:], process_multimodal=False)
