"""GLM-5.3 Flash image rendering.

GLM-5.3 Flash shares GLM-5.3's text template (covered by the shared parity
barrage) and adds images: each image part of a system, user or tool message
renders as ``<|begin_of_image|><|image|><|end_of_image|>``. The renderer emits
the logical prompt (``process_multimodal=False``, one ``<|image|>`` per image)
and leaves expansion to the inference engine, so these tests compare against
the template's untokenized output rather than the processor's expanded ids.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
import pytest
from openai import AsyncOpenAI

from renderers.client import generate

MODEL = "zai-org/GLM-5.3-Flash-BF16"

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
def _load():
    from renderers import create_renderer
    from renderers.base import load_tokenizer

    tok = load_tokenizer(MODEL)
    return tok, create_renderer(tok)


def _reference(messages, *, tools=None) -> list[int]:
    tok, _ = _load()
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


def _parallel_image_results():
    image_b = {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}}
    return [
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


def test_bridge_declines_parallel_image_results():
    _, renderer = _load()
    messages = _parallel_image_results()
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
async def test_generate_keeps_parallel_images_in_prompt_order(prompt_source):
    tok, renderer = _load()
    messages = _parallel_image_results()
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
    assert messages[2]["tool_call_id"] == "call_b"


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
