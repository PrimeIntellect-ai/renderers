"""NumPy fast-path coverage for the native PyO3 module."""

from __future__ import annotations

import os

import numpy as np
import pytest

from renderers import _native_router as router

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


@pytest.fixture(scope="module")
def qwen3_native():
    native = router.load_native()
    if native is None:
        pytest.skip("renderers_native not built; run `maturin develop`")

    try:
        from renderers.base import load_tokenizer

        tokenizer = load_tokenizer("Qwen/Qwen3-8B")
        tok_path = router.resolve_tokenizer_path(tokenizer)
    except Exception as exc:
        pytest.skip(f"could not resolve Qwen3 tokenizer: {exc}")
    if not os.path.exists(tok_path):
        pytest.skip(f"tokenizer.json missing on disk at {tok_path}")

    return native.Renderer.qwen3(tok_path)


def test_render_ids_np_matches_list_api(qwen3_native):
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hi."},
    ]

    ids = qwen3_native.render_ids_np(messages, add_generation_prompt=True)

    assert ids.dtype == np.uint32
    assert ids.tolist() == qwen3_native.render_ids(
        messages,
        add_generation_prompt=True,
    )


def test_parse_response_np_borrows_uint32_completion(qwen3_native):
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    assistant = {"role": "assistant", "content": "4"}
    prompt_ids = qwen3_native.render_ids_np(prompt, add_generation_prompt=True)
    full_ids = qwen3_native.render_ids_np(prompt + [assistant])
    completion_ids = full_ids[len(prompt_ids) :]

    parsed = qwen3_native.parse_response_np(completion_ids)

    assert parsed.content == "4"


def test_bridge_to_next_turn_np_matches_list_api(qwen3_native):
    prompt = [{"role": "user", "content": "Plan Saturday."}]
    assistant = {"role": "assistant", "content": "Start with breakfast."}
    new_messages = [{"role": "user", "content": "Add one museum."}]

    prompt_ids = qwen3_native.render_ids_np(prompt, add_generation_prompt=True)
    full_ids = qwen3_native.render_ids_np(prompt + [assistant])
    completion_ids = full_ids[len(prompt_ids) :]

    bridged_np = qwen3_native.bridge_to_next_turn_np(
        prompt_ids,
        completion_ids,
        new_messages,
    )
    bridged_list = qwen3_native.bridge_to_next_turn(
        prompt_ids.tolist(),
        completion_ids.tolist(),
        new_messages,
    )

    assert bridged_np is not None
    assert bridged_list is not None
    assert bridged_np.dtype == np.uint32
    assert bridged_np.tolist() == bridged_list.token_ids


def test_prepared_tools_match_raw_tools(qwen3_native):
    messages = [
        {"role": "system", "content": "You call tools when useful."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    prepared = qwen3_native.prepare_tools(TOOLS)

    raw_ids = qwen3_native.render_ids(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )
    prepared_ids = qwen3_native.render_ids(
        messages,
        tools=prepared,
        add_generation_prompt=True,
    )

    assert len(prepared) == 1
    assert prepared_ids == raw_ids


def test_render_batch_ids_matches_single_calls(qwen3_native):
    batch = [
        [{"role": "user", "content": "Say hi."}],
        [{"role": "user", "content": "Say bye."}],
    ]

    batch_ids = qwen3_native.render_batch_ids(batch, add_generation_prompt=True)

    assert batch_ids == [
        qwen3_native.render_ids(messages, add_generation_prompt=True)
        for messages in batch
    ]


def test_render_batch_ids_np_packed_matches_single_calls(qwen3_native):
    batch = [
        [{"role": "user", "content": "A"}],
        [{"role": "user", "content": "B"}],
        [{"role": "user", "content": "C"}],
    ]

    ids, offsets = qwen3_native.render_batch_ids_np_packed(
        batch,
        add_generation_prompt=True,
    )

    assert ids.dtype == np.uint32
    assert offsets.dtype == np.int64
    assert offsets.tolist()[0] == 0
    assert len(offsets) == len(batch) + 1
    unpacked = [
        ids[offsets[idx] : offsets[idx + 1]].tolist() for idx in range(len(batch))
    ]
    assert unpacked == [
        qwen3_native.render_ids(messages, add_generation_prompt=True)
        for messages in batch
    ]


def test_render_fast_ids_matches_dict_messages(qwen3_native):
    roles = ["system", "user", "assistant"]
    contents = ["You are concise.", "Say hi.", "Hi."]
    messages = [
        {"role": role, "content": content}
        for role, content in zip(roles, contents, strict=True)
    ]

    fast_ids = qwen3_native.render_fast_ids(
        roles,
        contents,
        add_generation_prompt=True,
    )
    fast_np = qwen3_native.render_fast_ids_np(
        roles,
        contents,
        add_generation_prompt=True,
    )
    regular_ids = qwen3_native.render_ids(
        messages,
        add_generation_prompt=True,
    )

    assert fast_ids == regular_ids
    assert fast_np.dtype == np.uint32
    assert fast_np.tolist() == regular_ids


def test_session_render_and_bridge_match_renderer(qwen3_native):
    prompt = [{"role": "user", "content": "Plan Saturday."}]
    assistant = {"role": "assistant", "content": "Start with breakfast."}
    new_messages = [{"role": "user", "content": "Add one museum."}]
    session = qwen3_native.new_session(prompt)

    session_prompt = session.render_ids(add_generation_prompt=True)
    full_ids = qwen3_native.render_ids(prompt + [assistant])
    completion_ids = full_ids[len(session_prompt) :]
    session_bridge = session.bridge_to_next_turn(completion_ids, new_messages)
    direct_bridge = qwen3_native.bridge_to_next_turn(
        session_prompt,
        completion_ids,
        new_messages,
    )

    assert session_prompt == qwen3_native.render_ids(
        prompt,
        add_generation_prompt=True,
    )
    assert session_bridge is not None
    assert direct_bridge is not None
    assert session_bridge.token_ids == direct_bridge.token_ids


def test_session_fork_preserves_prompt_state(qwen3_native):
    prompt = [{"role": "user", "content": "Plan Monday."}]
    assistant = {"role": "assistant", "content": "Start with tea."}
    new_messages = [{"role": "user", "content": "Add one errand."}]
    session = qwen3_native.new_session(prompt)
    session_prompt = session.render_ids(add_generation_prompt=True)
    forked = session.fork()

    full_ids = qwen3_native.render_ids(prompt + [assistant])
    completion_ids = full_ids[len(session_prompt) :]
    forked_bridge = forked.bridge_to_next_turn(completion_ids, new_messages)
    direct_bridge = qwen3_native.bridge_to_next_turn(
        session_prompt,
        completion_ids,
        new_messages,
    )

    assert forked_bridge is not None
    assert direct_bridge is not None
    assert forked_bridge.token_ids == direct_bridge.token_ids


def test_session_numpy_bridge_match_renderer(qwen3_native):
    prompt = [{"role": "user", "content": "Plan Sunday."}]
    assistant = {"role": "assistant", "content": "Start with a walk."}
    new_messages = [{"role": "user", "content": "Add coffee."}]
    session = qwen3_native.new_session(prompt)

    session_prompt = session.render_ids_np(add_generation_prompt=True)
    full_ids = qwen3_native.render_ids_np(prompt + [assistant])
    completion_ids = full_ids[len(session_prompt) :]
    session_bridge = session.bridge_to_next_turn_np(completion_ids, new_messages)
    direct_bridge = qwen3_native.bridge_to_next_turn_np(
        session_prompt,
        completion_ids,
        new_messages,
    )

    assert session_bridge is not None
    assert direct_bridge is not None
    assert session_bridge.dtype == np.uint32
    assert session_bridge.tolist() == direct_bridge.tolist()
