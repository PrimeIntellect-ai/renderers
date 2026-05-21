"""Backend-free end-to-end renderer flow tests.

These tests simulate the token-in/token-out control loop without launching
vLLM, SGLang, Transformers generation, or Tinker. They cover the glue between
``render_ids``, ``parse_response``, and ``bridge_to_next_turn`` so the examples
have a local parity check for the renderer-owned part of the stack.
"""

from __future__ import annotations


def test_renderer_owned_two_turn_flow_preserves_sampled_prefix():
    from renderers import create_renderer
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer("Qwen/Qwen3.5-9B")
    renderer = create_renderer(tokenizer, renderer="auto")

    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hello."},
    ]
    assistant = {"role": "assistant", "content": "Hello."}

    prompt_ids = renderer.render_ids(messages, add_generation_prompt=True)
    full_ids = renderer.render_ids(messages + [assistant])
    completion_ids = full_ids[len(prompt_ids) :]

    parsed = renderer.parse_response(completion_ids)
    assert "Hello" in parsed.content

    bridged = renderer.bridge_to_next_turn(
        prompt_ids,
        completion_ids,
        [{"role": "user", "content": "Now say bye."}],
    )
    assert bridged is not None
    bridged_ids = list(bridged.token_ids)
    expected_prefix = prompt_ids + completion_ids
    assert bridged_ids[: len(expected_prefix)] == expected_prefix


def test_default_renderer_fallback_keeps_raw_decoded_completion_prefix():
    """DefaultRenderer cannot bridge, so callers fall back to a full render.

    The fallback must use raw decoded completion bytes, not parse-normalized
    assistant structure. For round-tripping tokenizers, that preserves the
    sampled assistant prefix even though the bridge API correctly returns
    ``None``.
    """

    from renderers import create_renderer
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    renderer = create_renderer(tokenizer, renderer="default")

    messages = [{"role": "user", "content": "Say hello."}]
    assistant = {"role": "assistant", "content": "HELLO_SENTINEL"}
    new_messages = [{"role": "user", "content": "Now say bye."}]

    prompt_ids = renderer.render_ids(messages, add_generation_prompt=True)
    full_ids = renderer.render_ids(messages + [assistant])
    completion_ids = full_ids[len(prompt_ids) :]

    assert (
        renderer.bridge_to_next_turn(prompt_ids, completion_ids, new_messages) is None
    )

    raw_completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    fallback_ids = renderer.render_ids(
        messages + [{"role": "assistant", "content": raw_completion}] + new_messages,
        add_generation_prompt=True,
    )
    expected_prefix = prompt_ids + completion_ids
    assert fallback_ids[: len(expected_prefix)] == expected_prefix
