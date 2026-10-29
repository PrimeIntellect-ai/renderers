"""Production client + actual generation boundary + three successive turns."""

import asyncio

import numpy as np
import pytest

from renderers.client import generate
from tests.conformance.harness import (
    TokenBackend,
    assert_attribution,
    assert_tokens,
    reference,
    renderer,
    tokenizer,
)
from tests.parity import MODEL_CATALOG, kwarg_combinations

pytestmark = pytest.mark.model


def matrix():
    for case in MODEL_CATALOG:
        for kwargs in kwarg_combinations(case):
            # These knobs accept pre-serialized assistant bodies, rather than
            # the structured assistant messages returned by generate(). Their
            # complete matrix is exercised by test_finished_trace.
            if (
                kwargs.get("is_training")
                or kwargs.get("render_assistant_messages_raw")
                or kwargs.get("raw_last_assistant")
            ):
                continue
            yield pytest.param(case, kwargs, id=f"{case.model}/{kwargs!r}")


def canonical_turn(case, kwargs, native, history):
    prompt = reference(case, history, kwargs, add_generation_prompt=True)
    # Some templates require an open reasoning channel, others prefill a closed
    # channel. Select a message representable at that exact reference boundary.
    # The generation prompt is NEVER replaced with a historical-message prefix.
    for assistant in (
        {"role": "assistant", "content": "Four.", "reasoning_content": "Compute."},
        {"role": "assistant", "content": "Four."},
    ):
        finished = reference(case, history + [assistant], kwargs)
        if finished[: len(prompt)] == prompt:
            break
    else:
        if case.resolved_renderer == "deepseek-r1":
            # R1 deliberately removes ALL historical reasoning, so historical
            # serialization cannot produce a live completion. This is its
            # official <think> prefill + vLLM's deepseek_r1 output grammar.
            completion = tokenizer(case.model).encode(
                "Compute.</think>Four.<｜end▁of▁sentence｜>", add_special_tokens=False
            )
            return prompt, completion, assistant, "Compute.", "Four."
        if case.resolved_renderer == "gpt-oss":
            # Harmony's live prefix is analysis; a finished content-only
            # history is final. Use the official Harmony channel transition.
            completion = tokenizer(case.model).encode(
                "Compute.<|end|><|start|>assistant<|channel|>final<|message|>Four.<|return|>",
                add_special_tokens=False,
            )
            return prompt, completion, assistant, "Compute.", "Four."
        closed_channels = ("</think>", "</think:opensource>", "<channel|>")
        if (
            tokenizer(case.model)
            .decode(prompt, skip_special_tokens=False)
            .rstrip()
            .endswith(closed_channels)
        ):
            assistant = {"role": "assistant", "content": "Four."}
            finished = reference(case, history + [assistant], kwargs)
            text = tokenizer(case.model).decode(finished, skip_special_tokens=False)
            body = text[text.rindex("Four.") :]
            completion = tokenizer(case.model).encode(body, add_special_tokens=False)
            return finish_turn(
                case,
                kwargs,
                native,
                history,
                prompt,
                finished,
                completion,
                assistant,
                None,
                "Four.",
            )
        assert_tokens(finished[: len(prompt)], prompt, case)
        raise AssertionError(
            "No canonical completion at the official generation boundary"
        )

    completion = finished[len(prompt) :]
    reason = assistant.get("reasoning_content")
    if (
        reference(case, history + [{"role": "assistant", "content": "Four."}], kwargs)
        == finished
    ):
        reason = None  # The reference template ignores this metadata field.
    content = "Four."
    if case.resolved_renderer == "glm-4.5":
        reason = "\n" + reason if reason else None
        content = "\n" + content
    return finish_turn(
        case,
        kwargs,
        native,
        history,
        prompt,
        finished,
        completion,
        assistant,
        reason,
        content,
    )


def finish_turn(
    case,
    kwargs,
    native,
    history,
    prompt,
    finished,
    completion,
    assistant,
    reason,
    content,
):
    stops = set(native.get_stop_token_ids())
    end = next(
        (i + 1 for i in range(len(completion) - 1, -1, -1) if completion[i] in stops),
        None,
    )
    if end is not None:
        completion = completion[:end]
    else:
        extended = reference(
            case, history + [assistant, {"role": "user", "content": "Next."}], kwargs
        )
        # A new query may drop historical reasoning, so its prefix need not
        # equal the live turn. GLM samples the next user role as its stop.
        stop = max(i for i, token in enumerate(extended) if token in stops)
        completion += extended[stop : stop + 1]
    return prompt, completion, assistant, reason, content


def expected_extension(case, kwargs, history, assistant, new, completion, native):
    following = reference(
        case, history + [assistant] + new, kwargs, add_generation_prompt=True
    )
    # Perturb only the previous assistant body. The common suffix locates the
    # actual turn-close boundary IN THE FOLLOWING REFERENCE RENDER, even when
    # the template drops historical reasoning upon arrival of a new query.
    probe = dict(assistant, content="BOUNDARY_PROBE_endXYZ")
    other = reference(case, history + [probe] + new, kwargs, add_generation_prompt=True)
    common = 0
    while (
        common < min(len(following), len(other))
        and following[-common - 1] == other[-common - 1]
    ):
        common += 1
    suffix = following[len(following) - common :]
    stops = set(native.get_stop_token_ids())
    if case.resolved_renderer == "gpt-oss":
        # Harmony history closes a message with end, whereas sampled final
        # answers close the turn with return. The reference owns both forms.
        stops.add(tokenizer(case.model).convert_tokens_to_ids("<|end|>"))
    stop = next(i for i, token in enumerate(suffix) if token in stops)
    return suffix[stop + 1 :]


@pytest.mark.parametrize("case,kwargs", tuple(matrix()))
def test_three_turn_client_rollout(case, kwargs):
    native = renderer(case, kwargs)
    history = [{"role": "user", "content": "What is two plus two?"}]
    next_prompt = None
    for turn in range(3):
        prompt, completion, assistant, reasoning, content = canonical_turn(
            case, kwargs, native, history
        )
        if next_prompt is not None:
            prompt = next_prompt.token_ids
        backend = TokenBackend(prompt, completion)
        result = asyncio.run(
            generate(
                client=backend,
                renderer=native,
                messages=history,
                model=case.model,
                prompt_ids=next_prompt.token_ids if next_prompt is not None else None,
                prompt_attribution=next_prompt,
                max_prompt_len=1_000_000,
            )
        )
        assert result["prompt_ids"] == prompt
        assert result["completion_ids"] == completion
        np.testing.assert_array_equal(
            result["completion_logprobs"], [-0.5] * len(completion)
        )
        assert result["content"] == content
        # Empty channel versus absent channel has the same OpenAI projection;
        # do not normalize ANY nonempty reasoning or content whitespace.
        assert (result["reasoning_content"] or None) == reasoning
        assert result["reasoning_complete"] is (
            None if case.resolved_renderer == "default" else True
        )
        assert not result["tool_calls"]
        assert len(backend.requests) == 1
        assert_attribution(result["prompt_attribution"])
        # Reuse the PUBLIC parser result, just as a real conversation does.
        # Reusing the fixture message here would hide parser/template drift.
        assistant = {"role": "assistant", "content": result["content"]}
        if result["reasoning_content"] is not None:
            assistant["reasoning_content"] = result["reasoning_content"]
        new = [{"role": "user", "content": f"Next query {turn}."}]
        bridge = native.bridge_to_next_turn(prompt, completion, new)
        extension = expected_extension(
            case, kwargs, history, assistant, new, completion, native
        )
        history += [assistant] + new
        expected = reference(case, history, kwargs, add_generation_prompt=True)
        if bridge is not None:
            # For this canonical stream no serialization normalization is
            # needed. Both prefix preservation AND complete template parity
            # must hold, not just a plausible-looking decoded suffix.
            assert (
                bridge.token_ids[: len(prompt) + len(completion)] == prompt + completion
            )
            assert_tokens(bridge.token_ids, prompt + completion + extension, case)
            assert_attribution(bridge)
            assert not any(bridge.sampled_mask)
            next_prompt = bridge
        else:
            next_prompt = native.render(history, add_generation_prompt=True)
            assert_tokens(next_prompt.token_ids, expected, case)


@pytest.mark.parametrize("case,kwargs", tuple(matrix()))
def test_budget_stop_preserves_sampled_prefix(case, kwargs):
    native = renderer(case, kwargs)
    history = [{"role": "user", "content": "What is two plus two?"}]
    prompt, completed, _, _, _ = canonical_turn(case, kwargs, native, history)
    sampled = completed[: max(1, len(completed) // 2)]
    backend = TokenBackend(prompt, sampled, finish="length")
    result = asyncio.run(
        generate(
            client=backend,
            renderer=native,
            messages=history,
            model=case.model,
            max_prompt_len=1_000_000,
        )
    )
    assert result["prompt_ids"] == prompt
    assert result["completion_ids"] == sampled
    np.testing.assert_array_equal(result["completion_logprobs"], [-0.5] * len(sampled))
    assert result["finish_reason"] == "length"
    bridge = native.bridge_to_next_turn(
        prompt, sampled, [{"role": "user", "content": "Continue."}]
    )
    if bridge is not None:
        assert bridge.token_ids[: len(prompt) + len(sampled)] == prompt + sampled
        assert_attribution(bridge)
        assert not any(bridge.sampled_mask)
