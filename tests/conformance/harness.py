"""No renderer internals: official encoder → token backend → public client."""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import httpx

from renderers import config_from_name, create_renderer
from renderers.base import load_tokenizer
from tests.conformance.encoder import deepseek_reference
from tests.parity import ModelCase
from tests.reference_rendering import _harmony_messages, render_reference


@cache
def tokenizer(model: str):
    pins = json.loads(Path(__file__).with_name("tokenizers.json").read_text())
    return load_tokenizer(model, revision=pins[model]["revision"])


def renderer(case: ModelCase, kwargs: dict[str, Any]):
    return create_renderer(
        tokenizer(case.model),
        config_from_name(case.resolved_renderer),
        chat_template_kwargs=kwargs or None,
    )


def reference(case: ModelCase, messages: list, kwargs: dict, **render_kwargs):
    options = dict(case.oracle_defaults) | kwargs
    options = {key: value for key, value in options.items() if value is not None}
    if (
        case.oracle == "deepseek-v4"
        and os.environ.get("RENDERERS_UPSTREAM_ENCODER") == "1"
    ):
        return deepseek_reference(
            tokenizer(case.model), messages, **options, **render_kwargs
        )
    return render_reference(
        tokenizer(case.model), messages, oracle=case.oracle, **options, **render_kwargs
    )


def unmodified_generation_reference(case: ModelCase, messages: list, kwargs: dict):
    if case.oracle != "harmony":
        return reference(case, messages, kwargs, add_generation_prompt=True)
    # Check the official encoder's default separately from the existing
    # forced-analysis profile; a shared oracle/renderer assumption can hide drift.
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Role,
        load_harmony_encoding,
    )

    options = {key: value for key, value in kwargs.items() if value is not None}
    conversation = Conversation.from_messages(
        _harmony_messages(
            messages,
            tools=None,
            reasoning_effort=options.get("reasoning_effort", "medium"),
            conversation_start_date=options.get("conversation_start_date"),
        )
    )
    return load_harmony_encoding(
        HarmonyEncodingName.HARMONY_GPT_OSS
    ).render_conversation_for_completion(conversation, next_turn_role=Role.ASSISTANT)


def assert_tokens(actual: list[int], expected: list[int], case: ModelCase):
    if actual == expected:
        return
    first = next(
        (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
        min(len(actual), len(expected)),
    )
    tok = tokenizer(case.model)
    window = slice(max(0, first - 12), first + 24)
    raise AssertionError(
        f"{case.model}: first differing token {first}; "
        f"lengths native={len(actual)}, reference={len(expected)}\n"
        f"native: {tok.decode(actual[window], skip_special_tokens=False)!r}\n"
        f"reference: {tok.decode(expected[window], skip_special_tokens=False)!r}"
    )


def assert_attribution(rendered):
    size = len(rendered.token_ids)
    for field in ("message_indices",):
        assert len(getattr(rendered, field)) == size, field
    assert len(rendered.sampled_mask) in {0, size}
    assert len(rendered.is_content) in {0, size}


class TokenBackend:
    """Script sampling only. Rendering and parsing remain the real implementation."""

    base_url = "http://conformance.invalid/v1"

    def __init__(self, prompt: list[int], completion: list[int], finish="stop"):
        self.prompt = prompt
        self.completion = completion
        self.finish = finish
        self.requests: list[dict] = []

    async def post(self, endpoint, *, body, cast_to, **kwargs):
        assert endpoint == "http://conformance.invalid/inference/v1/generate"
        assert body["token_ids"] == self.prompt
        assert body["sampling_params"]["skip_special_tokens"] is False
        self.requests.append(body)
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "choices": [
                        {
                            "token_ids": self.completion,
                            "finish_reason": self.finish,
                            "logprobs": {
                                "content": [
                                    {"token": f"token_id:{t}", "logprob": -0.5}
                                    for t in self.completion
                                ]
                            },
                        }
                    ]
                }
            ).encode(),
        )
