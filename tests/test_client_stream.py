"""``generate_stream`` against a scripted engine speaking real SSE.

The client under test is a real ``AsyncOpenAI`` over an in-memory transport,
so request building, SSE framing, error mapping and connection closing all
run through the SDK exactly as they do against vLLM.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing

import httpx
import numpy as np
import openai
import pytest
from openai import AsyncOpenAI

from renderers import MalformedGenerateResponseError, OverlongPromptError
from renderers.base import (
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
)
from renderers.client import generate, generate_stream
from renderers.streaming import ResponseStream, TextDelta
from tests.test_response_stream import (
    END_THINK,
    END_TOOL,
    EOS,
    REASONING_FIRST,
    TOOL,
    ByteTokenizer,
    b,
)

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "echo"}}]

# Prompt-opened reasoning, an answer, then a tool call.
COMPLETION = [
    *b(" weigh 🙂 options \n"),
    END_THINK,
    *b("\n\nI will echo. "),
    TOOL,
    *b('{"name": "echo", "arguments": {"text": "hi"}}'),
    END_TOOL,
    EOS,
]
REASONING = "weigh 🙂 options"
CONTENT = "I will echo."


class StreamRenderer:
    """A small real grammar: reasoning until ``</think>``, then content, then tools."""

    supports_tools = True

    def __init__(self):
        self.stream_prompts: list[list[int] | None] = []

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        return RenderedTokens(
            token_ids=[1, 2, 3],
            message_indices=[0, 0, -1],
            sampled_mask=[False, False, False],
            is_content=[False, True, False],
            message_roles=["user"],
        )

    def get_stop_token_ids(self):
        return [EOS]

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None):
        decode = ByteTokenizer().decode
        ids = token_ids[: token_ids.index(EOS)] if EOS in token_ids else token_ids
        # The prompt opens reasoning, so without a close it is all reasoning.
        cut = ids.index(END_THINK) if END_THINK in ids else len(ids)
        reasoning, ids = decode(ids[:cut]), ids[cut + 1 :]
        calls = []
        if TOOL in ids:
            start = ids.index(TOOL)
            raw = decode(ids[start + 1 : ids.index(END_TOOL)])
            ids = ids[:start]
            call = json.loads(raw)
            calls.append(
                ParsedToolCall(
                    raw=raw,
                    name=call["name"],
                    arguments=call["arguments"],
                    status=ToolCallParseStatus.OK,
                )
            )
        return ParsedResponse(
            content=decode(ids),
            reasoning_content=reasoning,
            tool_calls=calls,
            reasoning_complete=END_THINK in token_ids,
        )

    def response_stream(self, *, prompt_ids=None):
        self.stream_prompts.append(prompt_ids)
        return ResponseStream(ByteTokenizer(), REASONING_FIRST, reasoning=True)


class WholeResponseRenderer(StreamRenderer):
    """A renderer that predates streaming support."""

    response_stream = None


class DecliningRenderer(StreamRenderer):
    """A renderer whose grammar cannot be streamed for this tokenizer."""

    def response_stream(self, *, prompt_ids=None):
        return None


def logprob(token_id: int, position: int, head: int | None) -> dict:
    entry = {"token": f"token_id:{token_id}", "logprob": -0.01 * (position + 1)}
    if head is not None:
        entry["top_logprobs"] = [
            {"token": entry["token"], "logprob": entry["logprob"]},
            *(
                {"token": f"token_id:{1000 + rank}", "logprob": -5.0 - rank}
                for rank in range(head + 1)
            ),
        ]
    return entry


def stream_chunks(
    token_ids=COMPLETION,
    *,
    per_chunk=1,
    head=None,
    mask=False,
    finish_reason="stop",
    extras=None,
    usage=None,
) -> list[dict]:
    """The chunks vLLM's token stream sends for one completion."""
    chunks = []
    for start in range(0, len(token_ids), per_chunk):
        ids = token_ids[start : start + per_chunk]
        choice = {
            "index": 0,
            "token_ids": ids,
            "logprobs": {
                "content": [
                    logprob(token_id, start + offset, head)
                    for offset, token_id in enumerate(ids)
                ]
            },
            "finish_reason": None,
            "routed_experts": None,
            "sampling_mask": [[token_id, 1000] for token_id in ids] if mask else None,
            "weight_versions": None,
        }
        chunks.append({"request_id": "gen-stream", "choices": [choice]})
    chunks[-1]["choices"][0].update({"finish_reason": finish_reason, **(extras or {})})
    if usage is not None:
        chunks.append({"request_id": "gen-stream", "choices": [], "usage": usage})
    return chunks


def complete_response(chunks: list[dict]) -> dict:
    """The non-streaming response carrying the same completion."""
    choice = {"index": 0, "token_ids": [], "logprobs": {"content": []}}
    response = {"request_id": "gen-stream", "choices": [choice]}
    for chunk in chunks:
        if chunk.get("usage") is not None:
            response["usage"] = chunk["usage"]
        for part in chunk["choices"]:
            choice["token_ids"] += part["token_ids"]
            choice["logprobs"]["content"] += part["logprobs"]["content"]
            if part["sampling_mask"] is not None:
                choice.setdefault("sampling_mask", []).extend(part["sampling_mask"])
            for key in ("finish_reason", "routed_experts", "weight_versions"):
                if part.get(key) is not None:
                    choice[key] = part[key]
    return response


def sse(events: list) -> bytes:
    return b"".join(
        b"data: " + (e if isinstance(e, bytes) else json.dumps(e).encode()) + b"\n\n"
        for e in [*events, b"[DONE]"]
    )


class Engine:
    """An in-memory generate endpoint. Serves ``chunks`` streamed or whole."""

    def __init__(self, chunks, *, piece=None, status=200, raw=None):
        self.requests: list[httpx.Request] = []
        self.closed = False
        self.served = 0
        engine = self

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                payload = raw if raw is not None else sse(chunks)
                size = piece or len(payload)
                for start in range(0, len(payload), size):
                    engine.served += 1
                    yield payload[start : start + size]
                    await asyncio.sleep(0)

            async def aclose(self):
                engine.closed = True

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "rejected"}})
            if json.loads(request.content).get("stream"):
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=Body()
                )
            return httpx.Response(200, json=complete_response(chunks))

        self.client = AsyncOpenAI(
            base_url="http://engine:8000/v1",
            api_key="unused",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def run_stream(engine, renderer=None, **kwargs):
    """Drain ``generate_stream``; return (text deltas, result dict)."""

    async def drain():
        events = [
            event
            async for event in generate_stream(
                client=engine.client,
                renderer=renderer or StreamRenderer(),
                messages=MESSAGES,
                model="test-model",
                tools=TOOLS,
                **{"max_prompt_len": 100, **kwargs},
            )
        ]
        return events

    events = asyncio.run(drain())
    assert isinstance(events[-1], dict), "the result must be the last item"
    assert all(isinstance(event, TextDelta) for event in events[:-1])
    return events[:-1], events[-1]


def run_generate(engine, renderer=None, **kwargs):
    return asyncio.run(
        generate(
            client=engine.client,
            renderer=renderer or StreamRenderer(),
            messages=MESSAGES,
            model="test-model",
            tools=TOOLS,
            **{"max_prompt_len": 100, **kwargs},
        )
    )


def text(deltas, channel):
    return "".join(d.text for d in deltas if d.channel == channel)


def assert_same_result(streamed: dict, whole: dict) -> None:
    assert streamed.keys() == whole.keys()
    for key, expected in whole.items():
        actual = streamed[key]
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype, key
            np.testing.assert_array_equal(actual, expected, err_msg=key)
        elif key == "sampler_support" and expected is not None:
            assert actual.keys() == expected.keys()
            for name, value in expected.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(actual[name], value, err_msg=name)
                else:
                    assert actual[name] == value, name
        else:
            assert actual == expected, key


@pytest.mark.parametrize("per_chunk", [1, 3, len(COMPLETION)])
@pytest.mark.parametrize("head, mask", [(None, False), (2, False), (None, True)])
def test_streamed_result_equals_the_non_streaming_result(per_chunk, head, mask):
    """What gets recorded and trained on does not depend on the transport."""
    chunks = stream_chunks(
        per_chunk=per_chunk,
        head=head,
        mask=mask,
        extras={
            "routed_experts": "bnVtcHk=",
            "weight_versions": [{"version": 4, "start": 0, "end": len(COMPLETION)}],
        },
        usage={"prompt_tokens": 3, "completion_tokens": len(COMPLETION)},
    )
    sampling = {"temperature": 0.7, **({"logprobs": head} if head else {})}
    _, streamed = run_stream(Engine(chunks), sampling_params=sampling)
    whole = run_generate(Engine(chunks), sampling_params=sampling)

    assert_same_result(streamed, whole)
    # Spot-check the payload itself, so two equally empty results cannot pass.
    assert streamed["completion_ids"] == COMPLETION
    assert streamed["prompt_ids"] == [1, 2, 3]
    assert len(streamed["completion_logprobs"]) == len(COMPLETION)
    assert streamed["completion_logprobs"][2] == np.float32(-0.03)
    assert streamed["routed_experts"] == "bnVtcHk="
    assert streamed["weight_versions"][0]["version"] == 4
    assert streamed["usage"]["completion_tokens"] == len(COMPLETION)
    assert streamed["request_id"] == "gen-stream"
    assert (streamed["sampler_support"] is not None) == (mask or head is not None)
    if mask:
        assert streamed["sampler_support"]["replayed"]
        assert streamed["sampler_support"]["counts"].tolist() == [2] * len(COMPLETION)
    assert streamed["prompt_attribution"].is_content == [False, True, False]


def test_text_streams_before_the_result_and_tool_calls_arrive_whole():
    deltas, result = run_stream(Engine(stream_chunks()))

    assert text(deltas, "reasoning") == REASONING == result["reasoning_content"].strip()
    assert text(deltas, "content") == CONTENT == result["content"].strip()
    channels = [d.channel for d in deltas]
    assert channels == sorted(channels, key=["reasoning", "content"].index)
    # One token per chunk: text arrives in many pieces, not one final lump.
    assert len(deltas) > 20
    assert all("echo" not in d.text or d.text in CONTENT for d in deltas)
    assert not any("{" in d.text or "tool_call" in d.text for d in deltas)

    assert [(c.name, c.arguments) for c in result["tool_calls"]] == [
        ("echo", {"text": "hi"})
    ]
    # A clean tool call promotes the engine's "stop", as in generate().
    assert result["finish_reason"] == "tool_calls"


def test_deltas_are_yielded_while_the_engine_is_still_generating():
    engine = Engine(stream_chunks(), piece=40)
    seen_at: list[int] = []

    async def drain():
        async for event in generate_stream(
            client=engine.client,
            renderer=StreamRenderer(),
            messages=MESSAGES,
            model="test-model",
            max_prompt_len=100,
        ):
            if isinstance(event, TextDelta):
                seen_at.append(engine.served)

    asyncio.run(drain())
    total = engine.served
    assert total > 300
    # The first text shows up within the first few network reads, and text
    # keeps arriving across the body instead of after it.
    assert seen_at[0] <= 20
    assert len({*seen_at}) > 20
    assert seen_at[len(seen_at) // 2] < total * 0.6


@pytest.mark.parametrize("piece", [1, 7, 64, 1000])
def test_sse_framing_is_independent_of_network_chunking(piece):
    whole_deltas, whole = run_stream(Engine(stream_chunks()))
    deltas, result = run_stream(Engine(stream_chunks(), piece=piece))
    assert deltas == whole_deltas
    assert_same_result(result, whole)


@pytest.mark.parametrize("per_chunk", [1, 2, 5, len(COMPLETION)])
def test_engine_chunk_size_does_not_change_the_text(per_chunk):
    deltas, _ = run_stream(Engine(stream_chunks(per_chunk=per_chunk)))
    assert text(deltas, "reasoning") == REASONING
    assert text(deltas, "content") == CONTENT


def test_request_asks_for_a_token_stream_with_native_logprobs():
    engine = Engine(stream_chunks(head=2))
    run_stream(
        engine,
        sampling_params={"logprobs": 2, "max_tokens": 9, "extra_args": {"custom": 1}},
        cache_salt="salt",
        priority=3,
        extra_headers={"x-session": "abc"},
    )
    request = engine.requests[-1]
    assert str(request.url) == "http://engine:8000/inference/v1/generate"
    assert request.headers["x-session"] == "abc"
    body = engine.body
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["token_ids"] == [1, 2, 3]
    assert body["cache_salt"] == "salt" and body["priority"] == 3
    sampling = body["sampling_params"]
    assert sampling["stop_token_ids"] == [EOS]
    assert sampling["max_tokens"] == 9
    assert sampling["skip_special_tokens"] is False
    # Two spare candidates for the head, as in generate().
    assert sampling["logprobs"] == 4
    # The engine rejects compact logprobs on a streamed request.
    assert sampling["extra_args"] == {"custom": 1}


def test_non_streaming_request_is_unchanged_by_the_streaming_option():
    engine = Engine(stream_chunks(head=2))
    run_generate(engine, sampling_params={"logprobs": 2})
    body = engine.body
    assert "stream" not in body and "stream_options" not in body
    assert body["sampling_params"]["extra_args"] == {"prl_compact_logprobs": True}


def test_first_chunk_prompt_ids_and_placeholders_reach_the_result():
    chunks = stream_chunks()
    chunks[0]["prompt_token_ids"] = [1, 2, 2, 3]
    chunks[0]["mm_placeholders"] = {"image": [{"offset": 1, "length": 2}]}
    _, result = run_stream(Engine(chunks))
    assert result["prompt_ids"] == [1, 2, 2, 3]
    assert result["mm_placeholders"] == {"image": [{"offset": 1, "length": 2}]}


def test_other_choices_and_empty_chunks_are_ignored():
    chunks = stream_chunks(per_chunk=4)
    stray = json.loads(json.dumps(chunks[0]))
    stray["choices"][0].update({"index": 1, "token_ids": [9, 9], "finish_reason": "x"})
    chunks.insert(1, stray)
    chunks.insert(2, {"request_id": "gen-stream", "choices": []})
    deltas, result = run_stream(Engine(chunks))
    assert result["completion_ids"] == COMPLETION
    assert text(deltas, "content") == CONTENT


def test_splitter_starts_from_the_prompt_that_was_sent():
    renderer = StreamRenderer()
    run_stream(Engine(stream_chunks()), renderer=renderer)
    run_stream(Engine(stream_chunks()), renderer=renderer, prompt_ids=[5, 6])
    assert renderer.stream_prompts == [[1, 2, 3], [5, 6]]


@pytest.mark.parametrize("renderer_cls", [WholeResponseRenderer, DecliningRenderer])
def test_renderer_without_a_stream_delivers_its_text_with_the_result(renderer_cls):
    engine = Engine(stream_chunks())
    deltas, result = run_stream(engine, renderer=renderer_cls())
    assert deltas == [TextDelta("reasoning", REASONING), TextDelta("content", CONTENT)]
    assert_same_result(result, run_generate(Engine(stream_chunks())))


def test_length_finish_inside_reasoning_streams_what_was_generated():
    cut = COMPLETION[:8]
    deltas, result = run_stream(Engine(stream_chunks(cut, finish_reason="length")))
    assert result["finish_reason"] == "length"
    assert result["completion_ids"] == cut
    assert result["reasoning_complete"] is False
    # The emoji's first byte is not text yet; the final parse reports it.
    assert deltas[:-1] == [TextDelta("reasoning", c) for c in "weigh"]
    assert deltas[-1] == TextDelta("reasoning", " \ufffd")
    assert text(deltas, "reasoning") == result["reasoning_content"].strip()


def test_text_withheld_at_the_end_arrives_before_the_result():
    # Ends on half a character: only the final parse can say what it decodes to.
    cut = [*b("ok"), *b("🙂")[:2]]
    deltas, result = run_stream(Engine(stream_chunks(cut, finish_reason="length")))
    assert text(deltas, "reasoning") == "ok\ufffd" == result["reasoning_content"]
    assert deltas[-1] == TextDelta("reasoning", "\ufffd")


def test_engine_error_mid_stream_raises_after_the_text_already_sent():
    chunks = stream_chunks()[:12]
    engine = Engine(None, raw=sse([*chunks, {"error": {"message": "engine died"}}]))
    seen: list[TextDelta] = []

    async def drain():
        async for event in generate_stream(
            client=engine.client,
            renderer=StreamRenderer(),
            messages=MESSAGES,
            model="test-model",
            max_prompt_len=100,
        ):
            seen.append(event)

    with pytest.raises(openai.APIError, match="engine died"):
        asyncio.run(drain())
    assert text(seen, "reasoning") == "weigh 🙂"
    assert all(isinstance(event, TextDelta) for event in seen)
    assert engine.closed


def test_http_rejection_raises_before_any_event():
    engine = Engine(stream_chunks(), status=400)
    with pytest.raises(openai.BadRequestError, match="rejected"):
        run_stream(engine)


def test_overlong_prompt_is_rejected_before_any_request():
    engine = Engine(stream_chunks())
    with pytest.raises(OverlongPromptError):
        run_stream(engine, max_prompt_len=2)
    assert engine.requests == []


def test_tools_are_rejected_for_a_renderer_without_tool_support():
    class NoTools(StreamRenderer):
        supports_tools = False

    engine = Engine(stream_chunks())
    with pytest.raises(ValueError, match="does not support tools"):
        run_stream(engine, renderer=NoTools())
    assert engine.requests == []


def _without_finish(chunks):
    chunks[-1]["choices"][0]["finish_reason"] = None
    return chunks


def _without_logprobs(chunks):
    chunks[3]["choices"][0]["logprobs"] = None
    return chunks


def _with_wrong_logprob_token(chunks):
    chunks[3]["choices"][0]["logprobs"]["content"][0]["token"] = "token_id:1"
    return chunks


def _with_compact_logprobs(chunks):
    chunks[3]["choices"][0]["compact_logprobs"] = {"data": "", "offsets": [0]}
    return chunks


def _with_packed_mask(chunks):
    chunks[3]["choices"][0]["sampling_mask"] = {"ids": {}, "counts": {}}
    return chunks


def _with_short_mask(chunks):
    for chunk in chunks:
        chunk["choices"][0]["sampling_mask"] = [[chunk["choices"][0]["token_ids"][0]]]
    chunks[3]["choices"][0]["sampling_mask"] = None
    return chunks


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (_without_finish, "without a finish reason"),
        (_without_logprobs, "does not match logprob count"),
        (_with_wrong_logprob_token, "token must be"),
        (_with_compact_logprobs, "Compact logprobs cannot be merged"),
        (_with_packed_mask, "per-token rows"),
        (_with_short_mask, "Sampling mask token count mismatch"),
    ],
)
def test_unusable_sampling_evidence_is_rejected_not_returned(corrupt, message):
    engine = Engine(corrupt(stream_chunks()))
    with pytest.raises(MalformedGenerateResponseError, match=message):
        run_stream(engine)
    assert engine.closed


def test_stream_cut_off_by_the_network_is_not_a_result():
    payload = sse(stream_chunks())
    engine = Engine(
        None, raw=payload[: len(payload) // 2].rsplit(b"\n\n", 1)[0] + b"\n\n"
    )
    with pytest.raises(MalformedGenerateResponseError, match="without a finish reason"):
        run_stream(engine)


def test_closing_the_iterator_early_closes_the_connection():
    engine = Engine(stream_chunks(), piece=40)

    async def take_first():
        async with aclosing(
            generate_stream(
                client=engine.client,
                renderer=StreamRenderer(),
                messages=MESSAGES,
                model="test-model",
                max_prompt_len=100,
            )
        ) as events:
            async for event in events:
                break
        # Closed by the time the iterator is, not at some later cleanup.
        return event, engine.closed, engine.served

    first, closed, served = asyncio.run(take_first())
    assert first == TextDelta("reasoning", "w")
    assert closed
    # The body was abandoned, not drained.
    assert served < 30
