"""Replay only the inference HTTP boundary; exercise the real client stack."""

import json

import httpx
from openai import AsyncOpenAI


def replay_client(completions, requests):
    """Replay sampled IDs, logprobs, and masks without mocking renderer code."""
    responses = iter(completions)

    def respond(request):
        assert request.url.path == "/inference/v1/generate"
        body = json.loads(request.content)
        requests.append(body)
        ids, finish_reason = next(responses)
        return httpx.Response(
            200,
            json={
                "request_id": f"replay-{len(requests)}",
                "choices": [
                    {
                        "token_ids": ids,
                        "finish_reason": finish_reason,
                        "logprobs": {
                            "content": [
                                {"token": f"token_id:{tid}", "logprob": -0.25}
                                for tid in ids
                            ]
                        },
                        "sampling_mask": [[tid] for tid in ids],
                    }
                ],
            },
        )

    return AsyncOpenAI(
        api_key="replay",
        base_url="http://replay/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )


def assert_evidence(result, ids):
    assert result["completion_ids"] == ids
    assert result["completion_logprobs"].tolist() == [-0.25] * len(ids)
    assert result["sampling_mask"] == [[tid] for tid in ids]
