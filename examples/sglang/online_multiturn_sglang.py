#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "renderers>=0.1.6",
#   "transformers>=5.3.0",
#   "httpx>=0.27",
#   "openai-harmony==0.0.4",
#   "tiktoken",
#   "jinja2",
#   "numpy",
# ]
# ///
"""SGLang online generation from renderer-owned prompt token IDs.

Mirrors `multiturn_generate_sglang.py` but talks to an already-running SGLang
HTTP server over `/generate` instead of an in-process `sgl.Engine`. The
renderer owns chat templating and parsing; SGLang only does token-in,
token-out.

Streaming is intentionally not supported: `parse_response` and
`bridge_to_next_turn` both need the complete `completion_ids`.

Launch a server first, e.g.

    sglang serve --model-path Qwen/Qwen3.5-4B \\
        --host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --trust-remote-code

then, from a source checkout,

    uv run python examples/sglang/online_multiturn_sglang.py \\
        --base-url http://localhost:30000 --model Qwen/Qwen3.5-4B

The PEP 723 `uv run --script` form requires a published `renderers` package
that satisfies the script header.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx
from renderers.base import Renderer
from renderers.configs import Qwen35RendererConfig
from renderers.gpt_oss import GptOssRenderer
from renderers.qwen35 import Qwen35Renderer
from transformers import AutoTokenizer


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def make_renderer(model: str, enable_thinking: bool | None) -> Renderer:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    if model.startswith("Qwen/Qwen3.5-"):
        return Qwen35Renderer(
            tokenizer, Qwen35RendererConfig(enable_thinking=enable_thinking)
        )
    if model == "openai/gpt-oss-20b":
        return GptOssRenderer(tokenizer)
    raise ValueError(f"unsupported demo model: {model}")


def completion_ids(output: dict, prompt_ids: list[int]) -> list[int]:
    ids = list(output.get("output_ids") or output.get("token_ids") or [])
    if not ids:
        raise RuntimeError("SGLang did not return completion token IDs")
    # Match offline recipe: strip prefix only if SGLang echoed the prompt back.
    return ids[len(prompt_ids) :] if ids[: len(prompt_ids)] == prompt_ids else ids


async def generate_sglang(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    renderer: Renderer,
    prompt_ids: list[int],
    max_new_tokens: int,
    extra_key: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop_token_ids": renderer.get_stop_token_ids(),
            "skip_special_tokens": False,
            "no_stop_trim": True,
        },
        "stream": False,
    }
    if extra_key is not None:
        body["extra_key"] = extra_key
    response = await client.post(f"{base_url.rstrip('/')}/generate", json=body)
    response.raise_for_status()
    return response.json()


def print_parsed(label: str, turn: str, parsed) -> None:
    print(f"\n[{label}] {turn}")
    if parsed.reasoning_content:
        print(f"reasoning: {parsed.reasoning_content[:240]}")
    for tc in parsed.tool_calls:
        # ``parse_response`` returns ``ParsedToolCall`` dataclasses, not dicts.
        print(f"tool_call: {tc.name}({tc.arguments}) [{tc.status.value}]")
    if parsed.content:
        print(f"content: {parsed.content}")


async def run_one(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    enable_thinking: bool | None,
    max_new_tokens: int,
) -> None:
    label = (
        model
        if enable_thinking is None
        else f"{model} enable_thinking={enable_thinking}"
    )
    print(f"\n=== {label} ===")

    renderer = make_renderer(model, enable_thinking)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a concise tool-using assistant."},
        {
            "role": "user",
            "content": "Use the multiply tool for 17 * 23, then summarize.",
        },
    ]

    # Turn 1: render locally, send token IDs. SGLang never sees messages.
    prompt_ids = renderer.render_ids(messages, tools=TOOLS, add_generation_prompt=True)
    output1 = await generate_sglang(
        client=client,
        base_url=base_url,
        renderer=renderer,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    completion1 = completion_ids(output1, prompt_ids)
    parsed1 = renderer.parse_response(completion1)
    print_parsed(label, "turn 1", parsed1)

    assistant: dict[str, Any] = {"role": "assistant", "content": parsed1.content}
    if parsed1.reasoning_content:
        assistant["reasoning_content"] = parsed1.reasoning_content
    if parsed1.tool_calls:
        # Convert the parsed dataclasses back to OpenAI-format tool_calls.
        assistant["tool_calls"] = [
            {
                "id": tc.id or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments
                    if isinstance(tc.arguments, str)
                    else json.dumps(tc.arguments),
                },
            }
            for idx, tc in enumerate(parsed1.tool_calls)
        ]
    messages.append(assistant)

    if parsed1.tool_calls:
        new_messages: list[dict[str, Any]] = []
        for idx, tool_call in enumerate(parsed1.tool_calls):
            tool_args = tool_call.arguments or {}
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id or f"call_{idx}",
                    "name": tool_call.name or "multiply",
                    "content": json.dumps(
                        {"result": int(tool_args["a"]) * int(tool_args["b"])}
                    ),
                }
            )
    else:
        new_messages = [
            {"role": "user", "content": "Give the final answer in one sentence."}
        ]

    # Turn 2: bridge extends prompt_ids + completion1 exactly.
    # ``bridge_to_next_turn`` returns a ``RenderedTokens`` (or None); the
    # extended id stream is on ``.token_ids``.
    bridged = renderer.bridge_to_next_turn(
        prompt_ids, completion1, new_messages, tools=TOOLS
    )
    if bridged is None:
        raise RuntimeError("bridge_to_next_turn returned None")
    bridged_ids = bridged.token_ids
    assert bridged_ids[: len(prompt_ids) + len(completion1)] == (
        prompt_ids + completion1
    )

    output2 = await generate_sglang(
        client=client,
        base_url=base_url,
        renderer=renderer,
        prompt_ids=bridged_ids,
        max_new_tokens=max_new_tokens,
    )
    completion2 = completion_ids(output2, bridged_ids)
    print_parsed(label, "turn 2", renderer.parse_response(completion2))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://localhost:30000",
        help="SGLang HTTP server base URL.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="Must match the model the SGLang server is serving.",
    )
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false", "both"],
        default="both",
        help="Qwen3.5 thinking mode. Ignored for gpt-oss.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.model.startswith("Qwen/Qwen3.5-"):
        if args.enable_thinking == "both":
            modes: list[bool | None] = [True, False]
        else:
            modes = [args.enable_thinking == "true"]
    else:
        modes = [None]

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for mode in modes:
            await run_one(
                client=client,
                base_url=args.base_url,
                model=args.model,
                enable_thinking=mode,
                max_new_tokens=args.max_new_tokens,
            )


if __name__ == "__main__":
    asyncio.run(main())
