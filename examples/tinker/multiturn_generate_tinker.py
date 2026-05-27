#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "renderers>=0.1.6",
#   "tinker>=0.9.0",
#   "transformers>=4.57.6,<=5.5.3",
#   "openai-harmony>=0.0.8",
#   "openai>=1.108.1",
#   "tiktoken",
#   "jinja2",
#   "numpy",
# ]
# ///
"""Tinker remote sampling from renderer-owned prompt token IDs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import tinker
from renderers.configs import Qwen35RendererConfig
from renderers.gpt_oss import GptOssRenderer
from renderers.qwen35 import Qwen35Renderer
from tinker import types
from transformers import AutoTokenizer


MODELS = ["Qwen/Qwen3.5-4B", "openai/gpt-oss-20b"]
QWEN_THINKING_MODES = [True, False]

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


def make_renderer(model: str, enable_thinking: bool | None):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    if model.startswith("Qwen/Qwen3.5-"):
        return Qwen35Renderer(
            tokenizer, Qwen35RendererConfig(enable_thinking=enable_thinking)
        )
    if model == "openai/gpt-oss-20b":
        return GptOssRenderer(tokenizer)
    raise ValueError(f"unsupported demo model: {model}")


def print_parsed(label: str, turn: str, parsed) -> None:
    print(f"\n[{label}] {turn}")
    if parsed.reasoning_content:
        print(f"reasoning: {parsed.reasoning_content[:240]}")
    for tc in parsed.tool_calls:
        # ``parse_response`` returns ``ParsedToolCall`` dataclasses, not dicts.
        print(f"tool_call: {tc.name}({tc.arguments}) [{tc.status.value}]")
    if parsed.content:
        print(f"content: {parsed.content}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("Set TINKER_API_KEY before running this example.")

    service_client = tinker.ServiceClient(base_url=args.base_url)
    targets = []
    for model in MODELS:
        if model.startswith("Qwen/Qwen3.5-"):
            for enable_thinking in QWEN_THINKING_MODES:
                targets.append((model, enable_thinking))
        else:
            targets.append((model, None))

    for model, enable_thinking in targets:
        label = (
            model
            if enable_thinking is None
            else f"{model} enable_thinking={enable_thinking}"
        )
        print(f"\n=== {label} ===")

        renderer = make_renderer(model, enable_thinking)
        sampling_client = await service_client.create_sampling_client_async(
            base_model=model
        )
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            stop=renderer.get_stop_token_ids(),
        )

        messages = [
            {"role": "system", "content": "You are a concise tool-using assistant."},
            {
                "role": "user",
                "content": "Use the multiply tool for 17 * 23, then summarize.",
            },
        ]

        # Turn 1: render locally and pass token IDs to Tinker. Tinker receives
        # a ModelInput, not messages or a chat template.
        prompt_ids = renderer.render_ids(
            messages, tools=TOOLS, add_generation_prompt=True
        )
        output1 = await sampling_client.sample_async(
            prompt=types.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=sampling_params,
        )
        completion1 = list(output1.sequences[0].tokens)
        parsed1 = renderer.parse_response(completion1)
        print_parsed(label, "turn 1", parsed1)

        assistant = {"role": "assistant", "content": parsed1.content}
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
            new_messages = []
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

        output2 = await sampling_client.sample_async(
            prompt=types.ModelInput.from_ints(bridged_ids),
            num_samples=1,
            sampling_params=sampling_params,
        )
        completion2 = list(output2.sequences[0].tokens)
        print_parsed(label, "turn 2", renderer.parse_response(completion2))


if __name__ == "__main__":
    asyncio.run(main())
