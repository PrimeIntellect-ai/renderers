#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "renderers>=0.1.6",
#   "vllm>=0.20",
#   "transformers>=4.50.0",
#   "openai-harmony>=0.0.8",
#   "openai>=1.108.1",
#   "tiktoken",
#   "jinja2",
#   "numpy",
# ]
# ///
"""vLLM offline generation from renderer-owned prompt token IDs."""

from __future__ import annotations

import argparse
import gc
import json
import os

from renderers.configs import Qwen35RendererConfig
from renderers.gpt_oss import GptOssRenderer
from renderers.qwen35 import Qwen35Renderer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

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
        llm = LLM(
            model=model,
            tokenizer=model,
            trust_remote_code=False,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
        )

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            stop_token_ids=renderer.get_stop_token_ids(),
            skip_special_tokens=False,
        )

        messages = [
            {"role": "system", "content": "You are a concise tool-using assistant."},
            {
                "role": "user",
                "content": "Use the multiply tool for 17 * 23, then summarize.",
            },
        ]

        # Turn 1: render locally and pass token IDs to vLLM. vLLM never sees
        # messages and never applies a chat template.
        prompt_ids = renderer.render_ids(
            messages, tools=TOOLS, add_generation_prompt=True
        )
        output1 = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling_params=sampling,
            use_tqdm=False,
        )[0]
        completion1 = list(output1.outputs[0].token_ids)
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

        output2 = llm.generate(
            [{"prompt_token_ids": bridged_ids}],
            sampling_params=sampling,
            use_tqdm=False,
        )[0]
        completion2 = list(output2.outputs[0].token_ids)
        print_parsed(label, "turn 2", renderer.parse_response(completion2))

        del llm
        gc.collect()


if __name__ == "__main__":
    main()
