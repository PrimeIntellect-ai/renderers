#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "renderers>=0.1.6",
#   "transformers>=4.50.0",
#   "accelerate",
#   "torch",
#   "kernels>=0.12.0",
#   "openai-harmony>=0.0.8",
#   "openai>=1.108.1",
#   "tiktoken",
#   "jinja2",
#   "numpy",
# ]
# ///
"""Transformers generation from renderer-owned prompt token IDs."""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from renderers.gpt_oss import GptOssRenderer
from renderers.qwen35 import Qwen35Renderer


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
        return Qwen35Renderer(tokenizer, enable_thinking=enable_thinking), tokenizer
    if model == "openai/gpt-oss-20b":
        return GptOssRenderer(tokenizer), tokenizer
    raise ValueError(f"unsupported demo model: {model}")


def print_parsed(label: str, turn: str, parsed) -> None:
    print(f"\n[{label}] {turn}")
    if parsed.reasoning_content:
        print(f"reasoning: {parsed.reasoning_content[:240]}")
    if parsed.tool_calls:
        print(f"tool_calls: {json.dumps(parsed.tool_calls, ensure_ascii=False)}")
    if parsed.content:
        print(f"content: {parsed.content}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This example expects a CUDA GPU.")

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

        renderer, tokenizer = make_renderer(model, enable_thinking)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=torch.bfloat16,
            trust_remote_code=False,
        ).to("cuda")
        hf_model.eval()

        stop_token_ids = renderer.get_stop_token_ids()
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        messages = [
            {"role": "system", "content": "You are a concise tool-using assistant."},
            {
                "role": "user",
                "content": "Use the multiply tool for 17 * 23, then summarize.",
            },
        ]

        # Turn 1: render locally and pass token IDs to Transformers. The model
        # receives input_ids, not messages or a chat template.
        prompt_ids = renderer.render_ids(
            messages, tools=TOOLS, add_generation_prompt=True
        )
        input_ids = torch.tensor([prompt_ids], device="cuda")
        attention_mask = torch.ones_like(input_ids)
        output1 = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=stop_token_ids,
            pad_token_id=pad_token_id,
        )[0]
        completion1 = output1[input_ids.shape[-1] :].tolist()
        parsed1 = renderer.parse_response(completion1)
        print_parsed(label, "turn 1", parsed1)

        assistant = {"role": "assistant", "content": parsed1.content}
        if parsed1.reasoning_content:
            assistant["reasoning_content"] = parsed1.reasoning_content
        if parsed1.tool_calls:
            assistant["tool_calls"] = parsed1.tool_calls
        messages.append(assistant)

        if parsed1.tool_calls:
            new_messages = []
            for idx, tool_call in enumerate(parsed1.tool_calls):
                fn = tool_call.get("function") or tool_call
                tool_args = fn.get("arguments") or {}
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"call_{idx}"),
                        "name": fn.get("name", "multiply"),
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
        bridged_ids = renderer.bridge_to_next_turn(
            prompt_ids, completion1, new_messages, tools=TOOLS
        )
        if bridged_ids is None:
            raise RuntimeError("bridge_to_next_turn returned None")
        assert bridged_ids[: len(prompt_ids) + len(completion1)] == (
            prompt_ids + completion1
        )

        bridged_input_ids = torch.tensor([bridged_ids], device="cuda")
        bridged_attention_mask = torch.ones_like(bridged_input_ids)
        output2 = hf_model.generate(
            input_ids=bridged_input_ids,
            attention_mask=bridged_attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=stop_token_ids,
            pad_token_id=pad_token_id,
        )[0]
        completion2 = output2[bridged_input_ids.shape[-1] :].tolist()
        print_parsed(label, "turn 2", renderer.parse_response(completion2))

        del hf_model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
