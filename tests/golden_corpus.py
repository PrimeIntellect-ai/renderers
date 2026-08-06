"""Deterministic, reviewable behavior snapshots for every renderer.

The corpus keeps rendered text in plain sight so formatting changes are easy
to review, while a token-id digest catches tokenizer-level changes that decode
to the same text. Attribution arrays are run-length encoded to keep the JSON
compact. Upstream tokenizer inputs are pinned separately in ``model_assets``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, cast

from renderers import RendererConfig, RenderedTokens, config_from_name, create_renderer
from renderers.base import ParsedResponse, trim_to_turn_close
from tests.model_assets import load_test_tokenizer, model_revision


@dataclass(frozen=True)
class GoldenCase:
    slug: str
    renderer_name: str
    model_name: str
    config_overrides: tuple[tuple[str, Any], ...] = ()


# One representative for every concrete entry in RENDERER_REGISTRY. Keep the
# explicit renderer name even when auto-resolution would choose the same class:
# the corpus is a registry contract, not a model-routing test.
GOLDEN_CASES = (
    GoldenCase("default", "default", "Qwen/Qwen2.5-0.5B-Instruct"),
    GoldenCase("qwen3", "qwen3", "Qwen/Qwen3-8B"),
    GoldenCase("prime-qwen3", "prime-qwen3", "PrimeIntellect/Qwen3-0.6B"),
    GoldenCase("qwen3-vl", "qwen3-vl", "Qwen/Qwen3-VL-4B-Instruct"),
    GoldenCase("qwen3.5", "qwen3.5", "Qwen/Qwen3.5-9B"),
    GoldenCase("qwen3.6", "qwen3.6", "Qwen/Qwen3.6-35B-A3B"),
    GoldenCase("glm-5", "glm-5", "zai-org/GLM-5"),
    GoldenCase("glm-5.1", "glm-5.1", "zai-org/GLM-5.1"),
    GoldenCase("glm-4.5", "glm-4.5", "THUDM/GLM-4.5-Air"),
    GoldenCase("minimax-m2", "minimax-m2", "MiniMaxAI/MiniMax-M2.5"),
    GoldenCase("deepseek-v3", "deepseek-v3", "deepseek-ai/DeepSeek-V3"),
    GoldenCase("deepseek-r1", "deepseek-r1", "deepseek-ai/DeepSeek-R1"),
    GoldenCase("hy3", "hy3", "tencent/Hy3"),
    GoldenCase("kimi-k2", "kimi-k2", "moonshotai/Kimi-K2-Instruct"),
    GoldenCase("kimi-k2.5", "kimi-k2.5", "moonshotai/Kimi-K2.5"),
    GoldenCase("laguna-xs.2", "laguna-xs.2", "poolside/Laguna-XS.2"),
    GoldenCase("laguna-xs-2.1", "laguna-xs-2.1", "poolside/Laguna-XS-2.1"),
    GoldenCase("llama-3", "llama-3", "meta-llama/Llama-3.2-1B-Instruct"),
    GoldenCase(
        "nemotron-3",
        "nemotron-3",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ),
    GoldenCase(
        "nemotron-3-ultra",
        "nemotron-3-ultra",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    ),
    GoldenCase(
        "gpt-oss",
        "gpt-oss",
        "openai/gpt-oss-20b",
        (("conversation_start_date", "2025-01-15"),),
    ),
)


SYSTEM_AND_USER = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is 2+2?"},
]

ASSISTANT = {
    "role": "assistant",
    "reasoning_content": "Two plus two equals four.",
    "content": "Four.",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# Some renderers expose a template knob that must agree with an explicit
# ``thinking_retention="all"`` bridge policy. Keep the bridge probe on valid,
# user-constructible configs instead of bypassing Pydantic validation.
_BRIDGE_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "qwen3.6": {"preserve_thinking": True},
    "glm-5": {"clear_thinking": False},
    "glm-5.1": {"clear_thinking": False},
    "hy3": {"preserved_thinking": True},
    "nemotron-3": {"truncate_history_thinking": False},
    "nemotron-3-ultra": {"truncate_history_thinking": False},
    "gpt-oss": {"auto_drop_analysis": False},
}


def _sha256_ids(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runs(values: list[Any]) -> list[list[Any]]:
    if not values:
        return []
    out: list[list[Any]] = []
    value = values[0]
    count = 1
    for current in values[1:]:
        if current == value:
            count += 1
            continue
        out.append([value, count])
        value = current
        count = 1
    out.append([value, count])
    return out


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _token_snapshot(tokenizer, token_ids: list[int]) -> dict[str, Any]:
    return {
        "text": _decode(tokenizer, token_ids),
        "token_count": len(token_ids),
        "token_ids_sha256": _sha256_ids(token_ids),
    }


def _render_snapshot(tokenizer, rendered: RenderedTokens) -> dict[str, Any]:
    return {
        **_token_snapshot(tokenizer, rendered.token_ids),
        "message_indices_rle": _runs(rendered.message_indices),
        "sampled_mask_rle": _runs(rendered.sampled_mask),
        "is_content_rle": _runs(rendered.is_content),
        "message_roles": rendered.message_roles,
        "message_tool_names": rendered.message_tool_names,
    }


def _parsed_snapshot(parsed: ParsedResponse) -> dict[str, Any]:
    return {
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": [asdict(tool_call) for tool_call in parsed.tool_calls],
    }


def _renderer_for(case: GoldenCase, tokenizer, *, bridge: bool = False):
    config = config_from_name(case.renderer_name)
    assert config is not None
    config = cast(RendererConfig, config)
    config_data = config.model_dump(mode="python")
    config_data.update(case.config_overrides)
    if bridge and case.renderer_name != "default":
        config_data["thinking_retention"] = "all"
        config_data.update(_BRIDGE_CONFIG_OVERRIDES.get(case.renderer_name, {}))
    config = cast(RendererConfig, type(config).model_validate(config_data))
    return create_renderer(tokenizer, config)


def _completion_ids(
    case: GoldenCase, renderer, generation_prompt: RenderedTokens
) -> tuple[RenderedTokens, list[int]]:
    completed = renderer.render(SYSTEM_AND_USER + [ASSISTANT], tools=TOOLS)
    if case.renderer_name == "gpt-oss":
        # Harmony's parser consumes channel blocks beginning with <|start|>.
        # A historical final answer is the smallest deterministic render that
        # carries that header; sampled-mask extraction deliberately omits it.
        history = renderer.render(SYSTEM_AND_USER, tools=TOOLS)
        completion_ids = completed.token_ids[len(history.token_ids) :]
    elif completed.sampled_mask:
        completion_ids = [
            token_id
            for token_id, sampled in zip(
                completed.token_ids, completed.sampled_mask, strict=True
            )
            if sampled
        ]
    else:
        prefix_length = len(generation_prompt.token_ids)
        if completed.token_ids[:prefix_length] != generation_prompt.token_ids:
            raise AssertionError(
                "opaque renderer's completed turn does not extend its generation prompt"
            )
        completion_ids = completed.token_ids[prefix_length:]
    return completed, completion_ids


def _bridge_snapshot(case: GoldenCase, tokenizer) -> dict[str, Any] | None:
    if case.renderer_name == "default":
        return None

    renderer = _renderer_for(case, tokenizer, bridge=True)
    previous_prompt = renderer.render(
        SYSTEM_AND_USER, add_generation_prompt=True
    ).token_ids
    previous_turn = renderer.render(
        SYSTEM_AND_USER + [{"role": "assistant", "content": "Prior answer."}]
    ).token_ids
    previous_completion = previous_turn[len(previous_prompt) :]
    previous = trim_to_turn_close(
        previous_prompt,
        previous_completion,
        set(renderer.get_stop_token_ids()),
    )
    if previous is not None:
        previous_completion = previous[len(previous_prompt) :]

    bridged = renderer.bridge_to_next_turn(
        previous_prompt,
        previous_completion,
        [{"role": "user", "content": "And 3+3?"}],
    )
    if bridged is None:
        raise AssertionError(f"{case.slug}: hand-coded renderer declined clean bridge")

    prior_length = len(previous_prompt) + len(previous_completion)
    prior = previous_prompt + previous_completion
    if bridged.token_ids[:prior_length] != prior:
        raise AssertionError(f"{case.slug}: bridge did not preserve its token prefix")
    snapshot = _render_snapshot(tokenizer, bridged)
    snapshot["config"] = renderer.config.model_dump(mode="json")
    snapshot["extension"] = _token_snapshot(tokenizer, bridged.token_ids[prior_length:])
    return snapshot


def build_golden_case(case: GoldenCase) -> dict[str, Any]:
    """Render all public behavior probes for one registered renderer."""
    tokenizer = load_test_tokenizer(case.model_name)
    renderer = _renderer_for(case, tokenizer)

    generation_prompt = renderer.render(
        SYSTEM_AND_USER,
        tools=TOOLS,
        add_generation_prompt=True,
    )
    completed, completion_ids = _completion_ids(case, renderer, generation_prompt)
    parsed = renderer.parse_response(completion_ids, tools=TOOLS)

    return {
        "renderer": case.renderer_name,
        "renderer_class": type(renderer).__name__,
        "model": case.model_name,
        "model_revision": model_revision(case.model_name),
        "config": renderer.config.model_dump(mode="json"),
        "generation_prompt": _render_snapshot(tokenizer, generation_prompt),
        "completed_turn": _render_snapshot(tokenizer, completed),
        "parser_input": _token_snapshot(tokenizer, completion_ids),
        "parsed_completion": _parsed_snapshot(parsed),
        "bridge": _bridge_snapshot(case, tokenizer),
    }
