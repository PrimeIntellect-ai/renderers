"""Actual vLLM public parsers, isolated from portable tests (Linux runtime).

This module never substitutes parser classes, tokenizer IDs, or imports.
Streaming uses vLLM's own incremental detokenizer, including byte fragments.
"""

from __future__ import annotations

import importlib.metadata
import json

from tests.conformance.export_vllm import VLLM_VERSION

# (tool parser, reasoning parser), using serving CLI names, not private engines.
PARSERS = {
    "default": (None, None),
    "qwen3": ("hermes", "qwen3"),
    "qwen3-vl": ("hermes", "qwen3"),
    "prime-qwen3": ("qwen3_xml", "qwen3"),
    "qwen3.5": ("qwen3_xml", "qwen3"),
    "qwen3.6": ("qwen3_xml", "qwen3"),
    "qwen3.8": ("qwen3_xml", "qwen3"),
    "glm-4.5": ("glm47", "glm47"),
    "glm-5": ("glm47", "glm47"),
    "glm-5.1": ("glm47", "glm47"),
    "glm-5.3": ("glm47", "glm47"),
    "deepseek-v3": ("deepseek_v3", None),
    "deepseek-r1": ("deepseek_v3", "deepseek_r1"),
    "deepseek-v4": ("deepseek_v4", "deepseek_v4"),
    "minimax-m2": ("minimax_m2", "minimax_m2"),
    "kimi-k2": ("kimi_k2", "kimi_k2"),
    "kimi-k2.5": ("kimi_k2", "kimi_k2"),
    "nemotron-3": ("qwen3_xml", "nemotron_v3"),
    "nemotron-3-ultra": ("qwen3_xml", "nemotron_v3"),
    "nemotron-3.5": ("qwen3_xml", "nemotron_v3"),
    "gemma4": ("gemma4", "gemma4"),
    "hy3": ("hy_v3", "hy_v3"),
    "laguna-xs.2": ("poolside_v1", "poolside_v1"),
    "laguna-m.1": ("poolside_v1", "poolside_v1"),
    "laguna-xs-2.1": ("poolside_v1", "poolside_v1"),
    "laguna-s-2.1": ("poolside_v1", "poolside_v1"),
    "llama-3": ("llama3_json", None),
    "gpt-oss": ("openai", "openai_gptoss"),
    "inkling": ("inkling", "inkling"),
}


def arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value  # Keep unfinished argument bytes visible in the report.
    return value


def projection(reasoning, content, calls):
    """OpenAI projection; no whitespace stripping or argument coercion."""
    return {
        "reasoning": reasoning or None,
        "content": content or "",
        "tool_calls": [
            {"name": call["name"], "arguments": arguments(call["arguments"])}
            for call in calls or []
        ],
    }


def same_projection(left, right):
    # Python equality conflates True with 1 (and 1 with 1.0). JSON preserves
    # argument types while ignoring object key order.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def parse_reference(case, tok, prompt, completion, tools, kwargs):
    assert importlib.metadata.version("vllm") == VLLM_VERSION
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager
    from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

    # Renderer config None means "defer to the template default", exactly as
    # reference() treats it. Passing a literal None to a vLLM parser instead
    # would disable thinking without changing the actual generation prompt.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if case.model == "moonshotai/Kimi-K2-Instruct":
        # The Instruct checkpoint's official prefill opens content, unlike
        # the K2.5/K2.6 thinking checkpoint. This is a serving profile choice.
        kwargs = dict(kwargs, thinking=False)

    tool_name, reasoning_name = PARSERS[case.resolved_renderer]
    if case.resolved_renderer == "default" and tools:
        tool_name = "hermes"  # Explicit DefaultRenderer tool_parser="qwen3".
    cls = ParserManager.get_parser(
        tool_parser_name=tool_name,
        reasoning_parser_name=reasoning_name,
        enable_auto_tools=True,
        model_name=case.model,
        is_harmony=case.resolved_renderer == "gpt-oss",
    )
    text = tok.decode(completion, skip_special_tokens=False)
    if cls is None:
        return {"complete": projection(None, text, []), "stream": {}}
    request = ChatCompletionRequest(
        model=case.model,
        messages=[{"role": "user", "content": "test"}],
        tools=tools or None,
        chat_template_kwargs=kwargs or None,
        skip_special_tokens=False,
    )

    def parser():
        return cls(tok, tools=request.tools, chat_template_kwargs=kwargs)

    token_texts = []
    previous = None
    prefix_offset = read_offset = 0
    for i in range(len(completion)):
        added, delta, prefix_offset, read_offset = detokenize_incrementally(
            tok,
            prompt + completion[: i + 1],
            previous,
            prefix_offset,
            read_offset,
            skip_special_tokens=False,
        )
        previous = (previous or []) + added
        token_texts.append(delta)
    serving_text = "".join(token_texts)
    outputs = {
        "complete": None,
        "stream": {},
        "stream_errors": {},
        "detokenized_text": serving_text,
        "decode_agrees": serving_text == text,
        "parser_kwargs": kwargs,
    }
    try:
        reason, content, calls = parser().parse(
            serving_text,
            request,
            enable_auto_tools=True,
            model_output_token_ids=completion,
        )
        outputs["complete"] = projection(
            reason, content, [c.model_dump() for c in calls or []]
        )
    except Exception as exc:
        outputs["complete_error"] = f"{type(exc).__name__}: {exc}"

    def stream(chunk):
        instance = parser()
        reason_parts, content_parts, tool_parts = [], [], {}
        for start in range(0, len(completion), chunk):
            end = min(start + chunk, len(completion))
            delta = instance.parse_delta(
                "".join(token_texts[start:end]),
                completion[start:end],
                request,
                prompt_token_ids=prompt,
                finished=end == len(completion),
            )
            if delta is None:
                continue
            reason_parts.append(delta.reasoning or "")
            content_parts.append(delta.content or "")
            for tc in delta.tool_calls or []:
                fn = tc.function
                record = tool_parts.setdefault(tc.index, {"name": "", "arguments": ""})
                if fn:
                    record["name"] += fn.name or ""
                    record["arguments"] += fn.arguments or ""
        return projection(
            "".join(reason_parts),
            "".join(content_parts),
            [tool_parts[i] for i in sorted(tool_parts)],
        )

    for chunk in (1, 2, 5, len(completion) or 1):
        try:
            outputs["stream"][str(chunk)] = stream(chunk)
        except Exception as exc:
            outputs["stream_errors"][str(chunk)] = f"{type(exc).__name__}: {exc}"
    return outputs
