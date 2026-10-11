"""Execute the reviewed, pinned model-supplied Python encoder directly."""

import copy
import hashlib
import importlib.util
import json
import tempfile
from functools import cache
from pathlib import Path

from huggingface_hub import hf_hub_download

DEEPSEEK_ENCODER = {
    "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    "file": "encoding/encoding_dsv4.py",
    "sha256": "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6",
}


@cache
def encoder():
    source = DEEPSEEK_ENCODER
    path = hf_hub_download(
        source["repo"],
        source["file"],
        revision=source["revision"],
        cache_dir=Path(tempfile.gettempdir()) / "renderers-reference-encoders",
    )
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == source["sha256"]
    spec = importlib.util.spec_from_file_location("pinned_deepseek_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deepseek_reference(
    tok,
    messages,
    *,
    tools=None,
    add_generation_prompt=False,
    enable_thinking=False,
    drop_thinking=True,
    reasoning_effort="low",
):
    upstream = encoder()
    messages = copy.deepcopy(messages)
    if tools:
        target = next((m for m in messages if m["role"] == "developer"), None)
        if target is None:
            target = next((m for m in messages if m["role"] == "system"), None)
        if target is None:
            target = {"role": "system", "content": ""}
            messages.insert(0, target)
        target["tools"] = tools
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if not isinstance(function["arguments"], str):
                function["arguments"] = json.dumps(
                    function["arguments"], ensure_ascii=False
                )
    text = upstream.encode_messages(
        messages,
        thinking_mode="thinking" if enable_thinking else "chat",
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
    )
    # Upstream's API always opens generation after a terminal query. The
    # renderer API also exposes a completed-transcript view. Remove only that
    # terminal generation prefill; all internal transitions remain upstream's.
    if not add_generation_prompt:
        merged = upstream.merge_tool_messages(messages)
        if (
            merged
            and merged[-1]["role"] in {"user", "developer"}
            and not merged[-1].get("task")
        ):
            suffix = upstream.ASSISTANT_SP_TOKEN + (
                upstream.thinking_start_token
                if enable_thinking
                else upstream.thinking_end_token
            )
            assert text.endswith(suffix)
            text = text[: -len(suffix)]
    return tok.encode(text, add_special_tokens=False)
